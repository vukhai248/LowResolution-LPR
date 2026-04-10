import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class STNBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((4, 8))
        )
        self.fc_loc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 8, 128),
            nn.ReLU(True),
            nn.Linear(128, 6)
        )
        with torch.no_grad():
            self.fc_loc[-1].weight.zero_()
            self.fc_loc[-1].bias.copy_(
                torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
            )
    
    def forward(self, x):
        x = self.localization(x)
        x = self.fc_loc(x)
        x = x.view(-1, 2, 3)
        return x
    

class AttentionFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Conv2d(channels, channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, 1, kernel_size=1)
        )

    def forward(self, x):
        total_frames, C, H, W = x.size()
        num_frames = 5
        batch_size = total_frames // num_frames

        # Reshape to [Batch_size, Frames, C, H, W]
        x_view = x.view(batch_size, num_frames, C, H, W)

        # Calculate attention scores [Batch_size, Frames, 1, H, W]
        scores = self.score_net(x).view(batch_size, num_frames, 1, H, W)
        weights = F.softmax(scores, dim=1)

        # Weight sim fusion
        fused_features = torch.sum(x_view * weights, dim=1)
        return fused_features


class FeatureExtractor(nn.Module):
    def __init__(self, pretrained=True, out_dim=512, freeze_backbone=False):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)

        self.conv1   = backbone.conv1
        self.bn1     = backbone.bn1
        self.relu    = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1  = backbone.layer1
        self.layer2  = backbone.layer2
        self.layer3  = backbone.layer3
        self.layer4  = backbone.layer4

        # Modify stride (2,2) -> (2,1) in layer3 và layer4, keep w, shrink h only

        self.layer3[0].conv2.stride = (2, 1)
        self.layer3[0].downsample[0].stride = (2, 1)

        self.layer4[0].conv2.stride = (2, 1)
        self.layer4[0].downsample[0].stride = (2, 1)

        # Project 2048 -> out_dim
        self.proj = nn.Conv2d(2048, out_dim, kernel_size=1)

        if freeze_backbone:
            for name, p in self.named_parameters():
                if 'proj' not in name:
                    p.requires_grad = False

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.proj(x)                              # (B, out_dim, H', W')

        # Collapse height → 1, giữ width làm sequence
        x = F.adaptive_avg_pool2d(x, (1, None))       # (B, out_dim, 1, W')
        return x
    


class pos_encoding(nn.Module):
    def __init__(self, embed_dim, max_length=500):
        super().__init__()
        self.pos_encoding = nn.Parameter(
            (embed_dim ** -0.5) * torch.randn(1, max_length, embed_dim)
        )

    def forward(self, x):
        B, T, C = x.shape
        return x + self.pos_encoding[:, :T, :]
    


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, ff_dim, num_heads, drop_out):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=drop_out, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.drop1 = nn.Dropout(drop_out)
        self.drop2 = nn.Dropout(drop_out)

    def forward(self, q, k, v):
        q_norm = self.layernorm1(q)
        k_norm = self.layernorm1(k)
        v_norm = self.layernorm1(v)

        attn_out, _ = self.attn(q_norm, k_norm, v_norm)
        attn_out = self.drop1(attn_out)
        out1 = attn_out + q

        out1_norm = self.layernorm2(out1)
        ff_out = self.ff(out1_norm)
        ff_out = self.drop2(ff_out)
        out2 = ff_out + out1

        return out2
    


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, ff_dim, num_layers, num_heads, drop_out):
        super().__init__()

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim, ff_dim, num_heads, drop_out
            ) for _ in range(num_layers)
        ])

    def forward(self, x):
        output = x
        for block in self.blocks:
            output = block(output, output, output)
        return output