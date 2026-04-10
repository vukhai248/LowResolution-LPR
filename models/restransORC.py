import torch
import torch.nn as nn
import torch.nn.functional as F
from .components import (
    AttentionFusion,
    FeatureExtractor,
    STNBlock,
    TransformerEncoder,
    pos_encoding,
)


class ResTranOCR(nn.Module):
    def __init__(self, label_len, num_classes, embed_dim, ff_dim, num_layers, num_heads,
                 extractor_pretrained=True, freeze_extractor=True, drop_out=0.1):
        super().__init__()
        self.label_len = label_len
        self.stn           = STNBlock(3)
        self.extractor     = FeatureExtractor(pretrained=extractor_pretrained,
                                      out_dim=embed_dim,
                                      freeze_backbone=freeze_extractor)
        self.attn_fusion   = AttentionFusion(embed_dim)
        self.pos_encoder   = pos_encoding(embed_dim, max_length=5000)
        self.transformer_layer = TransformerEncoder(
            embed_dim, ff_dim, num_layers, num_heads, drop_out
        )
        self.head = nn.Linear(embed_dim, num_classes)  # 36

    def forward(self, x):
        B, Frames, C, H, W = x.size()

        x_flat    = x.view(B * Frames, C, H, W)
        theta     = self.stn(x_flat)
        grid      = F.affine_grid(theta, x_flat.size(), align_corners=False)
        x_aligned = F.grid_sample(x_flat, grid, align_corners=False)

        features  = self.extractor(x_aligned)       # (B*F, embed_dim, H', W')

        fused     = self.attn_fusion(features)       # (B, embed_dim, H', W')

        seq_input = fused.squeeze(2).permute(0, 2, 1)  # (B, W', out_dim)
        seq_input = self.pos_encoder(seq_input)
        seq_out   = self.transformer_layer(seq_input)   # (B, T, embed_dim)

        # Pool T timestep → đúng 7 vị trí
        # AdaptiveAvgPool hoạt động trên dim T
        seq_out = seq_out.permute(0, 2, 1)              # (B, embed_dim, T)
        seq_out = F.adaptive_avg_pool1d(seq_out, self.label_len)  # (B, embed_dim, 7)
        seq_out = seq_out.permute(0, 2, 1)              # (B, 7, embed_dim)

        out = self.head(seq_out)                        # (B, 7, 36)
        return out                                                                           