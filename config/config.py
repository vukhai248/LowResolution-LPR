from dataclasses import dataclass, field
from typing import List
import torch


@dataclass
class Config:
    path: str = '../data'
    seed: int = 42
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Dataset config
    vocab: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    num_classes: int = field(init=False)
    label_len: int = 7
    img_H: int = 32
    img_W: int = 128

    # Model config
    embed_dim: int = 512
    ff_dim: int = 512 * 4
    num_layers: int = 3
    num_heads: int = 8
    drop_out: float = 0.1

    # Training config
    lr: float = 5e-4
    batch_size: int = 64
    epochs: int = 30
    log_interval: int = 1
    early_stop_count: int = 5
    warmup_epochs: int = 3
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)
    best_model_path: str = 'ResTranOCR.pth'

    def __post_init__(self):
        self.num_classes = len(self.vocab)
