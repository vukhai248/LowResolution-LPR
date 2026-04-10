import torch

class Config:

    
    path = '../data'
    seed = 42
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Dataset config
    vocab = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    num_classes = len(vocab)   # 36
    label_len = 7
    img_H, img_W = 32, 128

    # Model config
    embed_dim = 512
    ff_dim = 512 * 4
    num_layers = 3
    num_heads = 8
    drop_out = 0.1

    # Training config
    lr = 5e-4
    batch_size = 64
    epochs = 30
    log_interval = 1
    early_stop_count = 5
    warmup_epochs = 3
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    best_model_path = 'ResTranOCR.pth'
    