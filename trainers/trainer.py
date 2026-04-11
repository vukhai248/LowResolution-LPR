import torch
import torch.nn as nn
from tqdm.auto import tqdm

from config import Config
from visualization import visualize_val_samples


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=2)
    correct = (preds == targets).all(dim=1).sum().item()
    return correct / targets.size(0)


def train_model(
    model,
    optimizer,
    scheduler,
    criterion,
    train_loader,
    val_loader,
    epochs,
    early_stop_count,
    best_model_path,
    log_interval,
    vocab,
    warmup_epochs=0,
    device="cpu",
):
    best_val_acc = 0.0
    break_count = 0
    train_loss_hist, val_loss_hist = [], []
    train_acc_hist, val_acc_hist = [], []

    base_model = model.module if isinstance(model, nn.DataParallel) else model

    if warmup_epochs > 0:
        for p in base_model.extractor.parameters():
            p.requires_grad = False
        print(f"Warmup: freeze extractor for first {warmup_epochs} epochs")

    for epoch in tqdm(range(epochs), desc="Epoch", position=0):
        if warmup_epochs > 0 and epoch == warmup_epochs:
            for p in base_model.extractor.parameters():
                p.requires_grad = True
            print(f"Unfreeze extractor at epoch {epoch}")

        model.train()
        if warmup_epochs > 0 and epoch < warmup_epochs:
            base_model.extractor.eval()

        epoch_loss = 0.0
        epoch_acc = 0.0
        for images, targets in tqdm(train_loader, desc="Train", position=1, leave=False):
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits.permute(0, 2, 1), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += compute_accuracy(logits.detach().cpu(), targets.cpu())

        train_loss_hist.append(epoch_loss / len(train_loader))
        train_acc_hist.append(epoch_acc / len(train_loader))

        model.eval()
        epoch_loss = 0.0
        epoch_acc = 0.0
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Val", position=1, leave=False):
                images = images.to(device)
                targets = targets.to(device)

                logits = model(images)
                loss = criterion(logits.permute(0, 2, 1), targets)
                epoch_loss += loss.item()
                epoch_acc += compute_accuracy(logits.cpu(), targets.cpu())

        val_loss_hist.append(epoch_loss / len(val_loader))
        val_acc_hist.append(epoch_acc / len(val_loader))

        scheduler.step()

        if epoch % log_interval == 0:
            print(
                f"| Epoch {epoch:3d} "
                f"| Train Loss {train_loss_hist[-1]:.4f}  Acc {train_acc_hist[-1]:.4f} "
                f"| Val Loss {val_loss_hist[-1]:.4f}  Acc {val_acc_hist[-1]:.4f} |"
            )
            visualize_val_samples(model, val_loader, vocab=vocab, device=device, n_samples=2)

        if val_acc_hist[-1] > best_val_acc:
            break_count = 0
            best_val_acc = val_acc_hist[-1]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_acc": val_acc_hist[-1],
                    "break_count": break_count,
                    "train_loss_hist": train_loss_hist,
                    "val_loss_hist": val_loss_hist,
                    "train_acc_hist": train_acc_hist,
                    "val_acc_hist": val_acc_hist,
                },
                best_model_path,
            )
        else:
            break_count += 1

        if break_count >= early_stop_count:
            print(f"Early stop at epoch {epoch} | best val acc: {best_val_acc:.4f}")
            break

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("Training done, loaded best model")

    return model, {
        "train_loss": train_loss_hist,
        "val_loss": val_loss_hist,
        "train_acc": train_acc_hist,
        "val_acc": val_acc_hist,
    }


def build_training_components(model, lr: float, epochs: int):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    return criterion, optimizer, scheduler


def run_training(model, train_loader, val_loader, cfg):
    device = getattr(cfg, "device", "cpu")
    model = model.to(device)

    criterion, optimizer, scheduler = build_training_components(
        model=model,
        lr=cfg.lr,
        epochs=cfg.epochs,
    )

    trained_model, history = train_model(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg.epochs,
        early_stop_count=cfg.early_stop_count,
        best_model_path=cfg.best_model_path,
        log_interval=cfg.log_interval,
        vocab=cfg.vocab,
        warmup_epochs=cfg.warmup_epochs,
        device=device,
    )
    return trained_model, history
