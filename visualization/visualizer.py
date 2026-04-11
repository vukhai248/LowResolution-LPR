import matplotlib.pyplot as plt
import torch

from utils import decode_pred


def unnormalize(tensor: torch.Tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def visualize_val_samples(model, val_loader, vocab: str, device: str = "cpu", n_samples: int = 2):
    model.eval()
    images, targets = next(iter(val_loader))
    images_dev = images[:n_samples].to(device)

    with torch.no_grad():
        logits = model(images_dev)

    preds = decode_pred(logits.cpu(), vocab)
    gts = ["".join(vocab[i] for i in t.tolist()) for t in targets[:n_samples]]

    fig, axes = plt.subplots(n_samples, 1, figsize=(6, 2.5 * n_samples))
    if n_samples == 1:
        axes = [axes]

    for i in range(n_samples):
        ax = axes[i]
        img = unnormalize(images[i, 0].cpu())
        ax.imshow(img)
        ax.axis("off")

        correct = preds[i] == gts[i]
        color = "green" if correct else "red"
        ax.set_title(f"GT: {gts[i]}   |   Pred: {preds[i]}", fontsize=11, color=color, fontweight="bold")

    plt.suptitle("Val sample preview", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.show()


def visualize_predictions(model, loader, vocab: str, device: str = "cpu", n_samples: int = 6, mode: str = "test"):
    model.eval()
    batch = next(iter(loader))
    if mode == "test":
        images, targets, track_ids = batch
    else:
        images, track_ids = batch

    images = images.to(device)
    with torch.no_grad():
        logits = model(images)
        preds = decode_pred(logits.cpu(), vocab)

    images_cpu = images.detach().cpu()
    n_samples = min(n_samples, images_cpu.size(0))
    n_frames = images_cpu.size(1)

    fig, axes = plt.subplots(n_samples, n_frames, figsize=(n_frames * 3, n_samples * 2.5))
    if n_samples == 1:
        axes = [axes]

    for row_idx in range(n_samples):
        pred_label = preds[row_idx]
        track_id = track_ids[row_idx]

        if mode == "test":
            gt = "".join(vocab[i] for i in targets[row_idx].tolist())
            correct = pred_label == gt
            color = "#4CAF50" if correct else "#F44336"
            title = f"{track_id}\nGT: {gt}  |  Pred: {pred_label}"
        else:
            color = "#2196F3"
            title = f"{track_id}\nPred: {pred_label}"

        for col_idx in range(n_frames):
            ax = axes[row_idx][col_idx]
            img = unnormalize(images_cpu[row_idx, col_idx])
            ax.imshow(img)
            ax.axis("off")
            ax.set_xlabel(f"frame {col_idx + 1}", fontsize=7, labelpad=2)

            if col_idx == 0:
                ax.set_title(title, fontsize=9, fontweight="bold", color=color, loc="left", pad=3)

    plt.suptitle(f'{"Test" if mode == "test" else "Blind Test"} - Inference Preview', fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(f"inference_preview_{mode}.png", dpi=120, bbox_inches="tight")
    plt.show()
