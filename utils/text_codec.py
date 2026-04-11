import torch


def encode_label(label: str, vocab: str) -> torch.Tensor:
    return torch.tensor([vocab.index(c) for c in label.upper()], dtype=torch.long)


def decode_pred(logits: torch.Tensor, vocab: str):
    indices = logits.argmax(dim=2)
    return ["".join(vocab[i] for i in seq.tolist()) for seq in indices]
