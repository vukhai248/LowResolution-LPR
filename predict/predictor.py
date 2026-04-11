import os

import torch
from tqdm.auto import tqdm

from utils import decode_pred


def predict_blind_test(model, test_loader, vocab: str, device="cpu", save_path="./test_predictions.csv", checkpoint_path=None):
    if checkpoint_path is not None:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            if any(k.startswith("module.") for k in state_dict.keys()):
                state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
            else:
                state_dict = {f"module.{k}": v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)

        print(f"Loaded model weights from: {checkpoint_path}")

    model.eval()
    results = []

    with torch.no_grad():
        for images, track_ids in tqdm(test_loader, desc="Predict", leave=False):
            images = images.to(device)
            logits = model(images)
            preds = decode_pred(logits.cpu(), vocab)

            for track_id, pred in zip(track_ids, preds):
                results.append({"track_id": track_id, "plate_text": pred})

    def _sort_key(row):
        track_id = row["track_id"]
        try:
            return int(track_id.split("_")[-1])
        except ValueError:
            return track_id

    results.sort(key=_sort_key)

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("track_id,plate_text\n")
        for row in results:
            f.write(f"{row['track_id']},{row['plate_text']}\n")

    print(f"Saved {len(results)} predictions to: {save_path}")
    return results
