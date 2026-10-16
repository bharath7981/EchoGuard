import numpy as np
import torch
from sklearn.metrics import roc_curve


def compute_eer(bonafide_scores, spoof_scores):
    if len(bonafide_scores) == 0:
        raise ValueError("No bonafide scores found.")
    if len(spoof_scores) == 0:
        raise ValueError("No spoof scores found.")

    labels = np.concatenate([np.ones(len(bonafide_scores)), np.zeros(len(spoof_scores))])
    scores = np.concatenate([bonafide_scores, spoof_scores])

    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr

    index = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[index] + fnr[index]) / 2

    return eer * 100


def evaluate_model(model, loader, device, criterion=None):
    model.eval()

    total_loss = 0.0
    total_samples = 0
    bonafide_scores = []
    spoof_scores = []

    with torch.no_grad():
        for mel, labels in loader:
            mel = mel.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(mel)

            if criterion is not None:
                loss = criterion(logits, labels)
                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

            probabilities = torch.softmax(logits, dim=1)
            scores = probabilities[:, 1].cpu().numpy()  # index 1 = bonafide
            labels_np = labels.cpu().numpy()

            bonafide_scores.extend(scores[labels_np == 1])
            spoof_scores.extend(scores[labels_np == 0])

    average_loss = total_loss / total_samples if total_samples > 0 else 0
    eer = compute_eer(np.array(bonafide_scores), np.array(spoof_scores))

    return average_loss, eer


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    bonafide = rng.normal(0.8, 0.1, 1000).clip(0, 1)
    spoof = rng.normal(0.2, 0.1, 1000).clip(0, 1)

    eer = compute_eer(bonafide, spoof)
    print(f"Test EER: {eer:.2f}%")
