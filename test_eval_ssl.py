import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append("data")
sys.path.append("models")

from embedding_dataset import EmbeddingDataset
from ssl_mlp import SSLClassifierMLP
from oc_softmax import OCSoftmaxLoss, SSLClassifierOC
from evaluate import compute_eer


def main():
    parser = argparse.ArgumentParser(description="Evaluate best SSL-MLP checkpoint on the cached EVAL embeddings.")
    parser.add_argument("--eval_embeddings", default="data/embeddings/eval.pt")
    parser.add_argument("--checkpoint", default="checkpoints/ssl_best_model.pt")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    print(f"Loading eval embeddings from {args.eval_embeddings}...")
    eval_dataset = EmbeddingDataset(args.eval_embeddings)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Eval samples: {len(eval_dataset)}")

    print(f"Loading checkpoint {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_type = checkpoint.get("model_type", "standard_mlp")

    if model_type == "oc_softmax":
        print("Detected OC-Softmax checkpoint architecture.")
        hidden_dim = checkpoint.get("hidden_dim", args.hidden_dim)
        model = SSLClassifierOC(in_features=1536, hidden_dim=hidden_dim).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        criterion = OCSoftmaxLoss(in_features=hidden_dim).to(device)
        criterion.load_state_dict(checkpoint["criterion_state_dict"])
        criterion.eval()
    else:
        print("Detected standard Cross-Entropy MLP checkpoint architecture.")
        model = SSLClassifierMLP(in_features=1536, hidden_dim=args.hidden_dim).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')} (dev EER: {checkpoint.get('best_dev_eer', float('nan')):.2f}%)")

    bonafide_scores = []
    spoof_scores = []

    with torch.no_grad():
        for embeddings, labels in eval_loader:
            embeddings = embeddings.to(device)
            if model_type == "oc_softmax":
                feats = model(embeddings)
                scores = criterion(feats).cpu().numpy()
            else:
                logits = model(embeddings)
                scores = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

            labels_np = labels.cpu().numpy()
            bonafide_scores.extend(scores[labels_np == 1])
            spoof_scores.extend(scores[labels_np == 0])

    eer = compute_eer(np.array(bonafide_scores), np.array(spoof_scores))

    print(f"\n==========================================")
    print(f" Phase 2 (SSL wav2vec2 + MLP) EVAL EER: {eer:.2f}%")
    print(f" Phase 1 (Log-Mel CNN) Baseline EER   : 6.16%")
    if eer < 6.16:
        print(f" Result: Outperformed CNN Baseline by {6.16 - eer:.2f}% absolute!")
    else:
        print(f" Result: Diff from CNN Baseline: {eer - 6.16:+.2f}%")
    print(f"==========================================\n")

    os.makedirs("reports", exist_ok=True)
    with open("reports/results.md", "a") as f:
        f.write(f"- Phase 2 (SSL wav2vec2-base + MLP) Eval EER: {eer:.2f}%  (checkpoint: {args.checkpoint})\n")
    print("Appended result to reports/results.md")


if __name__ == "__main__":
    main()
