import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.append("data")
sys.path.append("models")

from dataset import ASVspoofDataset
from cnn_baseline import LogMelCNN
from evaluate import evaluate_model


def main():
    parser = argparse.ArgumentParser(description="Final, one-time evaluation on the EVAL split.")
    parser.add_argument("--eval_flac", required=True)
    parser.add_argument("--eval_protocol", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_dataset = ASVspoofDataset(args.eval_flac, args.eval_protocol, train=False)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Eval samples: {len(eval_dataset)}")

    model = LogMelCNN().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')} "
          f"(dev EER at save time: {checkpoint.get('best_dev_eer', float('nan')):.2f}%)")

    _, eer = evaluate_model(model, eval_loader, device)

    print(f"\nFinal EVAL EER: {eer:.2f}%")

    os.makedirs("reports", exist_ok=True)
    with open("reports/results.md", "a") as f:
        f.write(f"- Eval EER: {eer:.2f}%  (checkpoint: {args.checkpoint})\n")


if __name__ == "__main__":
    main()
