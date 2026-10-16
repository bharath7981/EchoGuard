import argparse
import csv
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append("data")
sys.path.append("models")

from embedding_dataset import EmbeddingDataset
from ssl_mlp import SSLClassifierMLP
from evaluate import compute_eer


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_ssl(model, loader, device, criterion=None):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    bonafide_scores = []
    spoof_scores = []

    with torch.no_grad():
        for embeddings, labels in loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            logits = model(embeddings)

            if criterion is not None:
                loss = criterion(logits, labels)
                total_loss += loss.item() * labels.size(0)
                total_samples += labels.size(0)

            probabilities = torch.softmax(logits, dim=1)
            scores = probabilities[:, 1].cpu().numpy()  # index 1 = bonafide
            labels_np = labels.cpu().numpy()

            bonafide_scores.extend(scores[labels_np == 1])
            spoof_scores.extend(scores[labels_np == 0])

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    eer = compute_eer(np.array(bonafide_scores), np.array(spoof_scores))
    return avg_loss, eer


def train_ssl(args):
    set_seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    print(f"Loading training embeddings from {args.train_embeddings}...")
    train_dataset = EmbeddingDataset(args.train_embeddings)
    print(f"Loading dev embeddings from {args.dev_embeddings}...")
    dev_dataset = EmbeddingDataset(args.dev_embeddings)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Dev samples  : {len(dev_dataset)}")

    # Class weighting
    spoof_count = (train_dataset.labels == 0).sum().item()
    bonafide_count = (train_dataset.labels == 1).sum().item()
    total = spoof_count + bonafide_count
    spoof_weight = total / (2 * spoof_count)
    bonafide_weight = total / (2 * bonafide_count)
    class_weights = torch.tensor([spoof_weight, bonafide_weight], dtype=torch.float32).to(device)
    print(f"Bonafide samples : {bonafide_count}")
    print(f"Spoof samples    : {spoof_count}")
    print(f"Class weights    : [{spoof_weight:.4f}, {bonafide_weight:.4f}]")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)

    model = SSLClassifierMLP(in_features=1536, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6
    )

    history_path = os.path.join(args.checkpoint_dir, "ssl_training_history.csv")
    with open(history_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_eer", "dev_loss", "dev_eer", "learning_rate"])

    best_dev_eer = float("inf")
    patience_counter = 0

    print("\nStarting SSL-MLP Training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        train_bonafide_scores = []
        train_spoof_scores = []

        for embeddings, labels in train_loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(embeddings)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            lbls = labels.detach().cpu().numpy()
            train_bonafide_scores.extend(probs[lbls == 1])
            train_spoof_scores.extend(probs[lbls == 0])

        train_loss = total_loss / total_samples
        train_eer = compute_eer(np.array(train_bonafide_scores), np.array(train_spoof_scores))

        dev_loss, dev_eer = evaluate_ssl(model, dev_loader, device, criterion)
        scheduler.step(dev_eer)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\nEpoch {epoch:02d}")
        print(f"Train Loss   : {train_loss:.4f}")
        print(f"Train EER    : {train_eer:.2f}%")
        print(f"Dev Loss     : {dev_loss:.4f}")
        print(f"Dev EER      : {dev_eer:.2f}%")
        print(f"Learning Rate: {current_lr:.7f}")

        with open(history_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.6f}", f"{train_eer:.4f}",
                             f"{dev_loss:.6f}", f"{dev_eer:.4f}", f"{current_lr:.8f}"])

        # Checkpoints
        last_path = os.path.join(args.checkpoint_dir, "ssl_last_model.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_dev_eer": best_dev_eer,
        }, last_path)

        if dev_eer < best_dev_eer:
            best_dev_eer = dev_eer
            patience_counter = 0
            best_path = os.path.join(args.checkpoint_dir, "ssl_best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_dev_eer": best_dev_eer,
            }, best_path)
            print(f"New best SSL model saved. Dev EER = {best_dev_eer:.2f}%")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{args.patience}")

        if patience_counter >= args.patience:
            print("\nEarly stopping triggered.")
            break

    print("\nSSL-MLP Training Finished.")
    print(f"Best Dev EER: {best_dev_eer:.2f}%")
    print(f"Best model saved at: {os.path.join(args.checkpoint_dir, 'ssl_best_model.pt')}")


def main():
    parser = argparse.ArgumentParser(description="Train SSL-MLP classifier on cached embeddings.")
    parser.add_argument("--train_embeddings", default="data/embeddings/train.pt")
    parser.add_argument("--dev_embeddings", default="data/embeddings/dev.pt")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_ssl(args)


if __name__ == "__main__":
    main()
