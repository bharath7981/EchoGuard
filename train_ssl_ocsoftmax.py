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
from oc_softmax import OCSoftmaxLoss, SSLClassifierOC
from evaluate import compute_eer


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_oc_model(model, criterion, loader, device):
    model.eval()
    criterion.eval()
    total_loss = 0.0
    total_samples = 0
    bonafide_scores = []
    spoof_scores = []

    with torch.no_grad():
        for embeddings, labels in loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            feats = model(embeddings)
            loss, scores = criterion(feats, labels)

            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

            scores_np = scores.cpu().numpy()
            labels_np = labels.cpu().numpy()

            bonafide_scores.extend(scores_np[labels_np == 1])
            spoof_scores.extend(scores_np[labels_np == 0])

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    eer = compute_eer(np.array(bonafide_scores), np.array(spoof_scores))
    return avg_loss, eer


def train_oc_softmax(args):
    set_seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n==========================================================")
    print(" Training SSL + One-Class Softmax (OC-Softmax)")
    print(f" Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f" Train : {args.train_embeddings}")
    print(f" Dev   : {args.dev_embeddings}")
    print(f" Alpha : {args.alpha}, Margin Real: {args.m_real}, Margin Fake: {args.m_fake}")
    print("==========================================================\n")

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
    print(f"Bonafide samples : {bonafide_count}")
    print(f"Spoof samples    : {spoof_count}")
    print(f"Class weights    : Real={bonafide_weight:.4f}, Fake={spoof_weight:.4f}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)

    model = SSLClassifierOC(in_features=1536, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    criterion = OCSoftmaxLoss(
        in_features=args.hidden_dim,
        alpha=args.alpha,
        m_real=args.m_real,
        m_fake=args.m_fake,
        weight_real=bonafide_weight,
        weight_fake=spoof_weight,
    ).to(device)

    # Both model parameters and center vector w are optimized
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6
    )

    history_path = os.path.join(args.checkpoint_dir, f"{args.model_name}_training_history.csv")
    with open(history_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_eer", "dev_loss", "dev_eer", "learning_rate"])

    best_dev_eer = float("inf")
    best_smoothed_eer = float("inf")
    patience_counter = 0
    dev_eer_history = []

    print("\nStarting OC-Softmax Training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        criterion.train()
        total_loss = 0.0
        total_samples = 0
        train_bonafide_scores = []
        train_spoof_scores = []

        for embeddings, labels in train_loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            feats = model(embeddings)
            loss, scores = criterion(feats, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

            scores_np = scores.detach().cpu().numpy()
            lbls = labels.detach().cpu().numpy()
            train_bonafide_scores.extend(scores_np[lbls == 1])
            train_spoof_scores.extend(scores_np[lbls == 0])

        train_loss = total_loss / total_samples
        train_eer = compute_eer(np.array(train_bonafide_scores), np.array(train_spoof_scores))

        dev_loss, dev_eer = evaluate_oc_model(model, criterion, dev_loader, device)
        dev_eer_history.append(dev_eer)
        smoothed_dev_eer = float(np.mean(dev_eer_history[-args.smooth_window:]))

        scheduler.step(smoothed_dev_eer)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\nEpoch {epoch:02d}")
        print(f"Train Loss   : {train_loss:.4f}")
        print(f"Train EER    : {train_eer:.2f}%")
        print(f"Dev Loss     : {dev_loss:.4f}")
        print(f"Dev EER      : {dev_eer:.2f}% (Smoothed: {smoothed_dev_eer:.2f}%)")
        print(f"Learning Rate: {current_lr:.7f}")

        with open(history_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.6f}", f"{train_eer:.4f}",
                             f"{dev_loss:.6f}", f"{dev_eer:.4f}", f"{current_lr:.8f}"])

        # Checkpoints
        last_path = os.path.join(args.checkpoint_dir, f"{args.model_name}_last_model.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "criterion_state_dict": criterion.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_dev_eer": best_dev_eer,
            "smoothed_dev_eer": smoothed_dev_eer,
            "model_type": "oc_softmax",
            "hidden_dim": args.hidden_dim,
        }, last_path)

        if smoothed_dev_eer < best_smoothed_eer:
            best_smoothed_eer = smoothed_dev_eer
            best_dev_eer = dev_eer
            patience_counter = 0
            best_path = os.path.join(args.checkpoint_dir, f"{args.model_name}_best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "criterion_state_dict": criterion.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_dev_eer": best_dev_eer,
                "smoothed_dev_eer": smoothed_dev_eer,
                "model_type": "oc_softmax",
                "hidden_dim": args.hidden_dim,
            }, best_path)
            print(f"New best OC-Softmax model saved. Dev EER = {best_dev_eer:.2f}% (Smoothed: {smoothed_dev_eer:.2f}%)")
        else:
            patience_counter += 1
            print(f"No improvement in smoothed Dev EER. Patience: {patience_counter}/{args.patience}")

        if patience_counter >= args.patience:
            print("\nEarly stopping triggered based on smoothed Dev EER.")
            break

    print("\nOC-Softmax Training Finished.")
    print(f"Best Dev EER: {best_dev_eer:.2f}% (Best Smoothed: {best_smoothed_eer:.2f}%)")
    print(f"Best model saved at: {os.path.join(args.checkpoint_dir, f'{args.model_name}_best_model.pt')}")


def main():
    parser = argparse.ArgumentParser(description="Train SSL + OC-Softmax classifier.")
    parser.add_argument("--train_embeddings", default="data/embeddings/train.pt,data/embeddings/train_rawboost.pt")
    parser.add_argument("--dev_embeddings", default="data/embeddings/dev.pt")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--model_name", default="rawboost_ocsoftmax")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=20.0)
    parser.add_argument("--m_real", type=float, default=0.9)
    parser.add_argument("--m_fake", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--smooth_window", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_oc_softmax(args)


if __name__ == "__main__":
    main()
