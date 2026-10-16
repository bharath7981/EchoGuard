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

from dataset import ASVspoofDataset
from protocol_parser import parse_protocol
from cnn_baseline import LogMelCNN
from evaluate import compute_eer, evaluate_model


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_class_weights(protocol_path):
    utterances = parse_protocol(protocol_path)

    spoof_count = sum(1 for item in utterances if item.binary_label == 0)
    bonafide_count = sum(1 for item in utterances if item.binary_label == 1)
    total = spoof_count + bonafide_count

    spoof_weight = total / (2 * spoof_count)
    bonafide_weight = total / (2 * bonafide_count)

    print(f"Bonafide samples : {bonafide_count}")
    print(f"Spoof samples    : {spoof_count}")
    print(f"Class weights    : [{spoof_weight:.4f}, {bonafide_weight:.4f}]")

    return torch.tensor([spoof_weight, bonafide_weight], dtype=torch.float32)


def train_one_epoch(model, loader, optimizer, criterion, scaler, device, use_amp, accum_steps):
    model.train()

    total_loss = 0.0
    total_samples = 0
    bonafide_scores = []
    spoof_scores = []

    optimizer.zero_grad(set_to_none=True)
    last_step = 0

    for step, (mel, labels) in enumerate(loader):
        mel = mel.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(mel)
            loss = criterion(logits, labels) / accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        batch_size = labels.size(0)
        total_loss += loss.item() * accum_steps * batch_size
        total_samples += batch_size

        probabilities = torch.softmax(logits.detach(), dim=1)
        scores = probabilities[:, 1].cpu().numpy()
        labels_np = labels.cpu().numpy()

        bonafide_scores.extend(scores[labels_np == 1])
        spoof_scores.extend(scores[labels_np == 0])
        last_step = step

    # flush a partial accumulation window at the end of the epoch
    if total_samples and (last_step + 1) % accum_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    average_loss = total_loss / max(total_samples, 1)
    eer = compute_eer(np.array(bonafide_scores), np.array(spoof_scores))

    return average_loss, eer


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_eer):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_dev_eer": best_eer,
    }, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_flac", required=True)
    parser.add_argument("--train_protocol", required=True)
    parser.add_argument("--dev_flac", required=True)
    parser.add_argument("--dev_protocol", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16,
                         help="16 fits comfortably in 4GB VRAM with AMP on. Drop to 8 if you hit an out-of-memory error.")
    parser.add_argument("--accum_steps", type=int, default=1,
                         help="Gradient accumulation. Use e.g. 2 with --batch_size 8 to simulate an effective batch of 16 on tighter VRAM.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4,
                         help="4 is a safe default for a 6-core Ryzen 5, leaving headroom for the main process.")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--resume", default="")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        # Input spectrogram size is fixed every batch, so cudnn can safely
        # cache the fastest conv algorithms -- free speedup on a laptop GPU.
        torch.backends.cudnn.benchmark = True

    train_dataset = ASVspoofDataset(args.train_flac, args.train_protocol, train=True)
    dev_dataset = ASVspoofDataset(args.dev_flac, args.dev_protocol, train=False)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Dev samples  : {len(dev_dataset)}")

    loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    if args.num_workers > 0:
        # keep worker processes alive across epochs instead of respawning them
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    model = LogMelCNN().to(device)

    class_weights = get_class_weights(args.train_protocol).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 1
    best_dev_eer = float("inf")
    patience_counter = 0

    if args.resume:
        print(f"\nLoading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_dev_eer = checkpoint["best_dev_eer"]
        print(f"Resuming from epoch {start_epoch}")

    history_path = os.path.join(args.checkpoint_dir, "training_history.csv")
    if not args.resume:
        with open(history_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_eer", "dev_loss", "dev_eer", "learning_rate"])

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_eer = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, use_amp, args.accum_steps
        )

        dev_loss, dev_eer = evaluate_model(model, dev_loader, device, criterion)

        scheduler.step(dev_eer)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\nEpoch {epoch:02d}")
        print(f"Train Loss : {train_loss:.4f}")
        print(f"Train EER  : {train_eer:.2f}%")
        print(f"Dev Loss   : {dev_loss:.4f}")
        print(f"Dev EER    : {dev_eer:.2f}%")
        print(f"Learning Rate: {current_lr:.7f}")

        with open(history_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.6f}", f"{train_eer:.4f}",
                              f"{dev_loss:.6f}", f"{dev_eer:.4f}", f"{current_lr:.8f}"])

        last_path = os.path.join(args.checkpoint_dir, "last_model.pt")
        save_checkpoint(last_path, model, optimizer, scheduler, epoch, best_dev_eer)

        if dev_eer < best_dev_eer:
            best_dev_eer = dev_eer
            patience_counter = 0

            best_path = os.path.join(args.checkpoint_dir, "best_model.pt")
            save_checkpoint(best_path, model, optimizer, scheduler, epoch, best_dev_eer)
            print(f"New best model saved. Dev EER = {best_dev_eer:.2f}%")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{args.patience}")

        if patience_counter >= args.patience:
            print("\nEarly stopping triggered.")
            break

    print(f"\nTraining finished.")
    print(f"Best Dev EER: {best_dev_eer:.2f}%")
    print(f"Best model: {args.checkpoint_dir}/best_model.pt")


if __name__ == "__main__":
    main()
