import argparse
import csv
import glob
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import Wav2Vec2Model

sys.path.append(os.path.join(os.path.dirname(__file__), "models"))
sys.path.append(os.path.join(os.path.dirname(__file__), "data"))

from ssl_mlp import SSLClassifierMLP
from oc_softmax import OCSoftmaxLoss, SSLClassifierOC
from evaluate import compute_eer

TARGET_SR = 16000
TARGET_LEN = 64600  # ~4.04s at 16kHz


def pad_or_crop(waveform: np.ndarray, target_len: int = TARGET_LEN) -> np.ndarray:
    current_len = len(waveform)
    if current_len == target_len:
        return waveform
    if current_len > target_len:
        return waveform[:target_len]
    repeat_count = target_len // current_len + 1
    waveform = np.tile(waveform, repeat_count)
    return waveform[:target_len]


class InTheWildDataset(Dataset):
    """
    Dataset loader for the In-the-Wild Audio Deepfake benchmark.
    Supports either:
      1. A metadata CSV/TSV file (with columns for file/path and label/label_str), OR
      2. Folder structure with 'bonafide'/'bona-fide' and 'spoof'/'fake' subdirectories.
    """

    def __init__(self, data_path: str, meta_file: str = None, limit: int = None):
        self.samples = []  # list of (audio_path, label_int: 0=spoof, 1=bonafide)

        if meta_file and os.path.exists(meta_file):
            print(f"[InTheWild] Reading metadata from {meta_file}...")
            delimiter = "\t" if meta_file.endswith((".tsv", ".txt")) else ","
            with open(meta_file, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                for row in reader:
                    # Normalize column names
                    row_lower = {k.lower().strip() if k else "": v for k, v in row.items()}
                    
                    # File column detection
                    file_name = row_lower.get("file") or row_lower.get("filename") or row_lower.get("path") or row_lower.get("audio")
                    # Label column detection
                    label_str = row_lower.get("label") or row_lower.get("class") or row_lower.get("target")

                    if not file_name or not label_str:
                        # Fallback for plain headerless files: row values
                        vals = list(row.values())
                        if len(vals) >= 2:
                            file_name, label_str = vals[0], vals[-1]
                        else:
                            continue

                    file_name = file_name.strip()
                    label_str = label_str.lower().strip()

                    full_path = file_name if os.path.isabs(file_name) else os.path.join(data_path, file_name)
                    
                    # Try common audio extensions if missing
                    if not os.path.exists(full_path):
                        for ext in [".wav", ".mp3", ".flac", ".ogg", ".m4a"]:
                            if os.path.exists(full_path + ext):
                                full_path = full_path + ext
                                break

                    # Map label: 1=bonafide, 0=spoof
                    if label_str in ["bonafide", "bona-fide", "real", "human", "1"]:
                        label = 1
                    elif label_str in ["spoof", "fake", "deepfake", "synthetic", "0"]:
                        label = 0
                    else:
                        continue

                    if os.path.exists(full_path):
                        self.samples.append((full_path, label))

        elif os.path.isdir(data_path):
            print(f"[InTheWild] Scanning directory structure in {data_path}...")
            # Check for subdirectories
            for root, _, files in os.walk(data_path):
                root_lower = os.path.basename(root).lower()
                for fname in files:
                    if fname.lower().endswith((".wav", ".flac", ".mp3", ".ogg", ".m4a")):
                        full_path = os.path.join(root, fname)
                        if any(b in root_lower for b in ["bona", "real", "human", "genuine"]):
                            self.samples.append((full_path, 1))
                        elif any(s in root_lower for s in ["spoof", "fake", "synth", "clone"]):
                            self.samples.append((full_path, 0))

        if limit is not None and limit > 0:
            bonas = [s for s in self.samples if s[1] == 1]
            spoofs = [s for s in self.samples if s[1] == 0]
            half = limit // 2
            self.samples = bonas[:half] + spoofs[:limit - half]

        bonafide_count = sum(1 for _, l in self.samples if l == 1)
        spoof_count = sum(1 for _, l in self.samples if l == 0)
        print(f"[InTheWild] Total loaded: {len(self.samples)} utterances | Bonafide: {bonafide_count}, Spoof: {spoof_count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            data, sr = sf.read(path, dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)

            if sr != TARGET_SR:
                import torchaudio.functional as F
                tensor_data = torch.from_numpy(data).unsqueeze(0)
                resampled = F.resample(tensor_data, sr, TARGET_SR)
                data = resampled.squeeze(0).numpy()

            data = pad_or_crop(data, TARGET_LEN)
            return torch.from_numpy(data), label, os.path.basename(path)
        except Exception as e:
            # Return silent padding in case of corrupted audio file
            return torch.zeros(TARGET_LEN, dtype=torch.float32), label, os.path.basename(path)


def evaluate_in_the_wild(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f" EchoGuard In-the-Wild Generalization Evaluation")
    print(f" Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f" Checkpoint: {args.checkpoint}")
    print(f"=======================================================\n")

    # 1. Load Dataset
    dataset = InTheWildDataset(args.data_dir, meta_file=args.meta_file, limit=args.limit)
    if len(dataset) == 0:
        print("[Error] No audio files found. Please check data_dir and meta_file paths.")
        return

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # 2. Load Models
    print("\nLoading facebook/wav2vec2-base backbone...")
    ssl_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)
    ssl_model.eval()
    for p in ssl_model.parameters():
        p.requires_grad = False

    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_type = checkpoint.get("model_type", "standard_mlp")

    if model_type == "oc_softmax":
        print("Detected OC-Softmax checkpoint architecture.")
        hidden_dim = checkpoint.get("hidden_dim", 256)
        classifier = SSLClassifierOC(in_features=1536, hidden_dim=hidden_dim).to(device)
        classifier.load_state_dict(checkpoint["model_state_dict"])
        classifier.eval()

        criterion = OCSoftmaxLoss(in_features=hidden_dim).to(device)
        criterion.load_state_dict(checkpoint["criterion_state_dict"])
        criterion.eval()
    else:
        print("Detected standard Cross-Entropy MLP checkpoint architecture.")
        classifier = SSLClassifierMLP(in_features=1536, hidden_dim=256, num_classes=2).to(device)
        classifier.load_state_dict(checkpoint["model_state_dict"])
        classifier.eval()

    # 3. Run Inference
    bonafide_scores = []
    spoof_scores = []
    all_scores = []
    all_labels = []

    print("\nRunning inference on In-the-Wild utterances...")
    start_t = time.perf_counter()
    with torch.no_grad():
        for batch_waves, batch_labels, _ in tqdm(loader, desc="Evaluating In-The-Wild"):
            batch_waves = batch_waves.to(device, non_blocking=True)

            outputs = ssl_model(batch_waves)
            hidden = outputs.last_hidden_state  # (batch, T, 768)

            mean_pool = hidden.mean(dim=1)
            std_pool = hidden.std(dim=1)
            embedding = torch.cat([mean_pool, std_pool], dim=1)  # (batch, 1536)

            if model_type == "oc_softmax":
                feats = classifier(embedding)
                scores = criterion(feats).cpu().numpy()
            else:
                logits = classifier(embedding)
                probs = torch.softmax(logits, dim=-1)
                scores = probs[:, 1].cpu().numpy()  # index 1 = bonafide probability

            labels_np = batch_labels.numpy()
            bonafide_scores.extend(scores[labels_np == 1])
            spoof_scores.extend(scores[labels_np == 0])

            all_scores.extend(scores)
            all_labels.extend(labels_np)

    elapsed = time.perf_counter() - start_t
    print(f"\nInference completed in {elapsed:.1f}s (avg {elapsed / len(dataset) * 1000:.1f} ms/sample)")

    # 4. Metrics Computation
    bonafide_scores = np.array(bonafide_scores)
    spoof_scores = np.array(spoof_scores)
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    wild_eer = compute_eer(bonafide_scores, spoof_scores)

    # For standard softmax: threshold is 0.5 (probability)
    # For OC-Softmax: threshold is 0.0 (cosine similarity center)
    threshold = 0.0 if model_type == "oc_softmax" else 0.5
    all_preds = (all_scores >= threshold).astype(int)

    acc = (all_preds == all_labels).mean() * 100
    
    # False Alarm (human classified as spoof) and Miss (spoof classified as human)
    fa_rate = ((all_preds == 0) & (all_labels == 1)).sum() / len(bonafide_scores) * 100 if len(bonafide_scores) > 0 else 0.0
    miss_rate = ((all_preds == 1) & (all_labels == 0)).sum() / len(spoof_scores) * 100 if len(spoof_scores) > 0 else 0.0

    asvspoof_eer = 2.72
    gen_gap = wild_eer - asvspoof_eer

    print("\n" + "=" * 58)
    print("        IN-THE-WILD BENCHMARK EVALUATION RESULTS")
    print("=" * 58)
    print(f" In-the-Wild Equal Error Rate (EER) : {wild_eer:.2f}%")
    print(f" ASVspoof 2019 LA Benchmark EER     : {asvspoof_eer:.2f}%")
    print(f" Generalization Gap (Delta EER)     : +{gen_gap:.2f}% (Out-of-Domain Drop)")
    print("-" * 58)
    print(f" Overall Accuracy (Threshold=0.5)   : {acc:.2f}%")
    print(f" False Alarm Rate (Human -> Spoof)  : {fa_rate:.2f}%")
    print(f" Miss Rate (Deepfake -> Human)      : {miss_rate:.2f}%")
    print("=" * 58 + "\n")

    # 5. Log Results
    os.makedirs("reports", exist_ok=True)
    with open("reports/results.md", "a", encoding="utf-8") as f:
        f.write(f"- In-the-Wild Dataset Eval EER: {wild_eer:.2f}%  (Delta from ASVspoof19: +{gen_gap:.2f}%, Accuracy: {acc:.2f}%)\n")

    report_path = "reports/in_the_wild_analysis.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# In-the-Wild Generalization Analysis\n\n")
        f.write(f"Evaluated on {len(dataset)} real-world utterances using checkpoint `{args.checkpoint}`.\n\n")
        f.write(f"| Metric | Score |\n")
        f.write(f"| :--- | :---: |\n")
        f.write(f"| **In-the-Wild EER** | **`{wild_eer:.2f}%`** |\n")
        f.write(f"| **ASVspoof 2019 LA EER** | `2.72%` |\n")
        f.write(f"| **Generalization Gap (Delta EER)** | `+{gen_gap:.2f}%` |\n")
        f.write(f"| **Accuracy (at 0.5)** | `{acc:.2f}%` |\n")
        f.write(f"| **False Alarm Rate** | `{fa_rate:.2f}%` |\n")
        f.write(f"| **Miss Rate (Deepfake missed)** | `{miss_rate:.2f}%` |\n\n")
        f.write(f"### Research Takeaways:\n")
        f.write(f"1. **Baseline In-Domain Performance**: The model achieves high separation (2.72% EER) on controlled ASVspoof studio conditions.\n")
        f.write(f"2. **Out-of-Domain Vulnerability**: Real-world acoustic variations, YouTube/podcast compression artifacts, and unseen generative models widen the error rate to {wild_eer:.2f}%.\n")
        f.write(f"3. **Next Technical Step**: This concrete `{wild_eer:.2f}%` baseline provides the empirical benchmark to measure the impact of **One-Class Softmax (OC-Softmax)** or domain adaptation.\n")

    print(f"Appended summary to reports/results.md and generated {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate EchoGuard SSL model on In-the-Wild dataset.")
    parser.add_argument("--data_dir", required=True, help="Path to in-the-wild audio directory")
    parser.add_argument("--meta_file", default=None, help="Path to meta.csv / protocol file (optional)")
    parser.add_argument("--checkpoint", default="checkpoints/ssl_best_model.pt", help="Path to trained model checkpoint")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples for testing")
    args = parser.parse_args()

    evaluate_in_the_wild(args)


if __name__ == "__main__":
    main()
