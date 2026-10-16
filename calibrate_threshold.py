import csv
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Wav2Vec2Model

sys.path.append("models")
sys.path.append("data")

from sklearn.metrics import roc_curve
from ssl_mlp import SSLClassifierMLP
from eval_in_the_wild import InTheWildDataset



def compute_eer(bonafide_scores, spoof_scores):
    if len(bonafide_scores) == 0:
        raise ValueError("No bonafide scores found.")
    if len(spoof_scores) == 0:
        raise ValueError("No spoof scores found.")

    labels = np.concatenate([np.ones(len(bonafide_scores)), np.zeros(len(spoof_scores))])
    scores = np.concatenate([bonafide_scores, spoof_scores])

    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr

    index = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[index] + fnr[index]) / 2 * 100.0
    opt_thresh = float(thresholds[index])

    return eer, opt_thresh


def run_calibration(
    checkpoint_path="checkpoints/rawboost_ssl_best_model.pt",
    data_dir="release_in_the_wild/test",
    output_csv="reports/model_b_in_the_wild_scores.csv",
):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset: {data_dir}")

    # Dataset
    dataset = InTheWildDataset(data_dir)

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )

    # Wav2Vec2 backbone
    print("Loading facebook/wav2vec2-base...")

    ssl_model = (
        W2V := Wav2Vec2Model
        .from_pretrained("facebook/wav2vec2-base")
        .to(device)
        .eval()
    )

    # Model B classifier
    print("Loading Model B...")

    classifier = SSLClassifierMLP(
        in_features=1536,
        hidden_dim=256,
        num_classes=2,
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    classifier.load_state_dict(
        checkpoint["model_state_dict"]
    )

    classifier.eval()

    records = []

    bonafide_scores = []
    spoof_scores = []

    # Inference
    with torch.no_grad():

        for batch_waves, batch_labels, batch_files in tqdm(
            loader,
            desc="Scoring",
        ):

            batch_waves = batch_waves.to(device)

            hidden = ssl_model(
                batch_waves
            ).last_hidden_state

            mean_pool = hidden.mean(dim=1)
            std_pool = hidden.std(dim=1)

            embeddings = torch.cat(
                [mean_pool, std_pool],
                dim=1,
            )

            probs = torch.softmax(
                classifier(embeddings),
                dim=-1,
            )

            # Class 1 = bonafide
            scores = (
                probs[:, 1]
                .cpu()
                .numpy()
            )

            labels_np = batch_labels.numpy()

            for fname, label, score in zip(
                batch_files,
                labels_np,
                scores,
            ):

                records.append(
                    {
                        "filename": fname,
                        "ground_truth": (
                            "bonafide"
                            if label == 1
                            else "spoof"
                        ),
                        "bonafide_score": float(score),
                    }
                )

                if label == 1:
                    bonafide_scores.append(score)
                else:
                    spoof_scores.append(score)

    bonafide_scores = np.array(
        bonafide_scores
    )

    spoof_scores = np.array(
        spoof_scores
    )

    # ------------------------------------------------
    # Calculate EER + threshold
    # ------------------------------------------------

    eer, opt_thresh = compute_eer(
        bonafide_scores,
        spoof_scores,
    )

    # All scores / labels
    all_scores = np.array(
        [
            r["bonafide_score"]
            for r in records
        ]
    )

    all_labels = np.array(
        [
            1 if r["ground_truth"] == "bonafide"
            else 0
            for r in records
        ]
    )

    # >= threshold = bonafide
    all_preds = (
        all_scores >= opt_thresh
    ).astype(int)

    # Accuracy
    accuracy = (
        all_preds == all_labels
    ).mean() * 100.0

    # Spoof accepted as bonafide
    # Security false acceptance rate
    spoof_false_accepts = (
        (all_preds == 1)
        & (all_labels == 0)
    ).sum()

    spoof_false_accept_rate = (
        spoof_false_accepts
        / len(spoof_scores)
        * 100.0
    )

    # Bonafide rejected as spoof
    # Genuine false rejection / miss rate
    bonafide_false_rejects = (
        (all_preds == 0)
        & (all_labels == 1)
    ).sum()

    bonafide_false_reject_rate = (
        bonafide_false_rejects
        / len(bonafide_scores)
        * 100.0
    )

    # ------------------------------------------------
    # Print results
    # ------------------------------------------------

    print("\n" + "=" * 60)
    print("MODEL B CALIBRATION RESULTS")
    print("=" * 60)

    print(
        f"Total samples             : {len(records)}"
    )

    print(
        f"Bonafide samples          : {len(bonafide_scores)}"
    )

    print(
        f"Spoof samples             : {len(spoof_scores)}"
    )

    print(
        f"In-the-Wild EER           : {eer:.2f}%"
    )

    print(
        f"Calibrated threshold      : {opt_thresh:.6f} ({opt_thresh:.8e})"
    )

    print(
        f"Spoof false acceptance   : "
        f"{spoof_false_accept_rate:.2f}%"
    )

    print(
        f"Bonafide false rejection : "
        f"{bonafide_false_reject_rate:.2f}%"
    )

    print(
        f"Overall accuracy         : "
        f"{accuracy:.2f}%"
    )

    print("=" * 60)

    # ------------------------------------------------
    # Save CSV
    # ------------------------------------------------

    output_dir = os.path.dirname(output_csv)

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

    with open(
        output_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "ground_truth",
                "bonafide_score",
                "calibrated_prediction",
            ],
        )

        writer.writeheader()

        for record, pred in zip(
            records,
            all_preds,
        ):

            record["calibrated_prediction"] = (
                "bonafide"
                if pred == 1
                else "spoof"
            )

            writer.writerow(record)

    print(
        f"\nSaved scores to: {output_csv}"
    )

    return opt_thresh


if __name__ == "__main__":
    run_calibration()
