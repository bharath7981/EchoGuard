import argparse
import os
import sys

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import Wav2Vec2Model

sys.path.append("data")
from protocol_parser import parse_protocol

TARGET_SR = 16000
TARGET_LEN = 64600  # ~4.04s


def pad_or_crop(waveform, target_len=TARGET_LEN):
    current_len = len(waveform)
    if current_len == target_len:
        return waveform
    if current_len > target_len:
        return waveform[:target_len]
    repeat_count = target_len // current_len + 1
    waveform = np.tile(waveform, repeat_count)
    return waveform[:target_len]


class RawAudioDataset(Dataset):
    def __init__(self, flac_dir, protocol_path, limit=None):
        self.flac_dir = flac_dir
        self.utterances = parse_protocol(protocol_path)
        if limit is not None and limit > 0:
            self.utterances = self.utterances[:limit]

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        item = self.utterances[idx]
        path = os.path.join(self.flac_dir, item.utt_id + ".flac")

        data, sr = sf.read(path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)

        data = pad_or_crop(data, TARGET_LEN)
        return torch.from_numpy(data), item.binary_label, item.utt_id


def extract_embeddings(flac_dir, protocol_path, output_file, batch_size=16, num_workers=2, limit=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    print(f"Loading wav2vec2-base model...")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)
    model.eval()

    # Freeze entire backbone
    for param in model.parameters():
        param.requires_grad = False

    dataset = RawAudioDataset(flac_dir, protocol_path, limit=limit)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    all_embeddings = []
    all_labels = []
    all_utt_ids = []

    print(f"Extracting embeddings for {len(dataset)} utterances (batch size: {batch_size})...")
    with torch.no_grad():
        for batch_waves, batch_labels, batch_ids in tqdm(loader, desc="Extracting"):
            batch_waves = batch_waves.to(device, non_blocking=True)

            outputs = model(batch_waves)
            hidden = outputs.last_hidden_state  # (batch, time, 768)

            mean_pool = hidden.mean(dim=1)      # (batch, 768)
            std_pool = hidden.std(dim=1)        # (batch, 768)
            pooled = torch.cat([mean_pool, std_pool], dim=1).cpu()  # (batch, 1536)

            all_embeddings.append(pooled)
            all_labels.extend(batch_labels.tolist())
            all_utt_ids.extend(batch_ids)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.tensor(all_labels, dtype=torch.long)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    payload = {
        "embeddings": all_embeddings,
        "labels": all_labels,
        "utt_ids": all_utt_ids,
    }
    torch.save(payload, output_file)
    print(f"Saved embeddings to {output_file}: tensor shape {all_embeddings.shape}")


def main():
    parser = argparse.ArgumentParser(description="Extract and cache frozen wav2vec2 embeddings.")
    parser.add_argument("--flac_dir", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of utterances for quick testing")
    args = parser.parse_args()

    extract_embeddings(
        args.flac_dir,
        args.protocol,
        args.output,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
