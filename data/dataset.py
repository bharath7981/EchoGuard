import os
import random

import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset

from protocol_parser import parse_protocol

TARGET_SR = 16000
TARGET_LEN = 64600  # ~4.04s at 16kHz, matches the project's fixed-length convention


def pad_or_crop(waveform, target_len, train=False):
    current_len = waveform.shape[-1]

    if current_len == target_len:
        return waveform

    if current_len > target_len:
        # random crop during training (acts as light augmentation),
        # fixed crop from the start during eval (reproducible)
        start = random.randint(0, current_len - target_len) if train else 0
        return waveform[:, start:start + target_len]

    # shorter than target: loop the audio to fill the window
    repeat_count = target_len // current_len + 1
    waveform = waveform.repeat(1, repeat_count)
    return waveform[:, :target_len]


class ASVspoofDataset(Dataset):
    """
    Loads raw audio, resamples to 16kHz, pads/crops to a fixed length,
    then converts to a log-mel spectrogram -- all on the CPU, inside
    DataLoader worker processes.

    This split matters on a 4GB card like the RTX 2050: the GPU only
    ever receives a small, fixed-size spectrogram tensor and spends all
    its memory/compute on the conv layers, while the Ryzen 5's cores
    handle audio decoding + FFT work in parallel via num_workers.
    """

    def __init__(self, flac_dir, protocol_path, train=False, target_len=TARGET_LEN, n_mels=80):
        self.flac_dir = flac_dir
        self.train = train
        self.target_len = target_len

        self.utterances = parse_protocol(protocol_path)

        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=TARGET_SR,
            n_fft=1024,
            hop_length=256,
            n_mels=n_mels,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, index):
        item = self.utterances[index]

        audio_path = os.path.join(self.flac_dir, item.utt_id + ".flac")

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        data, sample_rate = sf.read(audio_path, dtype="float32")
        waveform = torch.from_numpy(data)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.t()
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

        if sample_rate != TARGET_SR:
            waveform = torchaudio.functional.resample(waveform, sample_rate, TARGET_SR)

        waveform = pad_or_crop(waveform, self.target_len, self.train)

        mel = self.melspec(waveform)
        mel = self.to_db(mel)

        label = torch.tensor(item.binary_label, dtype=torch.long)

        return mel, label


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python dataset.py <flac_dir> <protocol_path>")
        sys.exit(1)

    dataset = ASVspoofDataset(sys.argv[1], sys.argv[2])

    print("Dataset size:", len(dataset))

    mel, label = dataset[0]
    print(f"First item: mel shape={tuple(mel.shape)}, label={label.item()}")

    print("\nChecking 10 samples...")
    for i in range(min(10, len(dataset))):
        mel, label = dataset[i]
        print(f"  [{i}] mel shape={tuple(mel.shape)}  label={label.item()}")
