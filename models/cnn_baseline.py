import torch
import torch.nn as nn


class LogMelCNN(nn.Module):
    """
    Lightweight CNN baseline for bonafide/spoof classification.

    Expects a precomputed log-mel spectrogram as input:
        shape = (batch, 1, n_mels, time_frames)

    Small enough to train comfortably on a 4GB laptop GPU (RTX 2050).
    """

    def __init__(self, n_mels=80):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, 2)

    def forward(self, mel):
        x = self.features(mel)
        x = self.pool(x)
        x = x.flatten(1)
        return self.classifier(x)


if __name__ == "__main__":
    model = LogMelCNN()
    dummy = torch.randn(2, 1, 80, 253)  # batch=2, 80 mels, ~4s of frames
    output = model(dummy)
    print("Output shape:", output.shape)
