import torch
import torch.nn as nn


class SSLClassifierMLP(nn.Module):
    """
    Lightweight MLP classifier trained on cached wav2vec2 embeddings.
    Input: (batch, 1536) -> concatenated [mean_pool (768), std_pool (768)]
    Output: (batch, 2) -> logits for [spoof, bonafide]
    """

    def __init__(self, in_features=1536, hidden_dim=256, num_classes=2, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 4, num_classes),
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    model = SSLClassifierMLP()
    dummy = torch.randn(4, 1536)
    out = model(dummy)
    print("SSLClassifierMLP output shape:", out.shape)
