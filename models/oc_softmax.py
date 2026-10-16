import torch
import torch.nn as nn
import torch.nn.functional as F


class OCSoftmaxLoss(nn.Module):
    """
    One-Class Softmax (OC-Softmax) Loss for Audio Deepfake Detection.
    Reference: Zhang et al., "One-class learning towards synthetic voice spoofing detection",
               IEEE/ACM TASLP 2021.
    
    Bonafide speech is constrained to lie within a compact boundary around a learned center w,
    while spoofed speech is pushed beyond a separate boundary.
    
    Loss formulation:
        L_real = w_real * log(1 + exp(-alpha * (cos_theta - m_real)))
        L_fake = w_fake * log(1 + exp( alpha * (cos_theta - m_fake)))
    """

    def __init__(self, in_features: int = 256, alpha: float = 20.0,
                 m_real: float = 0.9, m_fake: float = 0.2,
                 weight_real: float = 1.0, weight_fake: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.alpha = alpha
        self.m_real = m_real
        self.m_fake = m_fake
        self.weight_real = weight_real
        self.weight_fake = weight_fake

        # Bonafide center vector w
        self.w = nn.Parameter(torch.randn(1, in_features))
        nn.init.xavier_uniform_(self.w)

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None):
        """
        Args:
            x: (batch_size, in_features) embedding vector from backbone/MLP
            labels: (batch_size,) integer labels: 1 = bonafide, 0 = spoof
        Returns:
            If labels is not None: (loss, scores)
            If labels is None (inference): scores (cosine similarities)
        """
        # L2-normalize both features and center weight
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.w, p=2, dim=1)

        # Cosine similarity: (batch_size,)
        scores = torch.mm(x_norm, w_norm.t()).squeeze(1)

        if labels is None:
            return scores

        # Numerically stable softplus: softplus(z) = log(1 + exp(z))
        loss_real = F.softplus(-self.alpha * (scores - self.m_real))
        loss_fake = F.softplus(self.alpha * (scores - self.m_fake))

        # Binary masking
        is_real = (labels == 1).float()
        is_fake = (labels == 0).float()

        weighted_loss = (self.weight_real * is_real * loss_real +
                         self.weight_fake * is_fake * loss_fake)

        return weighted_loss.mean(), scores


class SSLClassifierOC(nn.Module):
    """
    SSL Embedding to OC-Softmax feature projection.
    Maps 1536-dim pooled wav2vec2 embeddings into a compact latent space.
    """

    def __init__(self, in_features: int = 1536, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)
