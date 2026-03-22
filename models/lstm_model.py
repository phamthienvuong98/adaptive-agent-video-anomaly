"""
LSTM Anomaly Model — Baseline temporal model for video anomaly detection.

Takes a sequence of CNN features (B, T, feature_dim) and produces
an anomaly score per clip using bidirectional LSTM + attention.

This is the BASELINE model (Phase 1) before introducing Graph modules.

References:
    - Sultani et al., "Real-World Anomaly Detection in Surveillance Videos" (CVPR 2018)
    - Weakly supervised ranking loss for anomaly detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class TemporalAttention(nn.Module):
    """
    Soft attention over temporal dimension.
    Learns which timesteps are most relevant for anomaly classification.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lstm_out: (B, T, hidden_dim)
        Returns:
            context: (B, hidden_dim) — attention-weighted representation
        """
        weights = self.attention(lstm_out)         # (B, T, 1)
        weights = F.softmax(weights, dim=1)        # (B, T, 1)
        context = (lstm_out * weights).sum(dim=1)  # (B, hidden_dim)
        return context


class LSTMAnomaly(nn.Module):
    """
    Baseline anomaly detection model.

    Pipeline:
        CNN features (B, T, feat_dim)
        → Linear projection
        → LSTM
        → Temporal Attention
        → Classifier → anomaly score

    Supports:
        - Ranking loss (weakly supervised)
        - BCE loss (fully supervised)
    """

    def __init__(
        self,
        input_dim: int = config.CNN_FEATURE_DIM,
        hidden_dim: int = config.LSTM_HIDDEN_DIM,
        num_layers: int = config.LSTM_NUM_LAYERS,
        dropout: float = config.LSTM_DROPOUT,
        bidirectional: bool = config.LSTM_BIDIRECTIONAL,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Attention
        lstm_output_dim = hidden_dim * self.num_directions
        self.attention = TemporalAttention(lstm_output_dim)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_dim) — sequence of CNN features
        Returns:
            score: (B, 1) — anomaly score in [0, 1]
        """
        # Project input
        x = self.input_proj(x)  # (B, T, hidden_dim)

        # LSTM encoding
        lstm_out, _ = self.lstm(x)  # (B, T, hidden_dim * num_directions)

        # Attention pooling
        context = self.attention(lstm_out)  # (B, hidden_dim * num_directions)

        # Classify
        score = self.classifier(context)  # (B, 1)
        return score


class RankingLoss(nn.Module):
    """
    Weakly supervised ranking loss for anomaly detection.

    Given a batch of (normal, anomaly) pairs:
        max(0, margin - (score_anomaly - score_normal))

    This encourages anomaly videos to have higher scores than normal ones.

    Reference: Sultani et al., CVPR 2018
    """

    def __init__(self, margin: float = config.RANKING_MARGIN):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        score_normal: torch.Tensor,
        score_anomaly: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            score_normal:  (B, 1) scores from normal clips
            score_anomaly: (B, 1) scores from anomaly clips
        Returns:
            loss: scalar
        """
        # Ranking loss
        ranking_loss = F.relu(self.margin - (score_anomaly - score_normal))

        # Sparsity regularisation (anomaly scores should be sparse)
        sparsity = score_anomaly.mean()

        # Smoothness regularisation (temporal consistency)
        # Encourage smooth score transitions
        loss = ranking_loss.mean() + 0.01 * sparsity

        return loss


if __name__ == "__main__":
    model = LSTMAnomaly(input_dim=2048, hidden_dim=512)
    print(f"LSTM model parameters: {sum(p.numel() for p in model.parameters()):,}")

    dummy = torch.randn(4, 16, 2048)
    scores = model(dummy)
    print(f"Output shape: {scores.shape}")  # (4, 1)
    print(f"Scores: {scores.squeeze().tolist()}")
