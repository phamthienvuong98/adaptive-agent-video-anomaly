"""
models/lightweight_cnn.py
──────────────────────────
Student CNN model (sau knowledge distillation từ ResNet50/VideoMAE).
Dùng MobileNetV3-Small làm backbone — ~3ms inference trên GPU.

Training note: Distill từ teacher model (ResNet50 + LSTM) trên UCF-Crime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms


class LightweightCNN(nn.Module):
    """
    Student CNN: MobileNetV3-Small + anomaly head.
    Input: BGR frame (224x224)
    Output: anomaly_score ∈ [0, 1]
    """

    def __init__(self, backbone: str = "mobilenet_v3_small", checkpoint: Optional[str] = None, device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # ── Backbone ─────────────────────────────────────────────────────
        if backbone == "mobilenet_v3_small":
            base = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            feature_dim = 576
        elif backbone == "mobilenet_v3_large":
            base = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
            feature_dim = 960
        else:
            raise ValueError(f"Backbone không hỗ trợ: {backbone}")

        # Dùng feature extractor (bỏ classifier gốc)
        self.features = base.features
        self.avgpool = base.avgpool

        # ── Anomaly detection head ────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # ── Preprocessing ─────────────────────────────────────────────────
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.to(self.device)

        if checkpoint and Path(checkpoint).exists():
            self.load_checkpoint(checkpoint)
        else:
            if checkpoint:
                import warnings
                warnings.warn(f"Checkpoint không tìm thấy: {checkpoint}. Dùng pretrained ImageNet weights.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return self.head(x)

    @torch.inference_mode()
    def predict_score(self, frame: np.ndarray) -> float:
        """
        Predict anomaly score từ BGR frame numpy.
        Returns: float ∈ [0, 1]
        """
        # BGR → RGB
        rgb = frame[..., ::-1].copy()
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        score = self(tensor).item()
        return score

    def load_checkpoint(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        self.load_state_dict(state, strict=False)
        self.eval()

    # ── Knowledge Distillation Training ──────────────────────────────────

    @staticmethod
    def distillation_loss(
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        temperature: float = 4.0,
        alpha: float = 0.7,
    ) -> torch.Tensor:
        """
        Combined loss: CrossEntropy (hard labels) + KL Divergence (soft labels từ teacher).
        alpha: trọng số soft labels (cao hơn = học nhiều hơn từ teacher).
        """
        # Hard label loss
        ce_loss = F.binary_cross_entropy(student_logits.squeeze(), labels.float())

        # Soft label loss (temperature scaling)
        student_soft = torch.sigmoid(student_logits / temperature)
        teacher_soft = torch.sigmoid(teacher_logits / temperature)
        kd_loss = F.mse_loss(student_soft, teacher_soft.detach()) * (temperature ** 2)

        return alpha * kd_loss + (1 - alpha) * ce_loss
