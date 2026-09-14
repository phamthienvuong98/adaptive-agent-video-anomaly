"""
features/scene_embedding.py
────────────────────────────
MobileNetV3-Small scene complexity embedder.
Dùng pretrained ImageNet weights để trích xuất 576-dim embedding,
sau đó tính entropy làm proxy cho scene_complexity (routing weight 0.20).

Khác với tier1_mobilenet: module này KHÔNG fine-tune, chỉ dùng pretrained
features để đo độ phức tạp của cảnh — không phải để detect anomaly.

Latency: ~8ms GPU (Colab T4).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms


class SceneEmbedder:
    """
    Trích xuất scene embedding từ frame bằng MobileNetV3-Small pretrained.
    scene_complexity = entropy của softmax(embedding vector) ∈ [0, ~6.3].
    """

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._model: Optional[nn.Module] = None
        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _load(self) -> None:
        if self._model is not None:
            return
        base = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        # Giữ features + avgpool, bỏ classifier để lấy 576-dim embedding
        base.classifier = nn.Identity()
        self._model = base.to(self.device).eval()

    @torch.inference_mode()
    def embed(self, frame: np.ndarray) -> np.ndarray:
        """
        Trả về embedding vector (576,) từ BGR frame.
        """
        self._load()
        rgb = frame[..., ::-1].copy()
        tensor = self._transform(rgb).unsqueeze(0).to(self.device)
        embedding = self._model(tensor).cpu().numpy()[0]
        return embedding

    @torch.inference_mode()
    def scene_complexity(self, frame: np.ndarray) -> float:
        """
        Tính độ phức tạp cảnh bằng entropy của softmax(embedding).
        Cảnh đơn giản (nền trống) → entropy thấp.
        Cảnh phức tạp (đám đông) → entropy cao.
        """
        embedding = self.embed(frame)
        # Softmax để chuyển về phân phối xác suất
        exp_emb = np.exp(embedding - embedding.max())
        prob = exp_emb / exp_emb.sum()
        # Shannon entropy
        entropy = float(-np.sum(prob * np.log(prob + 1e-9)))
        return entropy
