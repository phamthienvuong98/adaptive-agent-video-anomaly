"""
models/clip_encoder.py
───────────────────────
CLIP Visual Encoder wrapper.
Dùng để tạo scene embedding nhanh trong Tầng 1.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

# Text prompts để CLIP phân biệt normal vs anomaly (không cần annotation video).
# Đây là cơ chế tạo pseudo-label cho knowledge distillation.
_NORMAL_PROMPTS = [
    "a normal street scene with pedestrians walking",
    "people walking normally in a public space",
    "a quiet and uneventful surveillance scene",
    "normal traffic flow on a road",
]
_ANOMALY_PROMPTS = [
    "people fighting or engaged in physical violence",
    "a robbery or theft in progress",
    "vandalism or destruction of property",
    "a person running away from a crowd in panic",
    "someone collapsing or falling to the ground",
    "a crowd stampede or mass panic",
]


class CLIPEncoder:
    """
    Wrapper cho CLIP (ViT-B/16) để encode frame thành vector 512 chiều.
    Chạy trên GPU, ~2ms/frame với batch_size=1.
    """

    def __init__(self, model_name: str = "ViT-B-16", pretrained: str = "openai", device: str = "cuda"):
        self.device = device
        self._model = None
        self._preprocess = None
        self.model_name = model_name
        self.pretrained = pretrained

    def _load(self):
        if self._model is not None:
            return
        try:
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(
                self.model_name, pretrained=self.pretrained
            )
            model = model.to(self.device).eval()
            self._model = model
            self._preprocess = preprocess
        except ImportError:
            raise ImportError("Cài open-clip-torch: pip install open-clip-torch")

    @torch.inference_mode()
    def encode_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Encode BGR frame → numpy vector (512,).
        """
        self._load()
        img = Image.fromarray(frame[..., ::-1])  # BGR → RGB
        tensor = self._preprocess(img).unsqueeze(0).to(self.device)
        features = self._model.encode_image(tensor)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()[0]

    @torch.inference_mode()
    def compute_similarity(self, frame: np.ndarray, text_queries: list[str]) -> list[float]:
        """
        Tính cosine similarity giữa frame và list text queries.
        Dùng cho coarse anomaly detection dựa trên text description.
        """
        self._load()
        import open_clip
        tokenizer = open_clip.get_tokenizer(self.model_name)

        img_emb = self.encode_frame(frame)
        tokens = tokenizer(text_queries).to(self.device)
        text_features = self._model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        similarities = (
            torch.tensor(img_emb).to(self.device) @ text_features.T
        ).cpu().numpy().tolist()
        return similarities

    @torch.inference_mode()
    def anomaly_score(self, frame: np.ndarray) -> float:
        """
        Tính anomaly score [0,1] dựa trên text prompts không cần annotation.
        score = max(anomaly_sims) - max(normal_sims), chuẩn hóa về [0,1].

        Đây là cơ chế tạo pseudo-label cho CLIP-guided knowledge distillation:
        student CNN học approximate output này mà không cần video có label.
        """
        normal_sims = self.compute_similarity(frame, _NORMAL_PROMPTS)
        anomaly_sims = self.compute_similarity(frame, _ANOMALY_PROMPTS)
        raw = max(anomaly_sims) - max(normal_sims)
        # Sigmoid để map về [0, 1]
        score = float(1.0 / (1.0 + np.exp(-raw * 5.0)))
        return score

    @classmethod
    def get_pseudo_label_prompts(cls) -> tuple[list[str], list[str]]:
        """Trả về (normal_prompts, anomaly_prompts) dùng cho KD training."""
        return _NORMAL_PROMPTS, _ANOMALY_PROMPTS
