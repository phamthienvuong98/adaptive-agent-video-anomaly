"""
agents/fast_filter_agent.py
───────────────────────────
Tầng 1: Fast Filter Agent
- Lightweight CNN (student model sau knowledge distillation)
- CLIP visual encoder để embedding scene
- Anomaly Scorer với adaptive threshold

Mục tiêu: ≤5ms/frame, xử lý 100% frame đầu vào
Tham khảo: TCVADS (arXiv 2412.20201), Cerberus (arXiv 2510.16290)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from models.clip_encoder import CLIPEncoder
from models.lightweight_cnn import LightweightCNN
from utils.anomaly_scorer import AdaptiveThreshold, TemporalScoreBuffer, AnomalyScorer
from utils.video_preprocessor import FrameData


@dataclass
class FilterResult:
    is_suspicious: bool
    anomaly_score: float          # [0, 1]
    clip_embedding: np.ndarray    # (512,) scene embedding
    motion_intensity: float       # Optical flow magnitude
    latency_ms: float


class FastFilterAgent:
    """
    Tầng 1 của cascade pipeline.
    Xử lý mọi frame với độ trễ thấp, chỉ chuyển frame đáng ngờ lên tầng 2.
    """

    def __init__(self, config: dict, device: str = "cuda"):
        self.cfg = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Khởi tạo các model nhỏ
        self.clip = CLIPEncoder(
            model_name=config.get("clip_encoder", {}).get("model_name", "ViT-B-16"),
            pretrained=config.get("clip_encoder", {}).get("pretrained", "openai"),
            device=str(self.device),
        )

        self.cnn = LightweightCNN(
            backbone=config.get("lightweight_cnn", {}).get("backbone", "mobilenet_v3_small"),
            checkpoint=config.get("lightweight_cnn", {}).get("checkpoint"),
            device=str(self.device),
        )

        self.scorer = AnomalyScorer(config.get("anomaly_scorer", {}))

    @torch.inference_mode()
    def process(self, frame_data: FrameData) -> FilterResult:
        t0 = time.perf_counter()

        # 1. CNN score (binary: normal=0, suspicious=1)
        cnn_score = self.cnn.predict_score(frame_data.frame)

        # 2. CLIP anomaly score từ text prompts
        clip_score = self.clip.anomaly_score(frame_data.frame)

        # 3. CLIP embedding (dùng cho tầng 2 nếu cần)
        clip_emb = self.clip.encode_frame(frame_data.frame)

        # 4. Motion-aware fusion: high motion → trust CNN; low motion → trust CLIP
        # CNN nhạy với local movement; CLIP nhạy với global scene semantics
        motion = min(frame_data.motion_score, 1.0)
        w_cnn = 0.5 + 0.2 * motion
        fused_score = w_cnn * cnn_score + (1.0 - w_cnn) * clip_score

        # 5. Temporal aggregation + Adaptive threshold check
        agg_score, suspicious = self.scorer.score(fused_score)

        latency_ms = (time.perf_counter() - t0) * 1000

        return FilterResult(
            is_suspicious=suspicious,
            anomaly_score=agg_score,
            clip_embedding=clip_emb,
            motion_intensity=frame_data.motion_score,
            latency_ms=latency_ms,
        )

    def calibrate(self, video_path: str, seconds: int = 60) -> None:
        """Calibrate adaptive threshold từ video bình thường."""
        import cv2
        from utils.video_preprocessor import FrameData

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        max_frames = int(fps * seconds)
        count = 0

        while cap.isOpened() and count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            fd = FrameData(frame=frame, index=count, timestamp=count / fps, motion_score=0.0)
            result = self.process(fd)
            count += 1

        cap.release()
