"""
utils/video_preprocessor.py
────────────────────────────
Xử lý video đầu vào: frame sampling, motion mask, optical flow.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Generator, Optional

import cv2
import numpy as np


@dataclass
class FrameData:
    frame: np.ndarray          # BGR frame (H, W, 3)
    index: int                 # Frame index
    timestamp: float           # Giây
    motion_score: float        # Optical flow magnitude trung bình
    camera_id: str = "default"
    prev_frame: Optional[np.ndarray] = None


class VideoPreprocessor:
    """
    Stream frame từ video file hoặc camera.
    Áp dụng:
      - Frame sampling (uniform hoặc motion-triggered)
      - Motion mask từ optical flow (loại bỏ vùng tĩnh)
      - Resize về kích thước chuẩn
    """

    def __init__(self, config: dict):
        self.strategy = config.get("strategy", "uniform")
        self.sample_rate = config.get("sample_rate", 1)
        self.motion_threshold = config.get("motion_threshold", 0.01)
        self.target_size = (224, 224)  # Kích thước chuẩn cho model

    def stream(
        self,
        source: str | int,
        camera_id: str = "default",
    ) -> Generator[FrameData, None, None]:
        """
        Generator yield FrameData từ video/camera.
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Không thể mở: {source}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        prev_gray = None
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # ── Frame sampling ─────────────────────────────────────────
            if self.strategy == "uniform" and frame_idx % self.sample_rate != 0:
                continue

            # ── Tính motion score (optical flow) ─────────────────────
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            motion_score = 0.0

            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray,
                    None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                motion_score = float(magnitude.mean())

            # Motion-triggered sampling: bỏ qua nếu ít chuyển động
            if self.strategy == "motion_triggered" and motion_score < self.motion_threshold:
                prev_gray = gray
                continue

            # ── Resize ────────────────────────────────────────────────
            resized = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)

            yield FrameData(
                frame=resized,
                index=frame_idx,
                timestamp=frame_idx / fps,
                motion_score=motion_score,
                camera_id=camera_id,
            )

            prev_gray = gray

        cap.release()

    def process_single(self, frame: np.ndarray, index: int = 0, camera_id: str = "default") -> FrameData:
        """Xử lý một frame đơn lẻ (dùng cho real-time API)."""
        resized = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)
        return FrameData(
            frame=resized,
            index=index,
            timestamp=time.time(),
            motion_score=0.0,
            camera_id=camera_id,
        )
