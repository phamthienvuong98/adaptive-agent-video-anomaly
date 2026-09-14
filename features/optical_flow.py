"""
features/optical_flow.py
─────────────────────────
Farneback optical flow extractor.
Trích xuất motion signal dùng làm routing input (weight 0.35).

Latency: ~2ms/frame CPU.
"""

from __future__ import annotations

import cv2
import numpy as np


class OpticalFlowExtractor:
    """
    Tính Farneback optical flow giữa 2 frame liên tiếp.
    Output dict được normalize về [0, 1] bởi RoutingAgent.
    """

    def __init__(
        self,
        pyr_scale: float = 0.5,
        levels: int = 3,
        winsize: int = 15,
        iterations: int = 3,
        poly_n: int = 5,
        poly_sigma: float = 1.2,
        motion_threshold: float = 0.5,
    ):
        self.params = dict(
            pyr_scale=pyr_scale,
            levels=levels,
            winsize=winsize,
            iterations=iterations,
            poly_n=poly_n,
            poly_sigma=poly_sigma,
            flags=0,
        )
        self.motion_threshold = motion_threshold

    def compute(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> dict:
        """
        Tính optical flow giữa 2 BGR frame.

        Returns:
            dict với keys: mean, max, std, coverage
            Trả về zeros nếu prev_frame là None (frame đầu tiên).
        """
        if prev_frame is None:
            return {"mean": 0.0, "max": 0.0, "std": 0.0, "coverage": 0.0}

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, **self.params)
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

        return {
            "mean": float(magnitude.mean()),
            "max": float(magnitude.max()),
            "std": float(magnitude.std()),
            "coverage": float((magnitude > self.motion_threshold).mean()),
        }
