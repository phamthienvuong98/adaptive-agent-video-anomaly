"""
utils/anomaly_scorer.py
────────────────────────
Score aggregation và adaptive thresholding cho Tầng 1.
Tách ra để tái sử dụng độc lập với FastFilterAgent.

Tham khảo: TCVADS (arXiv 2412.20201), Cerberus (arXiv 2510.16290)
"""

from __future__ import annotations

import collections

import numpy as np


class AdaptiveThreshold:
    """
    Tự động điều chỉnh threshold dựa trên phân phối score của scene.
    Giảm false positive khi cảnh bình thường có activity cao (ví dụ: giao lộ đông).

    Điểm đóng góp mới: per-scene calibration không cần label.
    """

    def __init__(self, base_threshold: float = 0.45, std_multiplier: float = 2.5, warmup: int = 300):
        self.base = base_threshold
        self.k = std_multiplier
        self.warmup = warmup
        self._scores: collections.deque = collections.deque(maxlen=warmup)
        self._calibrated = False

    def update(self, score: float) -> None:
        self._scores.append(score)
        if len(self._scores) >= self.warmup:
            self._calibrated = True

    @property
    def current(self) -> float:
        if not self._calibrated:
            return self.base
        mu = np.mean(self._scores)
        sigma = np.std(self._scores)
        # Threshold = mean + k*std (chỉ trigger khi score vượt xa mức bình thường)
        return float(np.clip(mu + self.k * sigma, 0.3, 0.85))

    def is_suspicious(self, score: float) -> bool:
        return score >= self.current

    def reset(self) -> None:
        self._scores.clear()
        self._calibrated = False


class TemporalScoreBuffer:
    """
    Tổng hợp score theo cửa sổ thời gian để giảm false positive do nhiễu đơn frame.
    """

    def __init__(self, window: int = 16, strategy: str = "weighted_mean"):
        self.window = window
        self.strategy = strategy
        self._buffer: collections.deque = collections.deque(maxlen=window)
        # Trọng số tăng dần (frame gần nhất quan trọng hơn)
        self._weights = np.linspace(0.5, 1.0, window)

    def push(self, score: float) -> float:
        self._buffer.append(score)
        buf = np.array(self._buffer)
        w = self._weights[-len(buf):]

        if self.strategy == "max":
            return float(buf.max())
        elif self.strategy == "weighted_mean":
            return float(np.average(buf, weights=w))
        return float(buf.mean())

    def clear(self) -> None:
        self._buffer.clear()


class AnomalyScorer:
    """
    Facade kết hợp TemporalScoreBuffer + AdaptiveThreshold.
    Giao diện đơn giản cho FastFilterAgent sử dụng.
    """

    def __init__(self, config: dict):
        self.buffer = TemporalScoreBuffer(
            window=config.get("temporal_window", 16),
            strategy=config.get("aggregation", "weighted_mean"),
        )
        adaptive = config.get("adaptive", {})
        self.threshold = AdaptiveThreshold(
            base_threshold=config.get("threshold", 0.45),
            std_multiplier=adaptive.get("std_multiplier", 2.5),
            warmup=adaptive.get("warmup_frames", 300),
        )

    def score(self, raw_score: float) -> tuple[float, bool]:
        """
        Nhận raw score từ CNN, trả về (aggregated_score, is_suspicious).
        """
        agg = self.buffer.push(raw_score)
        self.threshold.update(agg)
        return agg, self.threshold.is_suspicious(agg)

    @property
    def current_threshold(self) -> float:
        return self.threshold.current
