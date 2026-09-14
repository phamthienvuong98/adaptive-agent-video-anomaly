"""
agents/routing_agent.py
────────────────────────
Multi-signal Routing Agent — Core Novelty của hệ thống.

Thay vì cascade cứng (1 signal → threshold cố định), agent này đọc
4 tín hiệu độc lập từ mỗi frame và tính routing_score để quyết định
tier xử lý phù hợp nhất.

routing_score = (
    0.35 × motion_magnitude_norm
    + 0.25 × person_count_norm
    + 0.20 × scene_complexity_norm
    + 0.20 × risk_context_numeric
) × night_multiplier

Tiers:
  score < 0.35  → Tier 1 (~94% frames): MobileNetV3 anomaly scorer
  score 0.35–0.7 → Tier 2 (~5% frames): CLIP zero-shot
  score > 0.70  → Tier 3 (~1% frames): Qwen2.5-VL-7B dialogue loop

Tham khảo: PLAN.md §5.2, Cerberus (2510.16290), SlowFastVAD (2504.10320)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger


@dataclass
class RoutingDecision:
    tier: int                # 1, 2, hoặc 3
    routing_score: float     # [0, 1]
    motion_norm: float
    person_norm: float
    complexity_norm: float
    risk_numeric: float
    is_night: bool
    camera_id: str


class RoutingAgent:
    """
    Multi-signal routing agent. Load config từ routing_config.yaml.

    Ablation variants được tạo bằng cách override weights:
      V1 (fixed cascade): bỏ qua routing, dùng fixed threshold
      V2 (motion only): weights = {motion: 1.0, person: 0, complexity: 0, risk: 0}
      V3 (motion + person): weights = {motion: 0.58, person: 0.42, complexity: 0, risk: 0}
      V4 (3 signals): weights = {motion: 0.47, person: 0.33, complexity: 0.20, risk: 0}
      V5 (full): weights từ config (0.35/0.25/0.20/0.20)
    """

    DEFAULT_WEIGHTS = {
        "motion": 0.35,
        "person": 0.25,
        "complexity": 0.20,
        "risk": 0.20,
    }
    DEFAULT_THRESHOLDS = [0.35, 0.70]
    DEFAULT_NIGHT_MULTIPLIER = 1.3

    # Normalization ranges (upper bound cho từng signal)
    DEFAULT_NORM_RANGES = {
        "motion_mean": 5.0,      # mean optical flow magnitude
        "person_count": 20.0,    # max người expected
        "scene_complexity": 4.0, # max entropy value (~ln(576))
    }

    def __init__(
        self,
        config: Optional[dict] = None,
        config_path: Optional[str] = None,
        camera_config: Optional[dict] = None,
    ):
        cfg = {}
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                raw = yaml.safe_load(f)
            cfg = raw.get("routing_agent", {})
        if config:
            cfg.update(config)

        weights = cfg.get("weights", self.DEFAULT_WEIGHTS)
        self.weights = {k: float(weights.get(k, v)) for k, v in self.DEFAULT_WEIGHTS.items()}

        thresholds = cfg.get("tier_thresholds", self.DEFAULT_THRESHOLDS)
        self.tier_thresholds = [float(thresholds[0]), float(thresholds[1])]
        self.night_multiplier = float(cfg.get("night_multiplier", self.DEFAULT_NIGHT_MULTIPLIER))

        norm_ranges = cfg.get("normalization_ranges", {})
        self.norm_ranges = {
            k: float(norm_ranges.get(k, v))
            for k, v in self.DEFAULT_NORM_RANGES.items()
        }

        self.camera_config = camera_config or {}
        logger.debug(f"RoutingAgent init: weights={self.weights}, thresholds={self.tier_thresholds}")

    # ─── Public API ────────────────────────────────────────────────────────

    def route(self, features: dict) -> RoutingDecision:
        """
        Nhận dict features, trả về RoutingDecision với tier và routing_score.

        features keys:
          - motion_mean: float (mean optical flow magnitude)
          - person_count: int
          - scene_complexity: float (entropy)
          - camera_id: str
          - timestamp: float (unix seconds)
        """
        motion_norm = self._normalize(features.get("motion_mean", 0.0), "motion_mean")
        person_norm = self._normalize(float(features.get("person_count", 0)), "person_count")
        complexity_norm = self._normalize(features.get("scene_complexity", 0.0), "scene_complexity")
        risk_numeric = self._risk_numeric(features.get("camera_id", "default"))
        is_night = self._is_night(features.get("timestamp", 0.0))

        score = self.compute_score(
            motion_norm=motion_norm,
            person_norm=person_norm,
            complexity_norm=complexity_norm,
            risk_numeric=risk_numeric,
            is_night=is_night,
        )

        tier = self._score_to_tier(score)

        return RoutingDecision(
            tier=tier,
            routing_score=score,
            motion_norm=motion_norm,
            person_norm=person_norm,
            complexity_norm=complexity_norm,
            risk_numeric=risk_numeric,
            is_night=is_night,
            camera_id=features.get("camera_id", "default"),
        )

    def compute_score(
        self,
        motion_norm: float,
        person_norm: float,
        complexity_norm: float,
        risk_numeric: float,
        is_night: bool = False,
    ) -> float:
        """
        Tính routing_score từ normalized signals.
        Clamp về [0, 1].
        """
        score = (
            self.weights["motion"] * motion_norm
            + self.weights["person"] * person_norm
            + self.weights["complexity"] * complexity_norm
            + self.weights["risk"] * risk_numeric
        )
        if is_night:
            score *= self.night_multiplier
        return min(float(score), 1.0)

    # ─── Internal helpers ──────────────────────────────────────────────────

    def _normalize(self, value: float, key: str) -> float:
        """Normalize giá trị về [0, 1] dựa trên range config."""
        upper = self.norm_ranges.get(key, 1.0)
        if upper <= 0:
            return 0.0
        return min(value / upper, 1.0)

    def _score_to_tier(self, score: float) -> int:
        if score < self.tier_thresholds[0]:
            return 1
        elif score < self.tier_thresholds[1]:
            return 2
        else:
            return 3

    def _risk_numeric(self, camera_id: str) -> float:
        """Lookup risk_numeric từ camera_config. Default 0.3 (low risk)."""
        cam = self.camera_config.get("cameras", {})
        info = cam.get(camera_id) or cam.get("default", {})
        return float(info.get("risk_numeric", 0.3))

    def _is_night(self, timestamp: float) -> bool:
        """is_night = True nếu giờ < 6 hoặc > 22."""
        import datetime
        if timestamp <= 0:
            return False
        hour = datetime.datetime.fromtimestamp(timestamp).hour
        return hour < 6 or hour >= 22
