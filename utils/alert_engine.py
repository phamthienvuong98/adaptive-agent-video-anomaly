"""
utils/alert_engine.py
──────────────────────
Alert Engine: nhận kết quả từ Deep Reasoning Agent,
đánh giá cooldown và confidence, tạo Alert output.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from loguru import logger

if TYPE_CHECKING:
    from agents.deep_reasoning_agent import ReasoningResult

from utils.video_preprocessor import FrameData


@dataclass
class Alert:
    camera_id: str
    frame_index: int
    timestamp: float
    anomaly_class: str
    confidence: float
    explanation: str
    evidence: list[str]
    bounding_regions: list[dict]

    def to_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "anomaly_class": self.anomaly_class,
            "confidence": round(self.confidence, 4),
            "explanation": self.explanation,
            "evidence": self.evidence,
            "bounding_regions": self.bounding_regions,
        }

    def __str__(self) -> str:
        return (
            f"[ALERT] {self.anomaly_class.upper()} | "
            f"cam={self.camera_id} | t={self.timestamp:.1f}s | "
            f"conf={self.confidence:.2f}\n"
            f"  → {self.explanation}"
        )


class AlertEngine:
    """
    Quản lý việc tạo và lọc Alert.
    - Chỉ alert khi confidence đủ cao
    - Cooldown: không alert cùng event trong N giây
    - Ghi log JSON và video annotated
    """

    def __init__(self, config: dict):
        self.cfg = config
        self.min_confidence = config.get("min_confidence", 0.6)
        self.cooldown_s = config.get("cooldown_seconds", 10)
        self.output_dir = Path("outputs/")
        self.output_dir.mkdir(exist_ok=True)

        # {(camera_id, anomaly_class): last_alert_time}
        self._cooldown_map: dict[tuple, float] = {}

    def evaluate(self, frame_data: FrameData, reasoning: ReasoningResult) -> Optional[Alert]:
        """
        Trả về Alert nếu đủ điều kiện, None nếu không.
        """
        if not reasoning.is_anomaly:
            return None

        if reasoning.confidence < self.min_confidence:
            logger.debug(
                f"Alert bị lọc: confidence {reasoning.confidence:.2f} < {self.min_confidence}"
            )
            return None

        # Kiểm tra cooldown
        key = (frame_data.camera_id, reasoning.anomaly_class)
        now = time.time()
        last_alert = self._cooldown_map.get(key, 0)
        if now - last_alert < self.cooldown_s:
            logger.debug(f"Alert trong cooldown: {key}")
            return None

        self._cooldown_map[key] = now

        alert = Alert(
            camera_id=frame_data.camera_id,
            frame_index=frame_data.index,
            timestamp=frame_data.timestamp,
            anomaly_class=reasoning.anomaly_class or "unknown",
            confidence=reasoning.confidence,
            explanation=reasoning.explanation,
            evidence=reasoning.evidence,
            bounding_regions=reasoning.bounding_regions,
        )

        self._log_alert(alert)
        return alert

    def _log_alert(self, alert: Alert) -> None:
        log_path = self.output_dir / "alerts.jsonl"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(alert.to_dict(), ensure_ascii=False) + "\n")
