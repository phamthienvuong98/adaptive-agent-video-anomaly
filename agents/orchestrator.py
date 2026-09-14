"""
agents/orchestrator.py
───────────────────────
VAD Orchestrator — điều phối pipeline 3-tier với multi-signal routing.

Flow:
  Frame → FeatureExtractor (optical_flow + scene_embedding + pose + metadata)
        → RoutingAgent (compute_score → tier 1/2/3)
        → Tier 1: MobileNetV3 anomaly scorer  (~94% frames, ~10ms)
        → Tier 2: CLIP zero-shot              (~5% frames, ~50ms)
        → Tier 3: Qwen2.5-VL dialogue loop   (~1% frames, ~500ms)
        → AlertEngine (confidence gate + cooldown)

Tham khảo: PLAN.md §4, Cerberus (2510.16290), QVAD (2604.03040)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Optional

import yaml
from loguru import logger

from agents.routing_agent import RoutingAgent, RoutingDecision
from agents.deep_reasoning_agent import DeepReasoningAgent, ReasoningResult
from features.optical_flow import OpticalFlowExtractor
from features.scene_embedding import SceneEmbedder
from features.pose_detector import PoseDetector
from utils.video_preprocessor import VideoPreprocessor, FrameData
from utils.alert_engine import AlertEngine, Alert


@dataclass
class AgentConfig:
    routing_agent: dict = field(default_factory=dict)
    deep_reasoning: dict = field(default_factory=dict)
    alert_engine: dict = field(default_factory=dict)
    system: dict = field(default_factory=dict)
    datasets: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str) -> "AgentConfig":
        with open(path) as f:
            cfg = yaml.safe_load(f)
        # Load routing_config.yaml nếu có path riêng
        routing_cfg = cfg.get("routing_agent", {})
        routing_path = routing_cfg.get("config_path")
        if routing_path and Path(routing_path).exists():
            with open(routing_path) as f2:
                routing_full = yaml.safe_load(f2)
            routing_cfg = routing_full.get("routing_agent", routing_cfg)

        # Load camera_config.yaml nếu có
        camera_cfg = {}
        camera_path = cfg.get("system", {}).get("camera_config_path")
        if camera_path and Path(camera_path).exists():
            with open(camera_path) as f3:
                camera_cfg = yaml.safe_load(f3)

        obj = cls(
            routing_agent=routing_cfg,
            deep_reasoning=cfg.get("deep_reasoning", {}),
            alert_engine=cfg.get("alert_engine", {}),
            system=cfg.get("system", {}),
            datasets=cfg.get("datasets", {}),
        )
        obj._camera_config = camera_cfg
        return obj


@dataclass
class PipelineStats:
    total_frames: int = 0
    tier1_frames: int = 0
    tier2_frames: int = 0
    tier3_frames: int = 0
    alerts_fired: int = 0
    total_time_s: float = 0.0

    @property
    def fps(self) -> float:
        return self.total_frames / max(self.total_time_s, 1e-6)

    @property
    def tier_distribution(self) -> dict:
        n = max(self.total_frames, 1)
        return {
            "tier1": self.tier1_frames / n,
            "tier2": self.tier2_frames / n,
            "tier3": self.tier3_frames / n,
        }

    def summary(self) -> str:
        dist = self.tier_distribution
        return (
            f"Frames: {self.total_frames} | FPS: {self.fps:.1f} | "
            f"T1: {dist['tier1']:.1%} T2: {dist['tier2']:.1%} T3: {dist['tier3']:.1%} | "
            f"Alerts: {self.alerts_fired}"
        )


# Tier 1 escalation threshold: nếu Tier 1 score cao → escalate Tier 2
_TIER1_ESCALATE_THRESHOLD = 0.5
# Tier 2 escalation threshold: nếu CLIP score cao → escalate Tier 3
_TIER2_ESCALATE_THRESHOLD = 0.65


class VADOrchestrator:
    """
    Điều phối pipeline 3-tier với multi-signal routing agent.
    """

    def __init__(self, config_path: str = "configs/agent_config.yaml"):
        self.cfg = AgentConfig.from_yaml(config_path)
        device = self.cfg.system.get("device", "cuda")
        camera_config = getattr(self.cfg, "_camera_config", {})

        logger.info("Khởi tạo RoutingAgent...")
        self.routing_agent = RoutingAgent(
            config=self.cfg.routing_agent,
            camera_config=camera_config,
        )

        logger.info("Khởi tạo Feature Extractors...")
        self.flow_extractor = OpticalFlowExtractor()
        self.scene_embedder = SceneEmbedder(device=device)
        self.pose_detector = PoseDetector()

        logger.info("Khởi tạo DeepReasoningAgent (Tier 3)...")
        self.deep_agent = DeepReasoningAgent(self.cfg.deep_reasoning, device=device)

        logger.info("Khởi tạo Tier 1 và Tier 2 models...")
        self._tier1_model = None  # Lazy load
        self._tier2_model = None  # Lazy load
        self._device = device

        self.preprocessor = VideoPreprocessor(
            self.cfg.system.get("frame_sampling", {"strategy": "uniform", "sample_rate": 1})
        )
        self.alert_engine = AlertEngine(self.cfg.alert_engine)
        self.stats = PipelineStats()

    # ─── Public API ────────────────────────────────────────────────────────

    def process_video(self, source: str | int) -> Generator[Optional[Alert], None, PipelineStats]:
        """
        Generator: yield Alert khi phát hiện bất thường, None nếu bình thường.
        """
        t_start = time.perf_counter()
        logger.info(f"Bắt đầu xử lý: {source}")

        prev_frame = None

        for frame_data in self.preprocessor.stream(source):
            self.stats.total_frames += 1

            # ── Feature Extraction ──────────────────────────────────────
            flow_features = self.flow_extractor.compute(prev_frame, frame_data.frame)
            scene_complexity = self.scene_embedder.scene_complexity(frame_data.frame)
            pose_features = self.pose_detector.detect(frame_data.frame)
            prev_frame = frame_data.frame

            features = {
                "motion_mean": flow_features["mean"],
                "person_count": pose_features["person_count"],
                "scene_complexity": scene_complexity,
                "camera_id": frame_data.camera_id,
                "timestamp": frame_data.timestamp,
            }

            # ── Routing ─────────────────────────────────────────────────
            decision: RoutingDecision = self.routing_agent.route(features)

            logger.debug(
                f"Frame {frame_data.index} → Tier {decision.tier} "
                f"(score={decision.routing_score:.3f})"
            )

            # ── Tier Execution ───────────────────────────────────────────
            alert = self._execute_tier(frame_data, decision, flow_features)

            if alert:
                self.stats.alerts_fired += 1
                logger.warning(f"ALERT [{alert.anomaly_class}] confidence={alert.confidence:.2f}")

            yield alert

        self.stats.total_time_s = time.perf_counter() - t_start
        logger.info(f"Hoàn thành. {self.stats.summary()}")
        return self.stats

    def process_frame(self, frame_data: FrameData, prev_frame=None) -> Optional[Alert]:
        """Xử lý một frame đơn lẻ (dùng cho real-time API)."""
        flow_features = self.flow_extractor.compute(prev_frame, frame_data.frame)
        scene_complexity = self.scene_embedder.scene_complexity(frame_data.frame)
        pose_features = self.pose_detector.detect(frame_data.frame)

        features = {
            "motion_mean": flow_features["mean"],
            "person_count": pose_features["person_count"],
            "scene_complexity": scene_complexity,
            "camera_id": frame_data.camera_id,
            "timestamp": frame_data.timestamp,
        }
        decision = self.routing_agent.route(features)
        return self._execute_tier(frame_data, decision, flow_features)

    def get_stats(self) -> PipelineStats:
        return self.stats

    # ─── Internal ──────────────────────────────────────────────────────────

    def _execute_tier(
        self,
        frame_data: FrameData,
        decision: RoutingDecision,
        flow_features: dict,
    ) -> Optional[Alert]:
        """Thực thi tier được chọn bởi routing agent."""
        tier = decision.tier

        if tier == 1:
            return self._run_tier1(frame_data, decision)
        elif tier == 2:
            return self._run_tier2(frame_data, decision)
        else:
            return self._run_tier3(frame_data, decision, flow_features)

    def _run_tier1(self, frame_data: FrameData, decision: RoutingDecision) -> Optional[Alert]:
        """Tier 1: MobileNetV3 anomaly scorer (~10ms)."""
        self.stats.tier1_frames += 1

        model = self._get_tier1()
        anomaly_score = model.predict_score(frame_data.frame)

        # Safety net: Tier 1 escalate nếu score cao
        if anomaly_score > _TIER1_ESCALATE_THRESHOLD:
            logger.debug(f"  Tier 1 escalate → Tier 2 (anomaly_score={anomaly_score:.3f})")
            return self._run_tier2(frame_data, decision)

        # Frame bình thường → lưu vào per-camera memory
        self.deep_agent.rag.add_normal_scene(
            caption=f"normal frame score={anomaly_score:.3f}",
            camera_id=frame_data.camera_id,
            timestamp=frame_data.timestamp,
            routing_score=decision.routing_score,
        )
        return None

    def _run_tier2(self, frame_data: FrameData, decision: RoutingDecision) -> Optional[Alert]:
        """Tier 2: CLIP zero-shot (~50ms)."""
        self.stats.tier2_frames += 1

        model = self._get_tier2()
        clip_score = model.anomaly_score(frame_data.frame)

        if clip_score > _TIER2_ESCALATE_THRESHOLD:
            logger.debug(f"  Tier 2 escalate → Tier 3 (clip_score={clip_score:.3f})")
            # Rebuild routing decision với score cao hơn để Tier 3 nhận đúng context
            return self._run_tier3(
                frame_data,
                decision,
                flow_features={"mean": decision.motion_norm},
                clip_score=clip_score,
            )

        return None

    def _run_tier3(
        self,
        frame_data: FrameData,
        decision: RoutingDecision,
        flow_features: dict,
        clip_score: Optional[float] = None,
    ) -> Optional[Alert]:
        """Tier 3: Qwen2.5-VL dialogue loop (~500ms)."""
        self.stats.tier3_frames += 1

        reasoning: ReasoningResult = self.deep_agent.analyze(
            frame_data=frame_data,
            routing_score=clip_score or decision.routing_score,
            motion_intensity=flow_features.get("mean", decision.motion_norm),
        )

        alert = self.alert_engine.evaluate(
            frame_data=frame_data,
            reasoning=reasoning,
        )
        return alert

    def _get_tier1(self):
        if self._tier1_model is None:
            from models.tier1_mobilenet import LightweightCNN
            cfg = self.cfg.routing_agent.get("tier1", {})
            checkpoint = cfg.get("checkpoint", "checkpoints/student_cnn.pth")
            self._tier1_model = LightweightCNN(checkpoint=checkpoint, device=self._device)
        return self._tier1_model

    def _get_tier2(self):
        if self._tier2_model is None:
            from models.tier2_clip import CLIPEncoder
            self._tier2_model = CLIPEncoder(device=self._device)
        return self._tier2_model
