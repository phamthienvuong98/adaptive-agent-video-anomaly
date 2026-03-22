"""
Adaptive AI Agent — CORE CONTRIBUTION 2

The Agent acts as an ORCHESTRATOR (not just a post-processor).
It dynamically controls the entire inference pipeline to minimise
computational cost while maintaining detection accuracy.

Architecture:
    Agent
      │
      ├── gọi YOLO (cheap)         → Level 0
      ├── quyết định có gọi GCN    → Level 1
      ├── quyết định có gọi VLM    → Level 2 (expensive)
      └── update threshold theo context

Mechanisms:
    1. Cascading:     stop early if confident enough
    2. Adaptive threshold: T = f(time, location, history)
    3. Frame skipping: skip N% frames when scene is normal
    4. Region-of-Interest (ROI): only process motion regions
    5. Model selection: choose model complexity per situation

Key Insight:
    In real-world surveillance, ~95% of frames are normal.
    The Agent ensures heavy models (GCN, VLM) are called ONLY
    when necessary → massive compute savings.

References:
    - Cascading classifiers (Viola-Jones paradigm)
    - Adaptive computation (Graves, 2016)
    - Early exit networks
"""

import time
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class AgentAction(Enum):
    """Possible actions the agent can take."""
    SKIP = "skip"                   # Skip frame entirely
    STOP_NORMAL = "stop_normal"     # Confident normal → stop
    RUN_GCN = "run_gcn"            # Suspicious → run GCN
    RUN_VLM = "run_vlm"            # Complex → run heavy model
    FULL_PIPELINE = "full_pipeline"  # High alert → run everything


@dataclass
class AgentDecision:
    """Record of a single agent decision for logging/analysis."""
    timestamp: float
    frame_idx: int
    action: AgentAction
    yolo_score: float
    gcn_score: Optional[float] = None
    vlm_score: Optional[float] = None
    final_score: float = 0.0
    threshold_t1: float = 0.0
    threshold_t2: float = 0.0
    compute_cost: float = 0.0       # relative cost (0-1)
    is_anomaly: bool = False
    reason: str = ""


@dataclass
class AgentState:
    """Internal state of the agent, updated over time."""
    # History
    recent_scores: deque = field(default_factory=lambda: deque(maxlen=100))
    recent_actions: deque = field(default_factory=lambda: deque(maxlen=100))
    anomaly_count: int = 0
    total_frames: int = 0
    skip_count: int = 0

    # Adaptive thresholds
    current_t1: float = config.AGENT_T1
    current_t2: float = config.AGENT_T2

    # Scene stability
    scene_stability: float = 1.0    # 0 = unstable, 1 = stable
    consecutive_normal: int = 0
    consecutive_suspicious: int = 0

    # Compute tracking
    total_compute: float = 0.0
    yolo_calls: int = 0
    gcn_calls: int = 0
    vlm_calls: int = 0

    # Decisions log
    decisions: List[AgentDecision] = field(default_factory=list)


class AdaptiveAgent:
    """
    AI Agent = Orchestrator of the inference pipeline.

    The agent makes intelligent decisions about:
    1. WHICH frames to process (frame skipping)
    2. WHICH model to invoke (cascading)
    3. WHEN to escalate (adaptive thresholds)
    4. WHERE to focus (ROI)

    This is NOT just if-else logic — the thresholds adapt based on:
    - Time of day
    - Scene history (recent anomaly frequency)
    - Scene stability (motion patterns)
    - Computational budget
    """

    # Relative compute costs per level
    COMPUTE_COSTS = {
        AgentAction.SKIP: 0.0,
        AgentAction.STOP_NORMAL: 0.05,      # YOLO only
        AgentAction.RUN_GCN: 0.3,           # YOLO + GCN
        AgentAction.RUN_VLM: 1.0,           # full pipeline
        AgentAction.FULL_PIPELINE: 1.0,
    }

    def __init__(
        self,
        t1: float = config.AGENT_T1,
        t2: float = config.AGENT_T2,
        frame_skip_ratio: float = config.AGENT_FRAME_SKIP_RATIO,
        adaptive_threshold: bool = config.AGENT_ADAPTIVE_THRESHOLD,
        use_roi: bool = config.AGENT_USE_ROI,
        time_schedule: Optional[Dict] = None,
    ):
        self.base_t1 = t1
        self.base_t2 = t2
        self.frame_skip_ratio = frame_skip_ratio
        self.adaptive_threshold = adaptive_threshold
        self.use_roi = use_roi
        self.time_schedule = time_schedule or config.AGENT_TIME_SCHEDULE

        self.state = AgentState(current_t1=t1, current_t2=t2)

    # ==============================================================
    # Core Decision Logic
    # ==============================================================

    def decide(
        self,
        frame_idx: int,
        yolo_score: float,
        num_detections: int = 0,
        motion_magnitude: float = 0.0,
        current_time: Optional[datetime] = None,
    ) -> AgentDecision:
        """
        Main decision function — called for each frame.

        Cascading logic:
            Frame → YOLO (cheap)
              │
              ├─ score < T1 → STOP (normal) ✅
              │
              ├─ T1 ≤ score < T2 → RUN GCN
              │
              └─ score ≥ T2 → RUN VLM (expensive)

        Returns:
            AgentDecision with action to take
        """
        self.state.total_frames += 1

        # Update adaptive thresholds
        if self.adaptive_threshold:
            self._update_thresholds(current_time)

        t1 = self.state.current_t1
        t2 = self.state.current_t2

        # ── Mechanism 1: Frame Skipping ──
        if self._should_skip_frame(frame_idx, motion_magnitude):
            decision = AgentDecision(
                timestamp=time.time(),
                frame_idx=frame_idx,
                action=AgentAction.SKIP,
                yolo_score=yolo_score,
                final_score=0.0,
                threshold_t1=t1,
                threshold_t2=t2,
                compute_cost=0.0,
                is_anomaly=False,
                reason="Frame skipped (scene stable / low motion)",
            )
            self._record_decision(decision)
            return decision

        # ── Mechanism 2: Cascading Decision ──

        # Level 0: YOLO score analysis
        if yolo_score < t1:
            # Confident normal → STOP
            action = AgentAction.STOP_NORMAL
            reason = f"YOLO score {yolo_score:.3f} < T1={t1:.3f} → normal"
            self.state.consecutive_normal += 1
            self.state.consecutive_suspicious = 0

        elif yolo_score < t2:
            # Suspicious → need GCN for graph analysis
            action = AgentAction.RUN_GCN
            reason = f"T1={t1:.3f} ≤ YOLO score {yolo_score:.3f} < T2={t2:.3f} → run GCN"
            self.state.consecutive_suspicious += 1
            self.state.consecutive_normal = 0

        else:
            # High anomaly signal → escalate to heavy model
            if num_detections > 3:
                action = AgentAction.RUN_VLM
                reason = f"YOLO score {yolo_score:.3f} ≥ T2={t2:.3f} + {num_detections} objects → VLM"
            else:
                action = AgentAction.RUN_GCN
                reason = f"YOLO score {yolo_score:.3f} ≥ T2={t2:.3f} but few objects → GCN"
            self.state.consecutive_suspicious += 1
            self.state.consecutive_normal = 0

        # ── Mechanism 3: Escalation Override ──
        if self.state.consecutive_suspicious >= 5:
            action = AgentAction.FULL_PIPELINE
            reason += " [ESCALATED: 5+ consecutive suspicious]"

        decision = AgentDecision(
            timestamp=time.time(),
            frame_idx=frame_idx,
            action=action,
            yolo_score=yolo_score,
            final_score=yolo_score,
            threshold_t1=t1,
            threshold_t2=t2,
            compute_cost=self.COMPUTE_COSTS[action],
            is_anomaly=(yolo_score >= t1),
            reason=reason,
        )

        self._record_decision(decision)
        return decision

    def update_with_gcn_score(
        self, decision: AgentDecision, gcn_score: float
    ) -> AgentDecision:
        """
        Update decision after GCN result is available.
        May escalate to VLM if GCN score is high.
        """
        decision.gcn_score = gcn_score
        decision.final_score = gcn_score

        t2 = self.state.current_t2

        if gcn_score >= t2:
            # GCN says anomaly is complex → escalate
            decision.action = AgentAction.RUN_VLM
            decision.compute_cost = self.COMPUTE_COSTS[AgentAction.RUN_VLM]
            decision.reason += f" → GCN={gcn_score:.3f} ≥ T2 → escalate to VLM"
        elif gcn_score < self.state.current_t1:
            # GCN says actually normal
            decision.is_anomaly = False
            decision.reason += f" → GCN={gcn_score:.3f} < T1 → reclassified normal"
        else:
            decision.is_anomaly = True
            decision.reason += f" → GCN={gcn_score:.3f} → anomaly detected"

        return decision

    def update_with_vlm_score(
        self, decision: AgentDecision, vlm_score: float
    ) -> AgentDecision:
        """Update decision after VLM result."""
        decision.vlm_score = vlm_score
        decision.final_score = vlm_score
        decision.is_anomaly = (vlm_score >= 0.5)
        decision.compute_cost = self.COMPUTE_COSTS[AgentAction.RUN_VLM]
        decision.reason += f" → VLM={vlm_score:.3f}"
        self.state.vlm_calls += 1
        return decision

    # ==============================================================
    # Adaptive Threshold Mechanisms
    # ==============================================================

    def _update_thresholds(self, current_time: Optional[datetime] = None):
        """
        Adaptive threshold: T = f(time, location, history)

        Adjustments:
            1. Time-of-day modifier
            2. Recent anomaly rate
            3. Scene stability
        """
        t1 = self.base_t1
        t2 = self.base_t2

        # ── Time-based adjustment ──
        if current_time is not None:
            hour = current_time.hour
            for period, info in self.time_schedule.items():
                start_h, end_h = info["hours"]
                if start_h <= hour < end_h:
                    t1 += info["modifier"]
                    t2 += info["modifier"]
                    break

        # ── History-based adjustment ──
        if len(self.state.recent_scores) >= 10:
            recent_mean = np.mean(list(self.state.recent_scores))
            recent_std = np.std(list(self.state.recent_scores))

            # If recent anomaly rate is high → lower threshold (more sensitive)
            anomaly_rate = sum(
                1 for s in self.state.recent_scores if s > self.base_t1
            ) / len(self.state.recent_scores)

            if anomaly_rate > 0.3:
                t1 -= 0.05   # more sensitive
                t2 -= 0.05
            elif anomaly_rate < 0.05:
                t1 += 0.05   # less sensitive (save compute)
                t2 += 0.05

        # ── Stability-based adjustment ──
        stability = self.state.scene_stability
        if stability > 0.8:
            t1 += 0.02  # stable scene → raise threshold
        elif stability < 0.3:
            t1 -= 0.02  # unstable → lower threshold

        # Clamp thresholds
        self.state.current_t1 = max(0.1, min(0.6, t1))
        self.state.current_t2 = max(0.4, min(0.9, t2))

    def _should_skip_frame(
        self, frame_idx: int, motion_magnitude: float
    ) -> bool:
        """
        Frame skipping logic.

        Skip conditions:
            1. Scene has been stable (many consecutive normal)
            2. Low motion magnitude
            3. Not a keyframe (based on skip ratio)
        """
        # Never skip if recent suspicious activity
        if self.state.consecutive_suspicious > 0:
            return False

        # Skip based on stability
        if self.state.consecutive_normal > 10:
            # Very stable scene → aggressive skipping
            if frame_idx % max(1, int(1 / (1 - self.frame_skip_ratio))) != 0:
                self.state.skip_count += 1
                return True

        # Skip if very low motion
        if motion_magnitude < 0.01 and self.state.consecutive_normal > 5:
            self.state.skip_count += 1
            return True

        return False

    def _update_scene_stability(self, score: float):
        """Update scene stability metric based on recent score variance."""
        self.state.recent_scores.append(score)
        if len(self.state.recent_scores) >= 5:
            std = np.std(list(self.state.recent_scores)[-10:])
            self.state.scene_stability = max(0, 1 - std * 5)

    # ==============================================================
    # ROI (Region of Interest) Logic
    # ==============================================================

    def compute_roi_mask(
        self,
        frame: "np.ndarray",
        prev_frame: Optional["np.ndarray"] = None,
        threshold: float = 25.0,
    ) -> Optional["np.ndarray"]:
        """
        Compute ROI mask based on motion detection.
        Only regions with significant motion are processed.

        Returns:
            Binary mask (H, W) where 1 = process, 0 = skip
        """
        if not self.use_roi or prev_frame is None:
            return None

        import cv2

        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            prev_gray = prev_frame

        # Frame difference
        diff = cv2.absdiff(gray, prev_gray)
        _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, kernel, iterations=2)

        return mask

    def filter_detections_by_roi(
        self,
        detections: list,
        roi_mask: "np.ndarray",
    ) -> list:
        """Filter detections to only those within ROI regions."""
        if roi_mask is None:
            return detections

        filtered = []
        for det in detections:
            cx, cy = int(det.cx), int(det.cy)
            if 0 <= cy < roi_mask.shape[0] and 0 <= cx < roi_mask.shape[1]:
                if roi_mask[cy, cx] > 0:
                    filtered.append(det)
        return filtered

    # ==============================================================
    # Bookkeeping
    # ==============================================================

    def _record_decision(self, decision: AgentDecision):
        """Log decision and update statistics."""
        self.state.decisions.append(decision)
        self._update_scene_stability(decision.yolo_score)
        self.state.total_compute += decision.compute_cost

        if decision.action == AgentAction.STOP_NORMAL:
            self.state.yolo_calls += 1
        elif decision.action == AgentAction.RUN_GCN:
            self.state.yolo_calls += 1
            self.state.gcn_calls += 1
        elif decision.action in (AgentAction.RUN_VLM, AgentAction.FULL_PIPELINE):
            self.state.yolo_calls += 1
            self.state.gcn_calls += 1
            self.state.vlm_calls += 1

        if decision.is_anomaly:
            self.state.anomaly_count += 1

    def get_statistics(self) -> Dict[str, Any]:
        """Return agent performance statistics."""
        total = max(1, self.state.total_frames)
        return {
            "total_frames": self.state.total_frames,
            "skip_rate": self.state.skip_count / total,
            "anomaly_rate": self.state.anomaly_count / total,
            "yolo_calls": self.state.yolo_calls,
            "gcn_calls": self.state.gcn_calls,
            "vlm_calls": self.state.vlm_calls,
            "avg_compute_cost": self.state.total_compute / total,
            "current_t1": self.state.current_t1,
            "current_t2": self.state.current_t2,
            "scene_stability": self.state.scene_stability,
            "compute_saved_pct": (1 - self.state.total_compute / total) * 100,
        }

    def reset(self):
        """Reset agent state for new video / evaluation."""
        self.state = AgentState(
            current_t1=self.base_t1,
            current_t2=self.base_t2,
        )


# ==============================================================
# Demonstration
# ==============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("AI Agent — Adaptive Inference Controller")
    print("=" * 60)

    agent = AdaptiveAgent(t1=0.3, t2=0.7)

    # Simulate 100 frames (95% normal, 5% anomaly)
    np.random.seed(42)
    scores = np.concatenate([
        np.random.uniform(0.0, 0.2, 80),   # normal
        np.random.uniform(0.1, 0.4, 10),   # borderline
        np.random.uniform(0.5, 0.95, 5),   # anomaly
        np.random.uniform(0.0, 0.15, 5),   # normal again
    ])

    for i, score in enumerate(scores):
        decision = agent.decide(
            frame_idx=i,
            yolo_score=score,
            num_detections=np.random.randint(0, 8),
            motion_magnitude=score * 0.5,
        )

        if decision.action != AgentAction.SKIP:
            if decision.action == AgentAction.RUN_GCN:
                gcn_score = score * 1.1 + np.random.uniform(-0.1, 0.1)
                agent.update_with_gcn_score(decision, gcn_score)
            elif decision.action == AgentAction.RUN_VLM:
                vlm_score = score * 1.2 + np.random.uniform(-0.05, 0.05)
                agent.update_with_vlm_score(decision, vlm_score)

    # Print statistics
    stats = agent.get_statistics()
    print(f"\n📊 Agent Statistics:")
    print(f"  Total frames:     {stats['total_frames']}")
    print(f"  Skip rate:        {stats['skip_rate']:.1%}")
    print(f"  Anomaly rate:     {stats['anomaly_rate']:.1%}")
    print(f"  YOLO calls:       {stats['yolo_calls']}")
    print(f"  GCN calls:        {stats['gcn_calls']}")
    print(f"  VLM calls:        {stats['vlm_calls']}")
    print(f"  Avg compute cost: {stats['avg_compute_cost']:.3f}")
    print(f"  Compute saved:    {stats['compute_saved_pct']:.1f}%")
    print(f"  Current T1:       {stats['current_t1']:.3f}")
    print(f"  Current T2:       {stats['current_t2']:.3f}")
    print(f"  Scene stability:  {stats['scene_stability']:.3f}")
