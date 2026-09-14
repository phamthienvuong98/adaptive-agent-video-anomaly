"""
agents/deep_reasoning_agent.py
───────────────────────────────
Tầng 2: Deep Reasoning Agent
- LLM Orchestrator: dynamic query refinement (QVAD-style)
- VLM Visual Reasoner: frame captioning & spatial reasoning
- RAG Memory: scene-specific normality rules

Chỉ được gọi khi tầng 1 phát hiện frame đáng ngờ (~5% frame).
Tham khảo: QVAD (arXiv 2604.03040), SlowFastVAD (arXiv 2504.10320)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from models.tier3_vlm import VLMReasoner
from models.llm_orchestrator import LLMOrchestrator
from utils.per_camera_memory import PerCameraMemory as RAGMemory
from utils.video_preprocessor import FrameData


@dataclass
class ReasoningResult:
    is_anomaly: bool
    anomaly_class: Optional[str]       # e.g. "fighting", "robbery"
    confidence: float                  # [0, 1]
    explanation: str                   # Ngôn ngữ tự nhiên
    bounding_regions: list[dict]       # [{"label": ..., "region": "upper-left"}]
    raw_vlm_caption: str
    query_rounds: int                  # Số vòng refinement đã dùng
    evidence: list[str] = field(default_factory=list)  # Các dấu hiệu phát hiện


class DeepReasoningAgent:
    """
    Agent lý luận sâu với vòng lặp VLM ↔ LLM.

    Pipeline (QVAD-style dynamic dialogue):
        1. LLM tạo câu hỏi ban đầu dựa trên anomaly score
        2. VLM trả lời câu hỏi từ visual context
        3. LLM đánh giá câu trả lời, có thể refinement thêm
        4. RAG bổ sung context về scene-specific normality
        5. LLM đưa ra kết luận cuối (anomaly class + confidence)
    """

    def __init__(self, config: dict, device: str = "cuda"):
        self.cfg = config
        self.max_rounds = config.get("vlm_reasoner", {}).get("max_query_rounds", 3)

        self.vlm = VLMReasoner(config.get("vlm_reasoner", {}), device=device)
        self.llm = LLMOrchestrator(config.get("llm_orchestrator", {}), device=device)
        self.rag = RAGMemory(config.get("rag_memory", {}))

    def analyze(
        self,
        frame_data: FrameData,
        routing_score: float = 0.8,
        motion_intensity: float = 0.5,
    ) -> ReasoningResult:
        """
        Phân tích frame Tier 3 qua vòng lặp VLM-LLM (tối đa max_rounds=2).
        Được gọi khi routing_score > 0.70.
        """
        camera_id = getattr(frame_data, "camera_id", "default")

        # ── Lấy initial VLM caption để làm RAG query (semantic search) ───
        initial_caption = self.vlm.describe_scene(frame_data.frame)
        scene_rules = self.rag.retrieve(
            query=initial_caption,
            camera_id=camera_id,
        )

        # ── Vòng 1: LLM tạo câu hỏi phân tích ban đầu ────────────────────
        initial_question = self.llm.generate_initial_query(
            anomaly_score=routing_score,
            motion_intensity=motion_intensity,
            scene_rules=scene_rules,
        )

        vlm_caption = ""
        llm_assessment = {}
        rounds_used = 0

        for round_idx in range(self.max_rounds):
            rounds_used = round_idx + 1

            # ── VLM trả lời câu hỏi từ ảnh ──────────────────────────────
            vlm_response = self.vlm.answer_query(
                frame=frame_data.frame,
                question=initial_question,
                history_caption=vlm_caption or initial_caption,
            )
            vlm_caption = vlm_response

            # ── LLM đánh giá câu trả lời VLM ────────────────────────────
            llm_assessment = self.llm.evaluate_and_decide(
                vlm_response=vlm_response,
                anomaly_score=filter_result.anomaly_score,
                scene_rules=scene_rules,
                round_num=round_idx,
            )

            # Nếu LLM đã đủ tự tin → dừng sớm
            if llm_assessment.get("confidence", 0) >= 0.75:
                logger.debug(f"  Early stop tại vòng {rounds_used} (confidence={llm_assessment['confidence']:.2f})")
                break

            # Refinement: LLM tạo câu hỏi tinh chỉnh hơn
            if round_idx < self.max_rounds - 1:
                initial_question = self.llm.generate_refined_query(
                    previous_response=vlm_response,
                    assessment=llm_assessment,
                )

        # ── Tổng hợp kết quả cuối ─────────────────────────────────────────
        is_anomaly = (
            llm_assessment.get("is_anomaly", False)
            and llm_assessment.get("confidence", 0) >= self.cfg.get("trigger_threshold", 0.45)
        )

        result = ReasoningResult(
            is_anomaly=is_anomaly,
            anomaly_class=llm_assessment.get("anomaly_class"),
            confidence=llm_assessment.get("confidence", 0.0),
            explanation=llm_assessment.get("explanation", ""),
            bounding_regions=llm_assessment.get("bounding_regions", []),
            raw_vlm_caption=vlm_caption,
            query_rounds=rounds_used,
            evidence=llm_assessment.get("evidence", []),
        )

        # ── Cập nhật RAG memory với kết quả ─────────────────────────────
        if not is_anomaly:
            self.rag.add_normal_scene(
                caption=vlm_caption,
                camera_id=camera_id,
                timestamp=frame_data.timestamp,
                routing_score=routing_score,
            )

        return result

    def build_rag_memory(self, normal_video_path: str, camera_id: str = "default") -> None:
        """
        Xây dựng RAG memory từ video bình thường của một camera.
        Gọi sau calibrate() trong Orchestrator.
        """
        import cv2
        cap = cv2.VideoCapture(normal_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        # Lấy 1 frame mỗi 2 giây để build memory
        sample_interval = int(fps * 2)
        count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if count % sample_interval == 0:
                caption = self.vlm.describe_scene(frame)
                self.rag.add_normal_scene(
                    caption=caption,
                    camera_id=camera_id,
                    timestamp=count / fps,
                )
            count += 1

        cap.release()
        logger.info(f"RAG memory built: {self.rag.size(camera_id)} entries cho camera {camera_id}")
