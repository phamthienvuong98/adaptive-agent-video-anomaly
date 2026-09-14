"""
models/llm_orchestrator.py
───────────────────────────
LLM Orchestrator: dynamic query refinement (QVAD-style).
Tạo câu hỏi động cho VLM, đánh giá câu trả lời, đưa ra kết luận cuối.

Default: Qwen3-4B (nhanh, đủ mạnh cho reasoning).
"""

from __future__ import annotations

import json
import re
from typing import Optional

from loguru import logger


SYSTEM_PROMPT = """Bạn là AI chuyên gia phân tích hành vi trong camera giám sát đô thị.
Nhiệm vụ:
1. Tạo câu hỏi phân tích thông minh để hướng dẫn VLM quan sát frame
2. Đánh giá câu trả lời VLM để xác định có bất thường không
3. Đưa ra kết luận JSON có cấu trúc

Các loại bất thường cần phát hiện:
- fighting (đánh nhau), robbery (cướp giật), vandalism (phá hoại)
- suspicious_loitering (đi vòng đáng ngờ), illegal_parking
- crowd_gathering (tụ tập đông người bất thường), weapon_detected
- accident (tai nạn), fire_smoke (lửa/khói), trespassing (xâm nhập)

Khi kết luận, luôn trả về JSON với format:
{
  "is_anomaly": true/false,
  "anomaly_class": "tên_lớp_hoặc_null",
  "confidence": 0.0-1.0,
  "explanation": "giải thích ngắn gọn",
  "evidence": ["dấu hiệu 1", "dấu hiệu 2"],
  "bounding_regions": [{"label": "mô tả", "region": "vị trí trong frame"}],
  "need_more_info": true/false
}"""


class LLMOrchestrator:
    """
    Điều phối LLM để tạo câu hỏi động và đánh giá kết quả VLM.
    """

    def __init__(self, config: dict, device: str = "cuda"):
        self.cfg = config
        self.device = device
        self.model_name = config.get("model_name", "Qwen/Qwen3-4B")
        self.max_new_tokens = config.get("max_new_tokens", 512)
        self.temperature = config.get("temperature", 0.0)
        self._pipeline = None

    def _load(self):
        if self._pipeline is not None:
            return
        logger.info(f"Loading LLM Orchestrator: {self.model_name}")
        try:
            from transformers import pipeline
            self._pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                device_map=self.device,
                torch_dtype="auto",
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=False,
            )
            logger.info("LLM loaded.")
        except Exception as e:
            logger.error(f"Lỗi load LLM: {e}")
            raise

    def _call(self, user_message: str) -> str:
        self._load()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        result = self._pipeline(messages)
        return result[0]["generated_text"][-1]["content"].strip()

    def generate_initial_query(
        self,
        anomaly_score: float,
        motion_intensity: float,
        scene_rules: list[str],
    ) -> str:
        """
        Tạo câu hỏi ban đầu cho VLM dựa trên anomaly score và context.
        """
        rules_text = "\n".join(f"- {r}" for r in scene_rules[:3]) if scene_rules else "Không có"

        prompt = f"""Anomaly score từ Fast Filter: {anomaly_score:.3f} (ngưỡng: 0.45)
Motion intensity: {motion_intensity:.4f}
Cảnh bình thường của camera này thường: 
{rules_text}

Tạo MỘT câu hỏi ngắn gọn (≤20 từ) để hỏi VLM về frame này. 
Câu hỏi phải tập trung vào dấu hiệu quan trọng nhất cần quan sát."""

        response = self._call(prompt)
        # Trích xuất câu hỏi (loại bỏ giải thích thừa)
        lines = [l.strip() for l in response.split("\n") if l.strip()]
        for line in lines:
            if "?" in line:
                return line
        return lines[0] if lines else "Có hành vi bất thường nào trong frame này không? Mô tả cụ thể."

    def evaluate_and_decide(
        self,
        vlm_response: str,
        anomaly_score: float,
        scene_rules: list[str],
        round_num: int,
    ) -> dict:
        """
        Đánh giá câu trả lời VLM và đưa ra quyết định (JSON structured output).
        """
        rules_text = "\n".join(f"- {r}" for r in scene_rules[:3]) if scene_rules else "Không có"

        prompt = f"""VLM quan sát và mô tả: "{vlm_response}"

Anomaly score từ model: {anomaly_score:.3f}
Vòng phân tích: {round_num + 1}
Chuẩn bình thường của cảnh này:
{rules_text}

Dựa trên thông tin trên, đưa ra kết luận dạng JSON."""

        response = self._call(prompt)
        return self._parse_json(response)

    def generate_refined_query(self, previous_response: str, assessment: dict) -> str:
        """
        Tạo câu hỏi tinh chỉnh khi LLM cần thêm thông tin từ VLM.
        """
        prompt = f"""VLM đã trả lời: "{previous_response}"
Đánh giá hiện tại: {json.dumps(assessment, ensure_ascii=False)}

Câu trả lời chưa đủ rõ ràng. Tạo câu hỏi follow-up cụ thể hơn (≤20 từ) 
để làm rõ điểm còn mơ hồ."""

        response = self._call(prompt)
        lines = [l.strip() for l in response.split("\n") if l.strip()]
        for line in lines:
            if "?" in line:
                return line
        return lines[0] if lines else "Mô tả chi tiết hơn về vị trí và hành động của người trong frame."

    def _parse_json(self, text: str) -> dict:
        """Trích xuất JSON từ response text."""
        # Tìm JSON block
        match = re.search(r'\{.*?\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Fallback: trả về dict mặc định
        logger.debug(f"Không parse được JSON từ: {text[:100]}...")
        is_anomaly = any(kw in text.lower() for kw in ["bất thường", "nguy hiểm", "anomaly", "suspicious"])
        return {
            "is_anomaly": is_anomaly,
            "anomaly_class": None,
            "confidence": 0.4 if is_anomaly else 0.1,
            "explanation": text[:200],
            "evidence": [],
            "bounding_regions": [],
            "need_more_info": True,
        }
