"""
models/vlm_reasoner.py
───────────────────────
VLM (Vision-Language Model) Wrapper.
Default: Qwen2.5-VL-7B-Instruct (open-source, cân bằng tốc độ/accuracy).
Fallback: LLaVA-1.6-mistral-7b.

Dùng trong Tầng 2 để trả lời câu hỏi về visual content của frame đáng ngờ.
"""

from __future__ import annotations

import base64
import io
from typing import Optional

import numpy as np
from loguru import logger
from PIL import Image


def _frame_to_base64(frame: np.ndarray) -> str:
    """Chuyển BGR numpy frame → base64 JPEG string."""
    rgb = frame[..., ::-1].copy()
    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


class VLMReasoner:
    """
    Wrapper cho VLM để phân tích frame đáng ngờ.
    Hỗ trợ: Qwen2.5-VL, LLaVA, InternVL2.
    """

    def __init__(self, config: dict, device: str = "cuda"):
        self.cfg = config
        self.device = device
        self.model_name = config.get("model_name", "Qwen/Qwen2.5-VL-7B-Instruct")
        self.max_new_tokens = config.get("max_new_tokens", 256)
        self.temperature = config.get("temperature", 0.1)
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return
        logger.info(f"Loading VLM: {self.model_name}")
        try:
            from transformers import AutoProcessor, AutoModelForVision2Seq
            self._processor = AutoProcessor.from_pretrained(self.model_name)
            self._model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map=self.device,
            )
            self._model.eval()
            logger.info("VLM loaded.")
        except Exception as e:
            logger.error(f"Lỗi load VLM: {e}")
            raise

    def answer_query(self, frame: np.ndarray, question: str, history_caption: str = "") -> str:
        """
        Trả lời câu hỏi về frame. Đây là hàm core của QVAD dialogue loop.
        """
        self._load()
        import torch

        context = ""
        if history_caption:
            context = f"Bối cảnh từ lần phân tích trước: {history_caption}\n\n"

        prompt = f"{context}Quan sát hình ảnh giám sát này và trả lời: {question}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        rgb = Image.fromarray(frame[..., ::-1].copy())
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[text], images=[rgb], return_tensors="pt")
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
            )

        input_len = inputs["input_ids"].shape[-1]
        response = self._processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
        return response.strip()

    def describe_scene(self, frame: np.ndarray) -> str:
        """
        Mô tả tổng quát cảnh trong frame (dùng khi build RAG memory).
        """
        return self.answer_query(
            frame=frame,
            question=(
                "Mô tả ngắn gọn (2-3 câu) những gì xảy ra trong cảnh giám sát này. "
                "Tập trung vào: số người, hoạt động, vị trí, không khí chung."
            ),
        )
