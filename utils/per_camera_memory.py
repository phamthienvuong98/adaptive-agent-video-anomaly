"""
utils/per_camera_memory.py
───────────────────────────
Per-camera FAISS memory store cho scene normality context.
Chỉ lưu frame bình thường (routing_score < ANOMALY_GATE) để xây baseline
normality cho từng camera. Khi Tier 3 cần context, retrieve các captions
gần nhất để hỗ trợ VLM phán đoán.

Anomaly gate: routing_score >= 0.35 → KHÔNG lưu (frame suspicious).

Tham khảo: SlowFastVAD (arXiv 2504.10320), Cerberus (arXiv 2510.16290)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger


@dataclass
class MemoryEntry:
    caption: str
    camera_id: str
    timestamp: float
    embedding: Optional[np.ndarray] = field(default=None, repr=False)


ANOMALY_GATE_THRESHOLD = 0.35  # routing_score >= này → không lưu vào memory


class PerCameraMemory:
    """
    Per-camera FAISS memory store. Chỉ lưu frame bình thường (routing_score < 0.35).
    Cold start (camera mới): memory rỗng → Tier 3 vẫn chạy, chỉ thiếu RAG context.
    """

    def __init__(self, config: dict):
        self.cfg = config
        self.enabled = config.get("enabled", True)
        self.top_k = config.get("top_k", 5)
        self.per_camera = config.get("per_camera_memory", True)
        self.memory_dir = Path(config.get("memory_dir", "data/rag_memory/"))
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        # {camera_id: list[MemoryEntry]}
        self._entries: dict[str, list[MemoryEntry]] = {}
        # {camera_id: faiss.Index}
        self._indexes: dict = {}

        self._embedder = None  # Lazy load

    def _get_embedder(self):
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                model_name = self.cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
                self._embedder = SentenceTransformer(model_name)
                logger.info(f"RAG embedder loaded: {model_name}")
            except ImportError:
                logger.warning("sentence-transformers chưa cài. RAG sẽ dùng keyword matching.")
        return self._embedder

    def _embed(self, text: str) -> np.ndarray:
        embedder = self._get_embedder()
        if embedder is None:
            # Fallback: random vector (placeholder khi chưa cài sentence-transformers)
            return np.random.randn(384).astype(np.float32)
        return embedder.encode([text], convert_to_numpy=True)[0]

    def _get_index(self, camera_id: str):
        """Lazy-init FAISS index cho camera."""
        if camera_id not in self._indexes:
            try:
                import faiss
                self._indexes[camera_id] = faiss.IndexFlatL2(384)
            except ImportError:
                logger.warning("FAISS chưa cài. RAG dùng linear search.")
                self._indexes[camera_id] = None
        return self._indexes[camera_id]

    def add_normal_scene(
        self,
        caption: str,
        camera_id: str,
        timestamp: float = 0.0,
        routing_score: float = 0.0,
    ) -> None:
        """
        Thêm mô tả cảnh bình thường vào memory.
        Anomaly gate: bỏ qua nếu routing_score >= ANOMALY_GATE_THRESHOLD.
        """
        if not self.enabled:
            return
        if routing_score >= ANOMALY_GATE_THRESHOLD:
            return  # Frame suspicious — không lưu vào normality baseline

        key = camera_id if self.per_camera else "global"
        if key not in self._entries:
            self._entries[key] = []

        embedding = self._embed(caption)
        entry = MemoryEntry(caption=caption, camera_id=camera_id, timestamp=timestamp, embedding=embedding)
        self._entries[key].append(entry)

        idx = self._get_index(key)
        if idx is not None:
            idx.add(embedding.reshape(1, -1))

    def retrieve(self, query: str, camera_id: str, top_k: Optional[int] = None) -> list[str]:
        """
        Retrieve các mô tả cảnh bình thường gần nhất với query.
        Trả về list các caption text.
        """
        if not self.enabled:
            return []

        k = top_k or self.top_k
        key = camera_id if self.per_camera else "global"
        entries = self._entries.get(key, [])

        if not entries:
            return []

        query_emb = self._embed(query)
        idx = self._get_index(key)

        if idx is not None and idx.ntotal > 0:
            # FAISS search
            distances, indices = idx.search(query_emb.reshape(1, -1), min(k, len(entries)))
            results = [entries[i].caption for i in indices[0] if i < len(entries)]
        else:
            # Linear fallback
            scores = [
                np.dot(query_emb, e.embedding) / (np.linalg.norm(query_emb) * np.linalg.norm(e.embedding) + 1e-8)
                for e in entries if e.embedding is not None
            ]
            top_indices = np.argsort(scores)[-k:][::-1]
            results = [entries[i].caption for i in top_indices]

        return results

    def size(self, camera_id: str) -> int:
        key = camera_id if self.per_camera else "global"
        return len(self._entries.get(key, []))

    def save(self, camera_id: str) -> None:
        """Lưu memory ra disk."""
        key = camera_id if self.per_camera else "global"
        path = self.memory_dir / f"{key}_memory.json"
        data = [
            {"caption": e.caption, "camera_id": e.camera_id, "timestamp": e.timestamp}
            for e in self._entries.get(key, [])
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(data)} memory entries to {path}")

    def load(self, camera_id: str) -> None:
        """Load memory từ disk."""
        key = camera_id if self.per_camera else "global"
        path = self.memory_dir / f"{key}_memory.json"
        if not path.exists():
            return
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            self.add_normal_scene(item["caption"], item["camera_id"], item["timestamp"])
        logger.info(f"Loaded {len(data)} memory entries from {path}")
