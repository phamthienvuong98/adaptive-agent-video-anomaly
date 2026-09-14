"""
tests/test_fast_filter.py
──────────────────────────
Unit tests cho Fast Filter Agent.
Chạy: pytest tests/test_fast_filter.py -v
"""

import numpy as np
import pytest


def make_random_frame(h=224, w=224):
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


class TestAdaptiveThreshold:
    def test_uses_base_threshold_before_warmup(self):
        from agents.fast_filter_agent import AdaptiveThreshold
        at = AdaptiveThreshold(base_threshold=0.45, warmup=300)
        assert at.current == 0.45

    def test_adapts_after_warmup(self):
        from agents.fast_filter_agent import AdaptiveThreshold
        at = AdaptiveThreshold(base_threshold=0.45, warmup=10, std_multiplier=2.0)
        # Đẩy nhiều score thấp vào
        for _ in range(15):
            at.update(0.1)
        # Threshold phải thấp hơn base (vì mean thấp)
        assert at.current < 0.45

    def test_high_scores_raise_threshold(self):
        from agents.fast_filter_agent import AdaptiveThreshold
        at = AdaptiveThreshold(base_threshold=0.45, warmup=10, std_multiplier=2.0)
        for _ in range(15):
            at.update(0.3)
        threshold_mid = at.current
        # Reset và push score cao hơn
        at2 = AdaptiveThreshold(base_threshold=0.45, warmup=10, std_multiplier=2.0)
        for _ in range(15):
            at2.update(0.7)
        assert at2.current > threshold_mid


class TestTemporalScoreBuffer:
    def test_weighted_mean_favors_recent(self):
        from agents.fast_filter_agent import TemporalScoreBuffer
        buf = TemporalScoreBuffer(window=4, strategy="weighted_mean")
        buf.push(0.1)
        buf.push(0.1)
        buf.push(0.1)
        score = buf.push(0.9)
        # Weighted mean phải lớn hơn simple mean (0.3) vì 0.9 được trọng số cao hơn
        simple_mean = (0.1 + 0.1 + 0.1 + 0.9) / 4
        assert score > simple_mean

    def test_max_strategy(self):
        from agents.fast_filter_agent import TemporalScoreBuffer
        buf = TemporalScoreBuffer(window=5, strategy="max")
        for s in [0.1, 0.2, 0.8, 0.3, 0.2]:
            result = buf.push(s)
        assert result == 0.8


class TestVideoPreprocessor:
    def test_process_single_frame(self):
        from utils.video_preprocessor import VideoPreprocessor
        pp = VideoPreprocessor({"strategy": "uniform", "sample_rate": 1})
        frame = make_random_frame(480, 640)
        fd = pp.process_single(frame, index=1)
        assert fd.frame.shape == (224, 224, 3)
        assert fd.index == 1

    def test_framedata_fields(self):
        from utils.video_preprocessor import FrameData
        fd = FrameData(frame=make_random_frame(), index=5, timestamp=1.2, motion_score=0.03)
        assert fd.camera_id == "default"
        assert fd.timestamp == 1.2


class TestRAGMemory:
    def test_add_and_retrieve(self, tmp_path):
        from utils.rag_memory import RAGMemory
        cfg = {"enabled": True, "top_k": 2, "per_camera_memory": True, "memory_dir": str(tmp_path)}
        mem = RAGMemory(cfg)
        mem.add_normal_scene("Người đi bộ bình thường trên vỉa hè", camera_id="cam_01")
        mem.add_normal_scene("Xe máy dừng đèn đỏ", camera_id="cam_01")
        results = mem.retrieve("hành vi đáng ngờ", camera_id="cam_01")
        assert len(results) <= 2
        assert mem.size("cam_01") == 2

    def test_returns_empty_for_new_camera(self, tmp_path):
        from utils.rag_memory import RAGMemory
        cfg = {"enabled": True, "top_k": 3, "per_camera_memory": True, "memory_dir": str(tmp_path)}
        mem = RAGMemory(cfg)
        results = mem.retrieve("test", camera_id="cam_99")
        assert results == []
