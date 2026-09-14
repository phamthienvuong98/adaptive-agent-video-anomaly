"""
tests/test_deep_reasoning.py
─────────────────────────────
Unit tests cho Deep Reasoning components (không cần GPU/VLM thật).
Dùng mock để kiểm tra logic pipeline mà không cần load model nặng.

Chạy: pytest tests/test_deep_reasoning.py -v
"""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest


def make_frame(h=224, w=224):
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


# ─── Tests cho ReasoningResult dataclass ──────────────────────────────────────

class TestReasoningResult:
    def test_default_evidence_is_empty_list(self):
        from agents.deep_reasoning_agent import ReasoningResult
        r = ReasoningResult(
            is_anomaly=False,
            anomaly_class=None,
            confidence=0.1,
            explanation="bình thường",
            bounding_regions=[],
            raw_vlm_caption="người đi bộ",
            query_rounds=1,
        )
        assert r.evidence == []

    def test_anomaly_result_fields(self):
        from agents.deep_reasoning_agent import ReasoningResult
        r = ReasoningResult(
            is_anomaly=True,
            anomaly_class="fighting",
            confidence=0.87,
            explanation="Hai người đang đánh nhau gần cột đèn",
            bounding_regions=[{"label": "người đánh nhau", "region": "center"}],
            raw_vlm_caption="Hai người đàn ông đang vật lộn",
            query_rounds=2,
            evidence=["tư thế đánh đấm", "khoảng cách gần"],
        )
        assert r.is_anomaly is True
        assert r.anomaly_class == "fighting"
        assert r.confidence == pytest.approx(0.87)
        assert len(r.evidence) == 2
        assert len(r.bounding_regions) == 1


# ─── Tests cho LLMOrchestrator._parse_json ────────────────────────────────────

class TestLLMOrchestratorParsing:
    def _make_orchestrator(self):
        from models.llm_orchestrator import LLMOrchestrator
        return LLMOrchestrator(config={}, device="cpu")

    def test_parse_valid_json(self):
        llm = self._make_orchestrator()
        text = '{"is_anomaly": true, "anomaly_class": "robbery", "confidence": 0.82, "explanation": "test", "evidence": [], "bounding_regions": [], "need_more_info": false}'
        result = llm._parse_json(text)
        assert result["is_anomaly"] is True
        assert result["anomaly_class"] == "robbery"
        assert result["confidence"] == pytest.approx(0.82)

    def test_parse_json_embedded_in_text(self):
        llm = self._make_orchestrator()
        text = 'Dựa trên phân tích: {"is_anomaly": false, "confidence": 0.15, "anomaly_class": null, "explanation": "bình thường", "evidence": [], "bounding_regions": [], "need_more_info": false} Đây là kết luận.'
        result = llm._parse_json(text)
        assert result["is_anomaly"] is False

    def test_fallback_on_invalid_json(self):
        llm = self._make_orchestrator()
        text = "Đây là văn bản không có JSON. Có dấu hiệu bất thường và nguy hiểm."
        result = llm._parse_json(text)
        # fallback phải trả về dict hợp lệ
        assert isinstance(result, dict)
        assert "is_anomaly" in result
        assert "confidence" in result
        assert result["is_anomaly"] is True  # vì có keyword "bất thường"

    def test_fallback_normal_text(self):
        llm = self._make_orchestrator()
        text = "Cảnh hoàn toàn bình thường, không có gì đáng ngờ."
        result = llm._parse_json(text)
        assert result["is_anomaly"] is False


# ─── Tests cho AlertEngine ────────────────────────────────────────────────────

class TestAlertEngine:
    def _make_engine(self, tmp_path):
        from utils.alert_engine import AlertEngine
        cfg = {
            "min_confidence": 0.6,
            "cooldown_seconds": 5,
        }
        engine = AlertEngine(cfg)
        engine.output_dir = tmp_path
        return engine

    def _make_frame_data(self):
        from utils.video_preprocessor import FrameData
        return FrameData(
            frame=make_frame(),
            index=42,
            timestamp=5.5,
            motion_score=0.1,
            camera_id="cam_test",
        )

    def _make_reasoning(self, is_anomaly=True, confidence=0.85, anomaly_class="fighting"):
        from agents.deep_reasoning_agent import ReasoningResult
        return ReasoningResult(
            is_anomaly=is_anomaly,
            anomaly_class=anomaly_class,
            confidence=confidence,
            explanation="Phát hiện đánh nhau",
            bounding_regions=[],
            raw_vlm_caption="hai người đánh nhau",
            query_rounds=1,
        )

    def test_emits_alert_when_conditions_met(self, tmp_path):
        engine = self._make_engine(tmp_path)
        fd = self._make_frame_data()
        reasoning = self._make_reasoning(is_anomaly=True, confidence=0.85)
        alert = engine.evaluate(fd, reasoning)
        assert alert is not None
        assert alert.anomaly_class == "fighting"
        assert alert.confidence == pytest.approx(0.85)

    def test_no_alert_when_not_anomaly(self, tmp_path):
        engine = self._make_engine(tmp_path)
        fd = self._make_frame_data()
        reasoning = self._make_reasoning(is_anomaly=False, confidence=0.9)
        alert = engine.evaluate(fd, reasoning)
        assert alert is None

    def test_no_alert_below_min_confidence(self, tmp_path):
        engine = self._make_engine(tmp_path)
        fd = self._make_frame_data()
        reasoning = self._make_reasoning(is_anomaly=True, confidence=0.4)
        alert = engine.evaluate(fd, reasoning)
        assert alert is None

    def test_cooldown_suppresses_duplicate_alert(self, tmp_path):
        engine = self._make_engine(tmp_path)
        fd = self._make_frame_data()
        reasoning = self._make_reasoning(is_anomaly=True, confidence=0.85)
        alert1 = engine.evaluate(fd, reasoning)
        assert alert1 is not None
        # Ngay lập tức gọi lại → bị cooldown
        alert2 = engine.evaluate(fd, reasoning)
        assert alert2 is None

    def test_alert_to_dict_has_required_keys(self, tmp_path):
        engine = self._make_engine(tmp_path)
        fd = self._make_frame_data()
        reasoning = self._make_reasoning()
        alert = engine.evaluate(fd, reasoning)
        d = alert.to_dict()
        for key in ["camera_id", "frame_index", "timestamp", "anomaly_class", "confidence", "explanation"]:
            assert key in d

    def test_alert_str_representation(self, tmp_path):
        engine = self._make_engine(tmp_path)
        fd = self._make_frame_data()
        reasoning = self._make_reasoning()
        alert = engine.evaluate(fd, reasoning)
        s = str(alert)
        assert "ALERT" in s
        assert "FIGHTING" in s


# ─── Tests cho RAGMemory (persistence) ────────────────────────────────────────

class TestRAGMemoryPersistence:
    def test_save_and_load_roundtrip(self, tmp_path):
        from utils.rag_memory import RAGMemory
        cfg = {"enabled": True, "top_k": 3, "per_camera_memory": True, "memory_dir": str(tmp_path)}

        mem = RAGMemory(cfg)
        mem.add_normal_scene("Người đi bộ bình thường", camera_id="cam_01", timestamp=1.0)
        mem.add_normal_scene("Xe máy dừng đèn đỏ", camera_id="cam_01", timestamp=3.0)
        mem.save("cam_01")

        mem2 = RAGMemory(cfg)
        assert mem2.size("cam_01") == 0  # chưa load
        mem2.load("cam_01")
        assert mem2.size("cam_01") == 2

    def test_per_camera_isolation(self, tmp_path):
        from utils.rag_memory import RAGMemory
        cfg = {"enabled": True, "top_k": 3, "per_camera_memory": True, "memory_dir": str(tmp_path)}
        mem = RAGMemory(cfg)
        mem.add_normal_scene("Cảnh cam 01", camera_id="cam_01")
        mem.add_normal_scene("Cảnh cam 02", camera_id="cam_02")
        assert mem.size("cam_01") == 1
        assert mem.size("cam_02") == 1
        # cam_01 không truy xuất được entry của cam_02
        results = mem.retrieve("cảnh", camera_id="cam_01")
        assert all("cam 01" in r for r in results)

    def test_disabled_rag_returns_empty(self, tmp_path):
        from utils.rag_memory import RAGMemory
        cfg = {"enabled": False, "top_k": 3, "per_camera_memory": True, "memory_dir": str(tmp_path)}
        mem = RAGMemory(cfg)
        mem.add_normal_scene("test", camera_id="cam_01")
        assert mem.size("cam_01") == 0
        assert mem.retrieve("test", camera_id="cam_01") == []


# ─── Tests cho AnomalyScorer ──────────────────────────────────────────────────

class TestAnomalyScorer:
    def test_returns_base_threshold_before_warmup(self):
        from utils.anomaly_scorer import AnomalyScorer
        scorer = AnomalyScorer({
            "threshold": 0.45,
            "temporal_window": 16,
            "aggregation": "weighted_mean",
            "adaptive": {"warmup_frames": 300, "std_multiplier": 2.5},
        })
        assert scorer.current_threshold == pytest.approx(0.45)

    def test_score_returns_tuple(self):
        from utils.anomaly_scorer import AnomalyScorer
        scorer = AnomalyScorer({"threshold": 0.45, "temporal_window": 4, "aggregation": "mean"})
        agg, suspicious = scorer.score(0.1)
        assert isinstance(agg, float)
        assert isinstance(suspicious, bool)

    def test_high_score_triggers_suspicious(self):
        from utils.anomaly_scorer import AnomalyScorer
        scorer = AnomalyScorer({"threshold": 0.3, "temporal_window": 1, "aggregation": "mean",
                                 "adaptive": {"warmup_frames": 1000, "std_multiplier": 2.5}})
        _, suspicious = scorer.score(0.9)
        assert suspicious is True

    def test_low_score_not_suspicious(self):
        from utils.anomaly_scorer import AnomalyScorer
        scorer = AnomalyScorer({"threshold": 0.45, "temporal_window": 1, "aggregation": "mean",
                                 "adaptive": {"warmup_frames": 1000, "std_multiplier": 2.5}})
        _, suspicious = scorer.score(0.05)
        assert suspicious is False
