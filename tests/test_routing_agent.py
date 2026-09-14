"""
tests/test_routing_agent.py
────────────────────────────
Unit tests cho RoutingAgent: compute_score, tier boundaries, night_multiplier.
"""

import pytest
from agents.routing_agent import RoutingAgent, RoutingDecision


@pytest.fixture
def agent():
    return RoutingAgent()


@pytest.fixture
def agent_custom():
    return RoutingAgent(config={
        "weights": {"motion": 1.0, "person": 0.0, "complexity": 0.0, "risk": 0.0},
        "tier_thresholds": [0.35, 0.70],
        "night_multiplier": 1.3,
    })


class TestComputeScore:
    def test_all_zero_inputs(self, agent):
        score = agent.compute_score(0.0, 0.0, 0.0, 0.0)
        assert score == 0.0

    def test_all_max_inputs(self, agent):
        score = agent.compute_score(1.0, 1.0, 1.0, 1.0)
        assert score == 1.0

    def test_weights_sum(self, agent):
        # Tất cả signals = 1.0 → score = tổng weights = 1.0
        total_weight = sum(agent.weights.values())
        assert abs(total_weight - 1.0) < 1e-6

    def test_night_multiplier_applied(self, agent):
        score_day = agent.compute_score(0.5, 0.0, 0.0, 0.0, is_night=False)
        score_night = agent.compute_score(0.5, 0.0, 0.0, 0.0, is_night=True)
        assert score_night > score_day
        assert abs(score_night / score_day - agent.night_multiplier) < 1e-6

    def test_night_multiplier_clamped_to_one(self, agent):
        # Nếu tất cả signals cao + night → vẫn không vượt 1.0
        score = agent.compute_score(1.0, 1.0, 1.0, 1.0, is_night=True)
        assert score <= 1.0

    def test_motion_only_variant(self, agent_custom):
        # Chỉ motion signal, weights khác = 0
        score = agent_custom.compute_score(0.5, 1.0, 1.0, 1.0)
        assert abs(score - 0.5) < 1e-6


class TestTierBoundaries:
    def test_low_score_tier1(self, agent):
        score = agent.compute_score(0.1, 0.1, 0.1, 0.1)  # ~0.1
        tier = agent._score_to_tier(score)
        assert tier == 1

    def test_boundary_at_035_tier2(self, agent):
        assert agent._score_to_tier(0.35) == 2

    def test_just_below_035_tier1(self, agent):
        assert agent._score_to_tier(0.3499) == 1

    def test_boundary_at_070_tier3(self, agent):
        assert agent._score_to_tier(0.70) == 3

    def test_just_below_070_tier2(self, agent):
        assert agent._score_to_tier(0.6999) == 2

    def test_high_score_tier3(self, agent):
        assert agent._score_to_tier(1.0) == 3


class TestRouteMethod:
    def test_returns_routing_decision(self, agent):
        features = {
            "motion_mean": 0.5,
            "person_count": 2,
            "scene_complexity": 2.0,
            "camera_id": "default",
            "timestamp": 0.0,
        }
        decision = agent.route(features)
        assert isinstance(decision, RoutingDecision)
        assert decision.tier in (1, 2, 3)
        assert 0.0 <= decision.routing_score <= 1.0

    def test_missing_features_defaults(self, agent):
        decision = agent.route({})
        assert decision.tier == 1
        # default camera risk_numeric=0.3, weight=0.20 → score=0.06
        assert abs(decision.routing_score - 0.06) < 1e-6

    def test_high_motion_high_person_escalates(self, agent):
        features = {
            "motion_mean": 10.0,   # >> norm range → normalized = 1.0
            "person_count": 50,    # >> norm range → normalized = 1.0
            "scene_complexity": 5.0,
            "camera_id": "default",
            "timestamp": 0.0,
        }
        decision = agent.route(features)
        assert decision.tier == 3

    def test_empty_scene_stays_tier1(self, agent):
        features = {
            "motion_mean": 0.0,
            "person_count": 0,
            "scene_complexity": 0.5,
            "camera_id": "default",
            "timestamp": 14400.0,  # 04:00 local — night
        }
        decision = agent.route(features)
        # Even with night multiplier, zeros stay tier 1
        assert decision.tier == 1


class TestNormalization:
    def test_normalize_clamps_at_one(self, agent):
        result = agent._normalize(1000.0, "motion_mean")
        assert result == 1.0

    def test_normalize_zero(self, agent):
        result = agent._normalize(0.0, "motion_mean")
        assert result == 0.0

    def test_normalize_mid_range(self, agent):
        # motion_mean upper = 5.0 → 2.5 → 0.5
        result = agent._normalize(2.5, "motion_mean")
        assert abs(result - 0.5) < 1e-6


class TestCameraRisk:
    def test_default_camera_risk(self, agent):
        risk = agent._risk_numeric("unknown_cam")
        assert risk == 0.3  # default low

    def test_custom_camera_config(self):
        camera_config = {
            "cameras": {
                "cam_airport": {"risk_numeric": 0.9},
                "default": {"risk_numeric": 0.3},
            }
        }
        agent = RoutingAgent(camera_config=camera_config)
        assert agent._risk_numeric("cam_airport") == 0.9
        assert agent._risk_numeric("unknown") == 0.3
