# agents/__init__.py
# Lazy imports để tránh load heavy models khi chưa cần
from agents.routing_agent import RoutingAgent, RoutingDecision

__all__ = ["VADOrchestrator", "RoutingAgent", "RoutingDecision", "DeepReasoningAgent", "ReasoningResult"]


def __getattr__(name):
    if name == "VADOrchestrator":
        from agents.orchestrator import VADOrchestrator
        return VADOrchestrator
    if name in ("DeepReasoningAgent", "ReasoningResult"):
        from agents.deep_reasoning_agent import DeepReasoningAgent, ReasoningResult
        return {"DeepReasoningAgent": DeepReasoningAgent, "ReasoningResult": ReasoningResult}[name]
    raise AttributeError(f"module 'agents' has no attribute {name!r}")
