"""Rollout side modules."""

from typing import Any

from nanoverl.rollout.load_balancer import LoadBalancer
from nanoverl.rollout.rollout_manager import RolloutManager

__all__ = [
    "LoadBalancer",
    "RolloutManager",
    "VLLMServer",
]


def __getattr__(name: str) -> Any:
    if name == "VLLMServer":
        from nanoverl.rollout.vllm_server import VLLMServer

        return VLLMServer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
