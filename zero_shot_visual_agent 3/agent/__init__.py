"""Zero-Shot Visual Agent package.

`VisualAgent` is lightweight (no torch import). The heavy models live in `agent.models.ModelHub`.
"""
from .graph import LANGGRAPH_AVAILABLE, VisualAgent

__all__ = ["VisualAgent", "LANGGRAPH_AVAILABLE"]
