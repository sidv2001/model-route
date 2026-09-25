"""A synthetic, offline experiment in contextual model routing."""

from .core import ModelProfile, Outcome, Task
from .router import ContextualRouter

__all__ = ["ContextualRouter", "ModelProfile", "Outcome", "Task"]
