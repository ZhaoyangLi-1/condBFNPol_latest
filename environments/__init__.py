"""Environment wrappers for policy evaluation."""

from environments.base import BaseEnv

__all__ = ["BaseEnv"]

# Optional PushT (requires gymnasium and gym-pusht)
try:
    from environments.pusht import PushTEnv
    __all__.append("PushTEnv")
except ImportError:
    PushTEnv = None
