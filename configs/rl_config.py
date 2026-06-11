"""Backward-compatible RL config import.

This module is kept as a compatibility shim for older scripts that import
`configs.rl_config`. New code should import `configs.sarl_config`.
"""

from .sarl_config import RLConfig

__all__ = ["RLConfig"]
