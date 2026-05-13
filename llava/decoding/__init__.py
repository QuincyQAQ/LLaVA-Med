"""Decoding-time strategies (VGS, future VCD, …) kept in `strategies/<name>/` for isolated experiments."""

import llava.decoding.strategies  # noqa: F401 — registers vgs, …

from llava.decoding.registry import DECODING_STRATEGIES, register_decoding, get_decoding_fn

__all__ = ["DECODING_STRATEGIES", "register_decoding", "get_decoding_fn"]
