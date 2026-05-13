"""Register decoding strategies by name for `model_vqa.py --decoding-strategy`."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

DECODING_STRATEGIES: Dict[str, Callable[..., Any]] = {}


def register_decoding(name: str, fn: Callable[..., Any]) -> None:
    DECODING_STRATEGIES[name] = fn


def get_decoding_fn(name: str) -> Optional[Callable[..., Any]]:
    return DECODING_STRATEGIES.get(name)


def list_decoding_strategies() -> str:
    return ", ".join(sorted(DECODING_STRATEGIES.keys())) or "(none)"
