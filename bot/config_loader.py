from __future__ import annotations

import json
from dataclasses import asdic
from pathlib import Path
from typing import Any, Dict, cas

from .strategy import StrategyConfig


def load_overrides(file_path: Path) -> Dict[str, Any]:
    """Load strategy overrides from a JSON file if it exists.

    Expected keys match StrategyConfig fields, e.g.:
      {"ema_fast": 10, "ema_slow": 40, "rsi_period": 12, ...}
    """
    try:
        if file_path.is_file():
            with file_path.open("r", encoding="utf-8") as f:
                raw: Any = json.load(f)
            if isinstance(raw, dict):
                # best-effort cast to a typed mapping
                return cast(Dict[str, Any], raw)
    except Exception:
        # ignore errors and fall back to defaults
        pass
    return {}


def merge_config(base: StrategyConfig, overrides: Dict[str, Any]) -> StrategyConfig:
    payload = asdict(base)
    for k, v in overrides.items():
        if k in payload and v is not None:
            payload[k] = v
    return StrategyConfig(**payload)
