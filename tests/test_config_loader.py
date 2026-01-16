"""Tests for bot.config_loader module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from bot.config_loader import load_overrides, merge_config
from bot.strategy import StrategyConfig


class TestLoadOverrides:
    """Tests for load_overrides function."""

    def test_load_overrides_from_valid_json(self, tmp_path: Path) -> None:
        """Test loading overrides from a valid JSON file."""
        override_file = tmp_path / "overrides.json"
        overrides = {"ema_fast": 10, "ema_slow": 40, "rsi_period": 12}
        override_file.write_text(json.dumps(overrides))

        result = load_overrides(override_file)

        assert result == overrides

    def test_load_overrides_nonexistent_file(self, tmp_path: Path) -> None:
        """Test loading overrides from a non-existent file returns empty dict."""
        result = load_overrides(tmp_path / "nonexistent.json")

        assert result == {}

    def test_load_overrides_invalid_json(self, tmp_path: Path) -> None:
        """Test loading overrides from invalid JSON returns empty dict."""
        override_file = tmp_path / "invalid.json"
        override_file.write_text("not valid json {")

        result = load_overrides(override_file)

        assert result == {}

    def test_load_overrides_non_dict_json(self, tmp_path: Path) -> None:
        """Test loading overrides when JSON is not a dict returns empty dict."""
        override_file = tmp_path / "array.json"
        override_file.write_text('["item1", "item2"]')

        result = load_overrides(override_file)

        assert result == {}

    def test_load_overrides_empty_json(self, tmp_path: Path) -> None:
        """Test loading overrides from empty JSON object."""
        override_file = tmp_path / "empty.json"
        override_file.write_text("{}")

        result = load_overrides(override_file)

        assert result == {}

    def test_load_overrides_nested_dict(self, tmp_path: Path) -> None:
        """Test loading overrides with nested structure."""
        override_file = tmp_path / "nested.json"
        overrides = {"key": {"nested": "value"}}
        override_file.write_text(json.dumps(overrides))

        result = load_overrides(override_file)

        assert result == overrides


class TestMergeConfig:
    """Tests for merge_config function."""

    def test_merge_config_with_valid_overrides(self) -> None:
        """Test merging config with valid overrides."""
        base = StrategyConfig()
        overrides = {"ema_fast": 15, "ema_slow": 45}

        result = merge_config(base, overrides)

        assert result.ema_fast == 15
        assert result.ema_slow == 45

    def test_merge_config_with_empty_overrides(self) -> None:
        """Test merging config with empty overrides returns base values."""
        base = StrategyConfig()
        original_ema_fast = base.ema_fast

        result = merge_config(base, {})

        assert result.ema_fast == original_ema_fast

    def test_merge_config_ignores_unknown_keys(self) -> None:
        """Test that unknown keys are ignored during merge."""
        base = StrategyConfig()

        result = merge_config(base, {"unknown_key": "value"})

        assert not hasattr(result, "unknown_key")

    def test_merge_config_ignores_none_values(self) -> None:
        """Test that None values are ignored during merge."""
        base = StrategyConfig()
        original_ema_fast = base.ema_fast

        result = merge_config(base, {"ema_fast": None})

        assert result.ema_fast == original_ema_fast

    def test_merge_config_preserves_unspecified_values(self) -> None:
        """Test that unspecified values are preserved from base."""
        base = StrategyConfig()
        original_rsi_period = base.rsi_period

        result = merge_config(base, {"ema_fast": 20})

        assert result.rsi_period == original_rsi_period
        assert result.ema_fast == 20

    def test_merge_config_multiple_overrides(self) -> None:
        """Test merging multiple overrides at once."""
        base = StrategyConfig()
        overrides = {
            "ema_fast": 8,
            "ema_slow": 21,
            "rsi_period": 14,
            "rsi_overbought": 75,
            "rsi_oversold": 25,
        }

        result = merge_config(base, overrides)

        assert result.ema_fast == 8
        assert result.ema_slow == 21
        assert result.rsi_period == 14
        assert result.rsi_overbought == 75
        assert result.rsi_oversold == 25
