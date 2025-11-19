from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Sequence

from bot.macro import MacroSentimentEngine
from bot.playbook import (
    DEFAULT_HORIZONS,
    HorizonConfig,
    PlaybookAssetDefinition,
    PlaybookAssetFile,
    build_portfolio_playbook,
    load_playbook_asset_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the Multi-Market Portfolio Playbook for arbitrary assets.",
    )
    parser.add_argument(
        "--assets",
        type=str,
        help="Path to a YAML/JSON file listing custom playbook assets.",
    )
    parser.add_argument(
        "--starting-balance",
        type=float,
        default=10_000.0,
        help="Starting balance used when simulating each horizon (default: 10,000).",
    )
    parser.add_argument(
        "--macro-events",
        type=str,
        help="Optional path to a macro events JSON file overriding the default list.",
    )
    parser.add_argument(
        "--macro-refresh",
        type=int,
        default=300,
        help="Macro sentiment refresh cadence in seconds (default: 300).",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional path to write the generated playbook JSON payload.",
    )
    return parser.parse_args()


def _load_asset_config(path_value: Optional[str]) -> tuple[
    Optional[Sequence[PlaybookAssetDefinition]],
    Sequence[HorizonConfig],
]:
    if not path_value:
        return None, DEFAULT_HORIZONS

    config_path = Path(path_value).expanduser()
    asset_file: PlaybookAssetFile = load_playbook_asset_file(config_path)
    return asset_file.assets, asset_file.horizons


def main() -> None:
    args = parse_args()

    asset_definitions, horizons = _load_asset_config(args.assets)

    macro_events_path = Path(args.macro_events).expanduser() if args.macro_events else None
    macro_engine = MacroSentimentEngine(
        events_path=macro_events_path,
        refresh_interval=args.macro_refresh,
    )

    build_kwargs: Dict[str, object] = {
        "starting_balance": args.starting_balance,
        "macro_engine": macro_engine,
        "horizons": horizons,
    }
    if asset_definitions is not None:
        build_kwargs["asset_definitions"] = asset_definitions

    playbook = build_portfolio_playbook(**build_kwargs)
    payload = playbook.to_dict()

    output_text = json.dumps(payload, indent=2)
    print(output_text)

    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text, encoding="utf-8")
        print(f"\n💾 Saved playbook to {output_path}")


if __name__ == "__main__":
    main()
