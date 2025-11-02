from __future__ import annotations

import argparse
import logging
from pathlib import Path

from bot.portfolio import PortfolioConfig, PortfolioRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the multi-asset portfolio trading bot.",
    )
    parser.add_argument(
        "-c",
        "--config",
        default="data/portfolio.json",
        help="Path to the portfolio configuration JSON file (default: data/portfolio.json).",
    )
    parser.add_argument(
        "--capital",
        type=float,
        help="Override total portfolio capital (USD).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and exit without starting live threads.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    config_path = Path(args.config).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Portfolio configuration not found: {config_path}")

    config = PortfolioConfig.load(config_path)
    if args.capital is not None:
        config.portfolio_capital = args.capital

    if args.dry_run:
        for asset in config.assets:
            allocation = "n/a"
            if asset.allocation_pct is not None:
                allocation = f"{asset.allocation_pct:.2f}%"
            print(
                f"[DRY-RUN] symbol={asset.symbol} asset_type={asset.asset_type} "
                f"timeframe={asset.timeframe or config.default_timeframe} allocation={allocation}"
            )
        return

    runner = PortfolioRunner(config)
    runner.start()


if __name__ == "__main__":
    main()

