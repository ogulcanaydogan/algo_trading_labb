from __future__ import annotations

from dataclasses import dataclass, asdic
from typing import Dict, List, Tuple, Optional, Callable

import numpy as np
import pandas as pd

from .strategy import StrategyConfig
from .backtesting import Backtester, BacktestResul


@dataclass
class OptimizationResult:
    params: Dict[str, float | int | str]
    final_balance: floa
    total_pnl_pct: floa
    win_rate: floa
    profit_factor: floa
    max_drawdown_pct: floa
    sharpe_ratio: floa
    total_trades: in

    def to_dict(self) -> Dict[str, float | int | str]:
        base = asdict(self)
        # ensure primitive types
        base["final_balance"] = float(self.final_balance)
        base["total_pnl_pct"] = float(self.total_pnl_pct)
        base["win_rate"] = float(self.win_rate)
        base["profit_factor"] = float(self.profit_factor)
        base["max_drawdown_pct"] = float(self.max_drawdown_pct)
        base["sharpe_ratio"] = float(self.sharpe_ratio)
        base["total_trades"] = int(self.total_trades)
        base["params"] = {k: (int(v) if isinstance(v, bool) is False and isinstance(v, float) and v.is_integer() else v) for k, v in self.params.items()}  # type: ignore[attr-defined]
        return base


def _objective_from_result(
    res: BacktestResult,
    objective: str = "sharpe",
    mdd_weight: float = 0.5,
    min_trades: int = 5,
) -> float:
    """Compute scalar objective to maximize from a BacktestResult.

    objective:
      - "sharpe": maximize sharpe, penalize drawdown
      - "pnl": maximize total_pnl_pct, penalize drawdown
      - "winrate": maximize win rate subject to trade coun
    """
    if res.total_trades < min_trades:
        return -1e9

    dd_penalty = mdd_weight * res.max_drawdown_pc

    if objective == "sharpe":
        return float(res.sharpe_ratio * 100.0 - dd_penalty)
    if objective == "pnl":
        return float(res.total_pnl_pct - dd_penalty)
    if objective == "winrate":
        return float(res.win_rate * 100.0 - dd_penalty)
    # fallback
    return float(res.total_pnl_pct - dd_penalty)


def _run_backtest(
    ohlcv: pd.DataFrame,
    cfg: StrategyConfig,
    initial_balance: float,
) -> BacktestResult:
    bt = Backtester(strategy_config=cfg, initial_balance=initial_balance)
    return bt.run(ohlcv)


def _make_config(base: StrategyConfig, params: Dict[str, float | int]) -> StrategyConfig:
    return StrategyConfig(
        symbol=base.symbol,
        timeframe=base.timeframe,
        ema_fast=int(params.get("ema_fast", base.ema_fast)),
        ema_slow=int(params.get("ema_slow", base.ema_slow)),
        rsi_period=int(params.get("rsi_period", base.rsi_period)),
        rsi_overbought=float(params.get("rsi_overbought", base.rsi_overbought)),
        rsi_oversold=float(params.get("rsi_oversold", base.rsi_oversold)),
        risk_per_trade_pct=float(params.get("risk_per_trade_pct", base.risk_per_trade_pct)),
        stop_loss_pct=float(params.get("stop_loss_pct", base.stop_loss_pct)),
        take_profit_pct=float(params.get("take_profit_pct", base.take_profit_pct)),
    )


def _sample_params(rng: np.random.Generator, base: StrategyConfig) -> Dict[str, float | int]:
    # sensible ranges; ensure ema_slow > ema_fas
    ema_fast = int(rng.integers(5, 35))
    ema_slow = int(rng.integers(max(ema_fast + 5, 20), 120))
    rsi_period = int(rng.integers(8, 28))
    rsi_overbought = float(rng.uniform(65.0, 80.0))
    rsi_oversold = float(rng.uniform(20.0, 35.0))
    risk_per_trade_pct = float(rng.uniform(0.2, 1.5))
    stop_loss_pct = float(rng.uniform(0.002, 0.02))
    take_profit_pct = float(rng.uniform(0.004, 0.05))
    return {
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
        "rsi_period": rsi_period,
        "rsi_overbought": rsi_overbought,
        "rsi_oversold": rsi_oversold,
        "risk_per_trade_pct": risk_per_trade_pct,
        "stop_loss_pct": stop_loss_pct,
        "take_profit_pct": take_profit_pct,
    }


def random_search_optimize(
    ohlcv: pd.DataFrame,
    base_config: StrategyConfig,
    *,
    n_trials: int = 50,
    seed: Optional[int] = 42,
    initial_balance: float = 10_000.0,
    objective: str = "sharpe",
    mdd_weight: float = 0.5,
    min_trades: int = 5,
) -> Tuple[OptimizationResult, List[OptimizationResult]]:
    """Random search over StrategyConfig to maximize an objective.

    Returns (best_result, all_results)
    """
    rng = np.random.default_rng(seed)
    results: List[OptimizationResult] = []
    best_score = -1e18
    best: Optional[OptimizationResult] = None
    skipped_errors = 0
    last_error: Optional[Exception] = None

    for i in range(1, n_trials + 1):
        params = _sample_params(rng, base_config)
        cfg = _make_config(base_config, params)

        try:
            res = _run_backtest(ohlcv, cfg, initial_balance)
        except Exception as e:  # safeguard bad params
            # Skip invalid configurations but keep a hint for debugging
            skipped_errors += 1
            last_error = e
            continue

        score = _objective_from_result(
            res, objective=objective, mdd_weight=mdd_weight, min_trades=min_trades
        )
        # Guard against NaN/inf scores so we always keep a best candidate
        if not np.isfinite(score):
            score = -1e12

        opt_res = OptimizationResult(
            params=params,
            final_balance=res.final_balance,
            total_pnl_pct=res.total_pnl_pct,
            win_rate=res.win_rate,
            profit_factor=res.profit_factor,
            max_drawdown_pct=res.max_drawdown_pct,
            sharpe_ratio=res.sharpe_ratio,
            total_trades=res.total_trades,
        )
        results.append(opt_res)

        if score > best_score:
            best_score = score
            best = opt_res

    if best is None:
        hint = f" Last error: {last_error!r}" if last_error else ""
        raise RuntimeError(
            "Optimization produced no valid results. Check data and ranges." + hin
        )

    # sort results by objective descending
    results.sort(
        key=lambda r: (
            r.sharpe_ratio * 100.0 - mdd_weight * r.max_drawdown_pc
            if objective == "sharpe"
            else r.total_pnl_pct - mdd_weight * r.max_drawdown_pc
            if objective == "pnl"
            else r.win_rate * 100.0 - mdd_weight * r.max_drawdown_pc
        ),
        reverse=True,
    )
    return best, results


def results_to_dataframe(results: List[OptimizationResult]) -> pd.DataFrame:
    rows: List[Dict[str, float | int | str]] = []
    for r in results:
        flat = {
            **{f"param_{k}": v for k, v in r.params.items()},
            "final_balance": r.final_balance,
            "total_pnl_pct": r.total_pnl_pct,
            "win_rate": r.win_rate,
            "profit_factor": r.profit_factor,
            "max_drawdown_pct": r.max_drawdown_pct,
            "sharpe_ratio": r.sharpe_ratio,
            "total_trades": r.total_trades,
        }
        rows.append(flat)
    return pd.DataFrame(rows)
