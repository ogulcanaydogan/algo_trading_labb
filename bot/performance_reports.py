"""
Performance Reports Module.

Generates daily and weekly performance reports with email notifications.
"""

import json
import logging
import os
import smtplib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DailyReport:
    """Daily performance report."""
    date: str
    starting_balance: float
    ending_balance: float
    daily_pnl: float
    daily_pnl_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    best_trade: Dict[str, Any]
    worst_trade: Dict[str, Any]
    positions_opened: int
    positions_closed: int
    open_positions: List[Dict[str, Any]]
    max_drawdown_pct: float
    sharpe_ratio: float
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date,
            "starting_balance": self.starting_balance,
            "ending_balance": self.ending_balance,
            "daily_pnl": self.daily_pnl,
            "daily_pnl_pct": self.daily_pnl_pct,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "best_trade": self.best_trade,
            "worst_trade": self.worst_trade,
            "positions_opened": self.positions_opened,
            "positions_closed": self.positions_closed,
            "open_positions": self.open_positions,
            "max_drawdown_pct": self.max_drawdown_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "notes": self.notes,
        }

    def to_html(self) -> str:
        """Generate HTML report."""
        pnl_color = "#00cc00" if self.daily_pnl >= 0 else "#cc0000"

        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #1a1a2e; color: white; padding: 20px; border-radius: 10px; }}
                .stat-box {{ background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 8px; }}
                .positive {{ color: #00cc00; }}
                .negative {{ color: #cc0000; }}
                table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background: #333; color: white; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Daily Trading Report - {self.date}</h1>
            </div>

            <div class="stat-box">
                <h2>Performance Summary</h2>
                <table>
                    <tr>
                        <td><strong>Starting Balance</strong></td>
                        <td>${self.starting_balance:,.2f}</td>
                        <td><strong>Ending Balance</strong></td>
                        <td>${self.ending_balance:,.2f}</td>
                    </tr>
                    <tr>
                        <td><strong>Daily P&L</strong></td>
                        <td style="color: {pnl_color};">${self.daily_pnl:+,.2f} ({self.daily_pnl_pct:+.2f}%)</td>
                        <td><strong>Win Rate</strong></td>
                        <td>{self.win_rate:.1f}%</td>
                    </tr>
                    <tr>
                        <td><strong>Total Trades</strong></td>
                        <td>{self.total_trades}</td>
                        <td><strong>Wins/Losses</strong></td>
                        <td>{self.winning_trades}/{self.losing_trades}</td>
                    </tr>
                    <tr>
                        <td><strong>Max Drawdown</strong></td>
                        <td class="negative">{self.max_drawdown_pct:.2f}%</td>
                        <td><strong>Sharpe Ratio</strong></td>
                        <td>{self.sharpe_ratio:.2f}</td>
                    </tr>
                </table>
            </div>

            <div class="stat-box">
                <h2>Trade Highlights</h2>
                <table>
                    <tr>
                        <th>Type</th>
                        <th>Symbol</th>
                        <th>P&L</th>
                        <th>P&L %</th>
                    </tr>
                    <tr>
                        <td>Best Trade</td>
                        <td>{self.best_trade.get('symbol', 'N/A')}</td>
                        <td class="positive">${self.best_trade.get('pnl', 0):+,.2f}</td>
                        <td class="positive">{self.best_trade.get('pnl_pct', 0):+.2f}%</td>
                    </tr>
                    <tr>
                        <td>Worst Trade</td>
                        <td>{self.worst_trade.get('symbol', 'N/A')}</td>
                        <td class="negative">${self.worst_trade.get('pnl', 0):+,.2f}</td>
                        <td class="negative">{self.worst_trade.get('pnl_pct', 0):+.2f}%</td>
                    </tr>
                </table>
            </div>

            <div class="stat-box">
                <h2>Open Positions ({len(self.open_positions)})</h2>
                <table>
                    <tr>
                        <th>Symbol</th>
                        <th>Side</th>
                        <th>Entry Price</th>
                        <th>Unrealized P&L</th>
                    </tr>
        """

        for pos in self.open_positions:
            upnl = pos.get('unrealized_pnl', 0)
            upnl_color = "#00cc00" if upnl >= 0 else "#cc0000"
            html += f"""
                    <tr>
                        <td>{pos.get('symbol', 'N/A')}</td>
                        <td>{pos.get('side', 'N/A').upper()}</td>
                        <td>${pos.get('entry_price', 0):,.2f}</td>
                        <td style="color: {upnl_color};">${upnl:+,.2f}</td>
                    </tr>
            """

        if not self.open_positions:
            html += "<tr><td colspan='4'>No open positions</td></tr>"

        html += """
                </table>
            </div>
        </body>
        </html>
        """

        return html


@dataclass
class WeeklyReport:
    """Weekly performance report."""
    week_start: str
    week_end: str
    starting_balance: float
    ending_balance: float
    weekly_pnl: float
    weekly_pnl_pct: float
    total_trades: int
    winning_trades: int
    win_rate: float
    best_day: Dict[str, Any]
    worst_day: Dict[str, Any]
    daily_breakdown: List[Dict[str, Any]]
    avg_daily_pnl: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    kelly_fraction: float
    top_performers: List[Dict[str, Any]]
    worst_performers: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "week_start": self.week_start,
            "week_end": self.week_end,
            "starting_balance": self.starting_balance,
            "ending_balance": self.ending_balance,
            "weekly_pnl": self.weekly_pnl,
            "weekly_pnl_pct": self.weekly_pnl_pct,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "win_rate": self.win_rate,
            "best_day": self.best_day,
            "worst_day": self.worst_day,
            "daily_breakdown": self.daily_breakdown,
            "avg_daily_pnl": self.avg_daily_pnl,
            "max_drawdown_pct": self.max_drawdown_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "kelly_fraction": self.kelly_fraction,
            "top_performers": self.top_performers,
            "worst_performers": self.worst_performers,
        }


class PerformanceReporter:
    """
    Generates and sends performance reports.
    """

    def __init__(
        self,
        data_dir: str = "data/unified_trading",
        reports_dir: str = "data/reports",
        email_config: Optional[Dict[str, str]] = None,
    ):
        self.data_dir = Path(data_dir)
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Email configuration
        self.email_config = email_config or {
            "smtp_server": os.getenv("SMTP_SERVER", "smtp.gmail.com"),
            "smtp_port": int(os.getenv("SMTP_PORT", "587")),
            "sender_email": os.getenv("REPORT_EMAIL_SENDER"),
            "sender_password": os.getenv("REPORT_EMAIL_PASSWORD"),
            "recipient_email": os.getenv("REPORT_EMAIL_RECIPIENT"),
        }

    def generate_daily_report(
        self,
        date: Optional[str] = None,
    ) -> DailyReport:
        """Generate daily performance report."""
        date = date or datetime.now().strftime("%Y-%m-%d")

        # Load trade history
        trades_file = self.data_dir / "trades.json"
        trades = []
        if trades_file.exists():
            with open(trades_file) as f:
                all_trades = json.load(f)
                trades = [t for t in all_trades if t.get("timestamp", "").startswith(date)]

        # Load state
        state_file = self.data_dir / "state.json"
        state = {}
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)

        # Load equity history
        equity_file = self.data_dir / "equity.json"
        equity_history = []
        if equity_file.exists():
            with open(equity_file) as f:
                equity_history = json.load(f)

        # Calculate metrics
        today_equity = [e for e in equity_history if e.get("timestamp", "").startswith(date)]

        starting_balance = today_equity[0].get("equity", today_equity[0].get("balance", 10000)) if today_equity else state.get("initial_capital", 10000)
        ending_balance = today_equity[-1].get("equity", today_equity[-1].get("balance", 10000)) if today_equity else state.get("balance", 10000)

        daily_pnl = ending_balance - starting_balance
        daily_pnl_pct = (daily_pnl / starting_balance * 100) if starting_balance > 0 else 0

        # Trade stats
        winning_trades = [t for t in trades if t.get("pnl", 0) > 0]
        losing_trades = [t for t in trades if t.get("pnl", 0) < 0]

        win_rate = len(winning_trades) / len(trades) * 100 if trades else 0

        best_trade = max(trades, key=lambda t: t.get("pnl", 0)) if trades else {}
        worst_trade = min(trades, key=lambda t: t.get("pnl", 0)) if trades else {}

        # Open positions
        positions = state.get("positions", {})
        open_positions = [
            {
                "symbol": sym,
                "side": pos.get("side"),
                "entry_price": pos.get("entry_price"),
                "quantity": pos.get("quantity"),
                "unrealized_pnl": pos.get("unrealized_pnl", 0),
            }
            for sym, pos in positions.items()
        ]

        # Calculate max drawdown from equity curve
        max_dd = 0
        peak = starting_balance
        for e in today_equity:
            equity = e.get("equity", e.get("balance", starting_balance))
            peak = max(peak, equity)
            dd = (peak - equity) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        # Helper to get equity/balance
        def get_eq(e):
            return e.get("equity", e.get("balance", 10000))

        # Estimate daily Sharpe
        if len(today_equity) > 1:
            returns = []
            for i in range(1, len(today_equity)):
                prev_eq = get_eq(today_equity[i-1])
                curr_eq = get_eq(today_equity[i])
                if prev_eq > 0:
                    ret = (curr_eq - prev_eq) / prev_eq
                    returns.append(ret)

            import numpy as np
            sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if returns and np.std(returns) > 0 else 0
        else:
            sharpe = 0

        report = DailyReport(
            date=date,
            starting_balance=starting_balance,
            ending_balance=ending_balance,
            daily_pnl=daily_pnl,
            daily_pnl_pct=daily_pnl_pct,
            total_trades=len(trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            best_trade=best_trade,
            worst_trade=worst_trade,
            positions_opened=sum(1 for t in trades if t.get("action") == "open"),
            positions_closed=sum(1 for t in trades if t.get("action") == "close"),
            open_positions=open_positions,
            max_drawdown_pct=max_dd * 100,
            sharpe_ratio=sharpe,
        )

        # Save report
        report_path = self.reports_dir / f"daily_{date}.json"
        with open(report_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        # Save HTML
        html_path = self.reports_dir / f"daily_{date}.html"
        with open(html_path, "w") as f:
            f.write(report.to_html())

        logger.info(f"Generated daily report: {report_path}")

        return report

    def generate_weekly_report(
        self,
        week_end: Optional[str] = None,
    ) -> WeeklyReport:
        """Generate weekly performance report."""
        if week_end is None:
            week_end_dt = datetime.now()
        else:
            week_end_dt = datetime.strptime(week_end, "%Y-%m-%d")

        week_start_dt = week_end_dt - timedelta(days=6)
        week_start = week_start_dt.strftime("%Y-%m-%d")
        week_end = week_end_dt.strftime("%Y-%m-%d")

        # Load all trades for the week
        trades_file = self.data_dir / "trades.json"
        trades = []
        if trades_file.exists():
            with open(trades_file) as f:
                all_trades = json.load(f)
                trades = [
                    t for t in all_trades
                    if week_start <= t.get("timestamp", "")[:10] <= week_end
                ]

        # Load state
        state_file = self.data_dir / "state.json"
        state = {}
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)

        # Load equity history
        equity_file = self.data_dir / "equity.json"
        equity_history = []
        if equity_file.exists():
            with open(equity_file) as f:
                equity_history = json.load(f)

        # Filter equity for the week
        week_equity = [
            e for e in equity_history
            if week_start <= e.get("timestamp", "")[:10] <= week_end
        ]

        # Helper to get equity/balance
        def get_eq(e):
            return e.get("equity", e.get("balance", 10000))

        starting_balance = get_eq(week_equity[0]) if week_equity else state.get("initial_capital", 10000)
        ending_balance = get_eq(week_equity[-1]) if week_equity else state.get("balance", 10000)

        weekly_pnl = ending_balance - starting_balance
        weekly_pnl_pct = (weekly_pnl / starting_balance * 100) if starting_balance > 0 else 0

        # Daily breakdown
        daily_breakdown = []
        for i in range(7):
            day_dt = week_start_dt + timedelta(days=i)
            day_str = day_dt.strftime("%Y-%m-%d")
            day_trades = [t for t in trades if t.get("timestamp", "").startswith(day_str)]
            day_pnl = sum(t.get("pnl", 0) for t in day_trades)

            daily_breakdown.append({
                "date": day_str,
                "day": day_dt.strftime("%A"),
                "trades": len(day_trades),
                "pnl": day_pnl,
                "wins": sum(1 for t in day_trades if t.get("pnl", 0) > 0),
            })

        # Best/worst day
        best_day = max(daily_breakdown, key=lambda d: d["pnl"]) if daily_breakdown else {}
        worst_day = min(daily_breakdown, key=lambda d: d["pnl"]) if daily_breakdown else {}

        # Winning trades
        winning_trades = [t for t in trades if t.get("pnl", 0) > 0]
        win_rate = len(winning_trades) / len(trades) * 100 if trades else 0

        # Performance by symbol
        symbol_performance = {}
        for t in trades:
            sym = t.get("symbol", "unknown")
            if sym not in symbol_performance:
                symbol_performance[sym] = {"pnl": 0, "trades": 0}
            symbol_performance[sym]["pnl"] += t.get("pnl", 0)
            symbol_performance[sym]["trades"] += 1

        top_performers = sorted(
            [{"symbol": k, **v} for k, v in symbol_performance.items()],
            key=lambda x: x["pnl"],
            reverse=True
        )[:3]

        worst_performers = sorted(
            [{"symbol": k, **v} for k, v in symbol_performance.items()],
            key=lambda x: x["pnl"]
        )[:3]

        # Calculate ratios
        import numpy as np

        daily_pnls = [d["pnl"] for d in daily_breakdown]
        avg_daily_pnl = np.mean(daily_pnls) if daily_pnls else 0

        # Max drawdown
        max_dd = 0
        peak = starting_balance
        for e in week_equity:
            equity = e.get("equity", e.get("balance", starting_balance))
            peak = max(peak, equity)
            dd = (peak - equity) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        # Risk-adjusted metrics
        if len(daily_pnls) > 1 and np.std(daily_pnls) > 0:
            sharpe = (np.mean(daily_pnls) / np.std(daily_pnls)) * np.sqrt(252)

            negative_returns = [r for r in daily_pnls if r < 0]
            downside_std = np.std(negative_returns) if negative_returns else np.std(daily_pnls)
            sortino = (np.mean(daily_pnls) / downside_std) * np.sqrt(252) if downside_std > 0 else 0

            calmar = (weekly_pnl_pct / 100) / max_dd if max_dd > 0 else 0
        else:
            sharpe = sortino = calmar = 0

        # Kelly criterion
        if trades:
            wins = [t["pnl"] for t in trades if t.get("pnl", 0) > 0]
            losses = [abs(t["pnl"]) for t in trades if t.get("pnl", 0) < 0]

            if wins and losses:
                avg_win = np.mean(wins)
                avg_loss = np.mean(losses)
                p = len(wins) / len(trades)
                q = 1 - p
                b = avg_win / avg_loss if avg_loss > 0 else 1
                kelly = (b * p - q) / b
                kelly = max(0, min(1, kelly))
            else:
                kelly = 0
        else:
            kelly = 0

        report = WeeklyReport(
            week_start=week_start,
            week_end=week_end,
            starting_balance=starting_balance,
            ending_balance=ending_balance,
            weekly_pnl=weekly_pnl,
            weekly_pnl_pct=weekly_pnl_pct,
            total_trades=len(trades),
            winning_trades=len(winning_trades),
            win_rate=win_rate,
            best_day=best_day,
            worst_day=worst_day,
            daily_breakdown=daily_breakdown,
            avg_daily_pnl=avg_daily_pnl,
            max_drawdown_pct=max_dd * 100,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            kelly_fraction=kelly,
            top_performers=top_performers,
            worst_performers=worst_performers,
        )

        # Save report
        report_path = self.reports_dir / f"weekly_{week_start}_{week_end}.json"
        with open(report_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        logger.info(f"Generated weekly report: {report_path}")

        return report

    def send_email_report(
        self,
        report: DailyReport,
        subject: Optional[str] = None,
    ) -> bool:
        """Send report via email."""
        if not all([
            self.email_config.get("sender_email"),
            self.email_config.get("sender_password"),
            self.email_config.get("recipient_email"),
        ]):
            logger.warning("Email not configured - skipping email send")
            return False

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject or f"Trading Report - {report.date}"
            msg["From"] = self.email_config["sender_email"]
            msg["To"] = self.email_config["recipient_email"]

            # Plain text version
            text = f"""
Daily Trading Report - {report.date}

P&L: ${report.daily_pnl:+,.2f} ({report.daily_pnl_pct:+.2f}%)
Trades: {report.total_trades}
Win Rate: {report.win_rate:.1f}%
            """

            # HTML version
            html = report.to_html()

            msg.attach(MIMEText(text, "plain"))
            msg.attach(MIMEText(html, "html"))

            # Send
            with smtplib.SMTP(
                self.email_config["smtp_server"],
                self.email_config["smtp_port"],
            ) as server:
                server.starttls()
                server.login(
                    self.email_config["sender_email"],
                    self.email_config["sender_password"],
                )
                server.send_message(msg)

            logger.info(f"Sent email report to {self.email_config['recipient_email']}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False


def create_performance_reporter() -> PerformanceReporter:
    """Factory function to create performance reporter."""
    return PerformanceReporter()
