"""
Monitoring Module.

Provides production monitoring capabilities:
- Real-time P&L tracking
- Performance metrics
- System health monitoring
- Risk metrics
- Alerting system
"""

from typing import Any

from .production_monitor import (
    Alert,
    AlertLevel,
    MetricStatus,
    ProductionMonitor,
    RiskMetrics,
    SystemHealth,
    TradeRecord,
    get_production_monitor,
)

_LEGACY_EXPORTS: dict[str, Any] = {}
psutil: Any
try:
    from importlib.util import module_from_spec, spec_from_file_location
    from pathlib import Path
    import sys

    legacy_path = Path(__file__).resolve().parent.parent / "monitoring.py"
    if legacy_path.exists():
        spec = spec_from_file_location("bot._legacy_monitoring", legacy_path)
        if spec and spec.loader:
            legacy_module = module_from_spec(spec)
            sys.modules[spec.name] = legacy_module
            spec.loader.exec_module(legacy_module)
            for name in (
                "AlertLevel",
                "AlertChannel",
                "Alert",
                "MetricSnapshot",
                "TradingMetrics",
                "AlertManager",
                "SystemMonitor",
                "TradingMonitor",
                "MonitoringService",
                "psutil",
            ):
                if hasattr(legacy_module, name):
                    _LEGACY_EXPORTS[name] = getattr(legacy_module, name)

            if "psutil" in _LEGACY_EXPORTS:
                class _PsutilProxy:
                    def __getattr__(self, attr: str) -> Any:
                        target = globals().get("psutil")
                        return getattr(target, attr)

                globals()["psutil"] = _LEGACY_EXPORTS["psutil"]
                legacy_module.psutil = _PsutilProxy()  # type: ignore[attr-defined]
except Exception:
    _LEGACY_EXPORTS = {}

globals().update(_LEGACY_EXPORTS)

__all__ = [
    "Alert",
    "AlertLevel",
    "MetricStatus",
    "ProductionMonitor",
    "RiskMetrics",
    "SystemHealth",
    "TradeRecord",
    "get_production_monitor",
    *_LEGACY_EXPORTS.keys(),
]
