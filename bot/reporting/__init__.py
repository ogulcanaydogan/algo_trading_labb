"""
Reporting module for tax and compliance reporting.

Provides:
- Tax lot tracking (FIFO, LIFO, HIFO, Average)
- Capital gains calculation
- Tax report generation (Form 8949 format)
"""

from .tax_reporting import (
    TaxLotTracker,
    TaxReportGenerator,
    TaxLot,
    TaxSummary,
    CapitalGain,
    CostBasisMethod,
    GainType,
    DisposalEvent,
    create_tax_lot_tracker,
    create_tax_report_generator,
)

__all__ = [
    "TaxLotTracker",
    "TaxReportGenerator",
    "TaxLot",
    "TaxSummary",
    "CapitalGain",
    "CostBasisMethod",
    "GainType",
    "DisposalEvent",
    "create_tax_lot_tracker",
    "create_tax_report_generator",
]
