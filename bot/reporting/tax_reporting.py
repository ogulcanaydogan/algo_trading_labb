"""
Tax Reporting Module for Trading Activity.

Calculates capital gains/losses using various accounting methods
and generates tax reports for compliance.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import uuid

logger = logging.getLogger(__name__)


class CostBasisMethod(Enum):
    """Cost basis calculation methods."""

    FIFO = "fifo"  # First In, First Out
    LIFO = "lifo"  # Last In, First Out
    HIFO = "hifo"  # Highest In, First Out (tax optimal for gains)
    AVERAGE = "average"  # Average cost
    SPECIFIC = "specific"  # Specific identification


class GainType(Enum):
    """Type of capital gain/loss."""

    SHORT_TERM = "short_term"  # Held <= 1 year
    LONG_TERM = "long_term"  # Held > 1 year


@dataclass
class TaxLot:
    """A tax lot representing acquired assets."""

    id: str
    symbol: str
    quantity: Decimal
    cost_basis: Decimal  # Per unit
    total_cost: Decimal
    acquisition_date: datetime
    acquisition_type: str  # "buy", "transfer_in", etc.
    remaining_quantity: Decimal = field(default=Decimal("0"))

    def __post_init__(self):
        if self.remaining_quantity == Decimal("0"):
            self.remaining_quantity = self.quantity

    @property
    def cost_per_unit(self) -> Decimal:
        return self.cost_basis

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "quantity": str(self.quantity),
            "cost_basis": str(self.cost_basis),
            "total_cost": str(self.total_cost),
            "acquisition_date": self.acquisition_date.isoformat(),
            "acquisition_type": self.acquisition_type,
            "remaining_quantity": str(self.remaining_quantity),
        }


@dataclass
class DisposalEvent:
    """An asset disposal (sale) event."""

    id: str
    symbol: str
    quantity: Decimal
    proceeds_per_unit: Decimal
    total_proceeds: Decimal
    disposal_date: datetime
    disposal_type: str  # "sell", "transfer_out", etc.
    fees: Decimal = Decimal("0")

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "quantity": str(self.quantity),
            "proceeds_per_unit": str(self.proceeds_per_unit),
            "total_proceeds": str(self.total_proceeds),
            "disposal_date": self.disposal_date.isoformat(),
            "disposal_type": self.disposal_type,
            "fees": str(self.fees),
        }


@dataclass
class CapitalGain:
    """A realized capital gain/loss."""

    id: str
    symbol: str
    quantity: Decimal
    cost_basis: Decimal
    proceeds: Decimal
    gain_loss: Decimal
    gain_type: GainType
    acquisition_date: datetime
    disposal_date: datetime
    holding_period_days: int
    tax_lot_id: str
    disposal_id: str

    @property
    def is_gain(self) -> bool:
        return self.gain_loss > 0

    @property
    def is_loss(self) -> bool:
        return self.gain_loss < 0

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "quantity": str(self.quantity),
            "cost_basis": str(self.cost_basis),
            "proceeds": str(self.proceeds),
            "gain_loss": str(self.gain_loss),
            "gain_type": self.gain_type.value,
            "acquisition_date": self.acquisition_date.isoformat(),
            "disposal_date": self.disposal_date.isoformat(),
            "holding_period_days": self.holding_period_days,
            "tax_lot_id": self.tax_lot_id,
            "disposal_id": self.disposal_id,
        }


@dataclass
class TaxSummary:
    """Tax summary for a period."""

    start_date: datetime
    end_date: datetime
    total_proceeds: Decimal
    total_cost_basis: Decimal
    total_gains: Decimal
    total_losses: Decimal
    net_gain_loss: Decimal
    short_term_gains: Decimal
    short_term_losses: Decimal
    long_term_gains: Decimal
    long_term_losses: Decimal
    transaction_count: int
    symbols_traded: List[str]

    def to_dict(self) -> Dict:
        return {
            "period": {
                "start": self.start_date.isoformat(),
                "end": self.end_date.isoformat(),
            },
            "total_proceeds": str(self.total_proceeds),
            "total_cost_basis": str(self.total_cost_basis),
            "total_gains": str(self.total_gains),
            "total_losses": str(self.total_losses),
            "net_gain_loss": str(self.net_gain_loss),
            "short_term": {
                "gains": str(self.short_term_gains),
                "losses": str(self.short_term_losses),
                "net": str(self.short_term_gains + self.short_term_losses),
            },
            "long_term": {
                "gains": str(self.long_term_gains),
                "losses": str(self.long_term_losses),
                "net": str(self.long_term_gains + self.long_term_losses),
            },
            "transaction_count": self.transaction_count,
            "symbols_traded": self.symbols_traded,
        }


class TaxLotTracker:
    """
    Track tax lots for cost basis calculation.

    Supports multiple cost basis methods:
    - FIFO (First In, First Out)
    - LIFO (Last In, First Out)
    - HIFO (Highest In, First Out)
    - Average Cost
    """

    def __init__(self, method: CostBasisMethod = CostBasisMethod.FIFO):
        self.method = method
        self._lots: Dict[str, List[TaxLot]] = {}  # symbol -> lots
        self._disposals: List[DisposalEvent] = []
        self._gains: List[CapitalGain] = []

    def add_lot(
        self,
        symbol: str,
        quantity: float,
        cost_per_unit: float,
        acquisition_date: datetime,
        acquisition_type: str = "buy",
    ) -> TaxLot:
        """Add a tax lot for acquired assets."""
        qty = Decimal(str(quantity))
        cost = Decimal(str(cost_per_unit))

        lot = TaxLot(
            id=str(uuid.uuid4())[:8],
            symbol=symbol,
            quantity=qty,
            cost_basis=cost,
            total_cost=qty * cost,
            acquisition_date=acquisition_date,
            acquisition_type=acquisition_type,
        )

        if symbol not in self._lots:
            self._lots[symbol] = []
        self._lots[symbol].append(lot)

        logger.debug(f"Added tax lot: {lot.id} - {symbol} {quantity} @ {cost_per_unit}")
        return lot

    def dispose(
        self,
        symbol: str,
        quantity: float,
        proceeds_per_unit: float,
        disposal_date: datetime,
        disposal_type: str = "sell",
        fees: float = 0.0,
    ) -> List[CapitalGain]:
        """
        Dispose of assets and calculate gains/losses.

        Args:
            symbol: Asset symbol
            quantity: Quantity to dispose
            proceeds_per_unit: Sale price per unit
            disposal_date: Date of disposal
            disposal_type: Type of disposal
            fees: Transaction fees

        Returns:
            List of capital gains/losses
        """
        qty = Decimal(str(quantity))
        proceeds = Decimal(str(proceeds_per_unit))
        fee = Decimal(str(fees))

        disposal = DisposalEvent(
            id=str(uuid.uuid4())[:8],
            symbol=symbol,
            quantity=qty,
            proceeds_per_unit=proceeds,
            total_proceeds=qty * proceeds - fee,
            disposal_date=disposal_date,
            disposal_type=disposal_type,
            fees=fee,
        )
        self._disposals.append(disposal)

        # Get lots to match
        lots = self._get_lots_for_disposal(symbol, qty)
        gains = []

        remaining_qty = qty
        for lot, lot_qty in lots:
            if remaining_qty <= 0:
                break

            # Calculate gain/loss for this lot portion
            use_qty = min(lot_qty, remaining_qty)
            cost = use_qty * lot.cost_basis
            sale_proceeds = use_qty * proceeds - (fee * use_qty / qty)

            holding_days = (disposal_date - lot.acquisition_date).days
            gain_type = GainType.LONG_TERM if holding_days > 365 else GainType.SHORT_TERM

            gain = CapitalGain(
                id=str(uuid.uuid4())[:8],
                symbol=symbol,
                quantity=use_qty,
                cost_basis=cost,
                proceeds=sale_proceeds,
                gain_loss=sale_proceeds - cost,
                gain_type=gain_type,
                acquisition_date=lot.acquisition_date,
                disposal_date=disposal_date,
                holding_period_days=holding_days,
                tax_lot_id=lot.id,
                disposal_id=disposal.id,
            )
            gains.append(gain)
            self._gains.append(gain)

            # Update lot
            lot.remaining_quantity -= use_qty
            remaining_qty -= use_qty

        logger.info(
            f"Disposed {quantity} {symbol}: "
            f"gains={sum(g.gain_loss for g in gains if g.is_gain)}, "
            f"losses={sum(g.gain_loss for g in gains if g.is_loss)}"
        )

        return gains

    def _get_lots_for_disposal(
        self, symbol: str, quantity: Decimal
    ) -> List[Tuple[TaxLot, Decimal]]:
        """Get lots to use for disposal based on cost basis method."""
        if symbol not in self._lots:
            raise ValueError(f"No lots found for {symbol}")

        available_lots = [lot for lot in self._lots[symbol] if lot.remaining_quantity > 0]

        if not available_lots:
            raise ValueError(f"Insufficient lots for {symbol}")

        # Sort based on method
        if self.method == CostBasisMethod.FIFO:
            available_lots.sort(key=lambda l: l.acquisition_date)
        elif self.method == CostBasisMethod.LIFO:
            available_lots.sort(key=lambda l: l.acquisition_date, reverse=True)
        elif self.method == CostBasisMethod.HIFO:
            available_lots.sort(key=lambda l: l.cost_basis, reverse=True)
        elif self.method == CostBasisMethod.AVERAGE:
            # For average, we'll handle differently
            pass

        result = []
        remaining = quantity

        for lot in available_lots:
            if remaining <= 0:
                break

            use_qty = min(lot.remaining_quantity, remaining)
            result.append((lot, use_qty))
            remaining -= use_qty

        if remaining > 0:
            raise ValueError(
                f"Insufficient quantity for {symbol}: need {quantity}, have {quantity - remaining}"
            )

        return result

    def get_average_cost(self, symbol: str) -> Decimal:
        """Calculate average cost basis for a symbol."""
        if symbol not in self._lots:
            return Decimal("0")

        total_cost = Decimal("0")
        total_qty = Decimal("0")

        for lot in self._lots[symbol]:
            if lot.remaining_quantity > 0:
                total_cost += lot.remaining_quantity * lot.cost_basis
                total_qty += lot.remaining_quantity

        if total_qty == 0:
            return Decimal("0")

        return (total_cost / total_qty).quantize(Decimal("0.01"), ROUND_HALF_UP)

    def get_unrealized_gains(self, symbol: str, current_price: float) -> Dict[str, Decimal]:
        """Calculate unrealized gains for a symbol."""
        if symbol not in self._lots:
            return {"unrealized_gain": Decimal("0"), "quantity": Decimal("0")}

        price = Decimal(str(current_price))
        total_cost = Decimal("0")
        total_qty = Decimal("0")

        for lot in self._lots[symbol]:
            if lot.remaining_quantity > 0:
                total_cost += lot.remaining_quantity * lot.cost_basis
                total_qty += lot.remaining_quantity

        market_value = total_qty * price
        unrealized = market_value - total_cost

        return {
            "unrealized_gain": unrealized,
            "quantity": total_qty,
            "cost_basis": total_cost,
            "market_value": market_value,
        }

    def get_lots(self, symbol: Optional[str] = None) -> List[TaxLot]:
        """Get tax lots, optionally filtered by symbol."""
        if symbol:
            return self._lots.get(symbol, [])
        return [lot for lots in self._lots.values() for lot in lots]

    def get_gains(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        symbol: Optional[str] = None,
    ) -> List[CapitalGain]:
        """Get capital gains, optionally filtered."""
        gains = self._gains

        if symbol:
            gains = [g for g in gains if g.symbol == symbol]

        if start_date:
            gains = [g for g in gains if g.disposal_date >= start_date]

        if end_date:
            gains = [g for g in gains if g.disposal_date <= end_date]

        return gains


class TaxReportGenerator:
    """Generate tax reports from trading activity."""

    def __init__(self, tracker: TaxLotTracker):
        self.tracker = tracker

    def generate_summary(self, year: int, month: Optional[int] = None) -> TaxSummary:
        """Generate tax summary for a period."""
        if month:
            start_date = datetime(year, month, 1)
            if month == 12:
                end_date = datetime(year + 1, 1, 1) - timedelta(seconds=1)
            else:
                end_date = datetime(year, month + 1, 1) - timedelta(seconds=1)
        else:
            start_date = datetime(year, 1, 1)
            end_date = datetime(year, 12, 31, 23, 59, 59)

        gains = self.tracker.get_gains(start_date, end_date)

        total_proceeds = sum(g.proceeds for g in gains)
        total_cost = sum(g.cost_basis for g in gains)
        total_gains = sum(g.gain_loss for g in gains if g.is_gain)
        total_losses = sum(g.gain_loss for g in gains if g.is_loss)

        short_term = [g for g in gains if g.gain_type == GainType.SHORT_TERM]
        long_term = [g for g in gains if g.gain_type == GainType.LONG_TERM]

        short_gains = sum(g.gain_loss for g in short_term if g.is_gain)
        short_losses = sum(g.gain_loss for g in short_term if g.is_loss)
        long_gains = sum(g.gain_loss for g in long_term if g.is_gain)
        long_losses = sum(g.gain_loss for g in long_term if g.is_loss)

        symbols = list(set(g.symbol for g in gains))

        return TaxSummary(
            start_date=start_date,
            end_date=end_date,
            total_proceeds=total_proceeds,
            total_cost_basis=total_cost,
            total_gains=total_gains,
            total_losses=total_losses,
            net_gain_loss=total_gains + total_losses,
            short_term_gains=short_gains,
            short_term_losses=short_losses,
            long_term_gains=long_gains,
            long_term_losses=long_losses,
            transaction_count=len(gains),
            symbols_traded=symbols,
        )

    def generate_form_8949(self, year: int) -> List[Dict]:
        """
        Generate data for IRS Form 8949.

        Each row represents a sale/disposition:
        - Description of property
        - Date acquired
        - Date sold
        - Proceeds
        - Cost basis
        - Gain or loss
        """
        start = datetime(year, 1, 1)
        end = datetime(year, 12, 31, 23, 59, 59)

        gains = self.tracker.get_gains(start, end)

        rows = []
        for gain in gains:
            rows.append(
                {
                    "description": f"{gain.quantity} {gain.symbol}",
                    "date_acquired": gain.acquisition_date.strftime("%m/%d/%Y"),
                    "date_sold": gain.disposal_date.strftime("%m/%d/%Y"),
                    "proceeds": float(gain.proceeds.quantize(Decimal("0.01"))),
                    "cost_basis": float(gain.cost_basis.quantize(Decimal("0.01"))),
                    "adjustment_code": "",
                    "adjustment_amount": 0,
                    "gain_loss": float(gain.gain_loss.quantize(Decimal("0.01"))),
                    "term": "S" if gain.gain_type == GainType.SHORT_TERM else "L",
                }
            )

        return rows

    def export_csv(self, year: int, include_lots: bool = False) -> str:
        """Export tax data to CSV format."""
        output = io.StringIO()

        # Gains/Losses
        gains = self.tracker.get_gains(datetime(year, 1, 1), datetime(year, 12, 31, 23, 59, 59))

        writer = csv.writer(output)
        writer.writerow(
            [
                "Symbol",
                "Quantity",
                "Acquisition Date",
                "Disposal Date",
                "Cost Basis",
                "Proceeds",
                "Gain/Loss",
                "Type",
                "Holding Days",
            ]
        )

        for gain in gains:
            writer.writerow(
                [
                    gain.symbol,
                    str(gain.quantity),
                    gain.acquisition_date.strftime("%Y-%m-%d"),
                    gain.disposal_date.strftime("%Y-%m-%d"),
                    str(gain.cost_basis.quantize(Decimal("0.01"))),
                    str(gain.proceeds.quantize(Decimal("0.01"))),
                    str(gain.gain_loss.quantize(Decimal("0.01"))),
                    gain.gain_type.value,
                    gain.holding_period_days,
                ]
            )

        if include_lots:
            writer.writerow([])
            writer.writerow(["=== Open Tax Lots ==="])
            writer.writerow(
                ["Symbol", "Quantity", "Remaining", "Cost Basis", "Total Cost", "Acquisition Date"]
            )

            for lot in self.tracker.get_lots():
                if lot.remaining_quantity > 0:
                    writer.writerow(
                        [
                            lot.symbol,
                            str(lot.quantity),
                            str(lot.remaining_quantity),
                            str(lot.cost_basis),
                            str(lot.total_cost),
                            lot.acquisition_date.strftime("%Y-%m-%d"),
                        ]
                    )

        return output.getvalue()

    def export_json(self, year: int) -> str:
        """Export tax data to JSON format."""
        summary = self.generate_summary(year)
        form_8949 = self.generate_form_8949(year)

        data = {
            "year": year,
            "summary": summary.to_dict(),
            "transactions": form_8949,
            "generated_at": datetime.now().isoformat(),
        }

        return json.dumps(data, indent=2)


def create_tax_lot_tracker(method: CostBasisMethod = CostBasisMethod.FIFO) -> TaxLotTracker:
    """Factory function to create tax lot tracker."""
    return TaxLotTracker(method=method)


def create_tax_report_generator(tracker: TaxLotTracker) -> TaxReportGenerator:
    """Factory function to create tax report generator."""
    return TaxReportGenerator(tracker)
