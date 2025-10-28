from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query

from bot.state import StateStore, create_state_store

from .schemas import BotStateResponse, EquityPointResponse, SignalResponse

STATE_DIR = Path(os.getenv("DATA_DIR", "./data"))

app = FastAPI(
    title="Algo Trading Lab Status API",
    version="0.1.0",
    description="Status endpoints for the Algo Trading Lab bot.",
)

_state_store: Optional[StateStore] = None


def get_store() -> StateStore:
    global _state_store
    if _state_store is None:
        _state_store = create_state_store(STATE_DIR)
    return _state_store


@app.get("/status", response_model=BotStateResponse)
def read_status(store: StateStore = Depends(get_store)) -> BotStateResponse:
    store.load()
    payload = store.get_state_dict()
    if not payload:
        raise HTTPException(status_code=404, detail="State not found.")
    return BotStateResponse(**payload)


@app.get("/signals", response_model=List[SignalResponse])
def read_signals(
    limit: int = Query(default=50, ge=1, le=500),
    store: StateStore = Depends(get_store),
) -> List[SignalResponse]:
    store.load()
    signals = store.get_signals(limit)
    return [SignalResponse(**item) for item in signals]


@app.get("/equity", response_model=List[EquityPointResponse])
def read_equity(store: StateStore = Depends(get_store)) -> List[EquityPointResponse]:
    store.load()
    curve = store.get_equity_curve()
    return [EquityPointResponse(**point) for point in curve]

