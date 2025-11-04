from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pathlib import Path

app = FastAPI(title="Algo Trading Lab - API")


@app.get("/status")
def status() -> dict:
    """Lightweight status endpoint used by health checks and smoke runs."""
    return {"ok": True}


# Load a simple dashboard HTML template shipped alongside the API module.
DASHBOARD_TEMPLATE = Path(__file__).parent / "dashboard.html"


def load_dashboard_template() -> str:
    try:
        return DASHBOARD_TEMPLATE.read_text(encoding="utf-8")
    except Exception:
        return "<!DOCTYPE html><html><body><h1>Dashboard template missing.</h1></body></html>"


@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
@app.get("/dashboard/preview", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    """Serve a lightweight HTML dashboard for quick monitoring."""
    return HTMLResponse(content=load_dashboard_template())
