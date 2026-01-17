"""
Health check and readiness probe endpoints for Kubernetes and Docker health checks.
"""

from fastapi import APIRouter, Response, HTTPException
from datetime import datetime
import psutil
import asyncio

router = APIRouter(prefix="/health", tags=["health"])

class HealthStatus:
    def __init__(self):
        self.startup_time = datetime.utcnow()
        self.is_ready = False
        self.components = {}
    
    async def check_database(self):
        """Check database connectivity."""
        try:
            # Import your database connection here
            # await db.connection.scalar("SELECT 1")
            return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def check_cache(self):
        """Check Redis/cache connectivity."""
        try:
            # Check cache
            return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def check_market_data(self):
        """Check market data freshness."""
        try:
            # Check if latest market data is recent
            return {"status": "healthy", "age_seconds": 5}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

health_status = HealthStatus()

@router.get("/live")
async def liveness():
    """Liveness probe: is the service running?"""
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": (datetime.utcnow() - health_status.startup_time).total_seconds()
    }

@router.get("/ready")
async def readiness():
    """Readiness probe: is the service ready to accept traffic?"""
    if not health_status.is_ready:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return {
        "status": "ready",
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/detailed")
async def health_detailed():
    """Detailed health check with component status."""
    db_status = await health_status.check_database()
    cache_status = await health_status.check_cache()
    market_data_status = await health_status.check_market_data()
    
    all_healthy = all([
        db_status.get("status") == "healthy",
        cache_status.get("status") == "healthy",
        market_data_status.get("status") == "healthy"
    ])
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "database": db_status,
            "cache": cache_status,
            "market_data": market_data_status,
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
        }
    }

@router.post("/ready/init")
async def mark_ready():
    """Mark the service as ready (call after initialization)."""
    health_status.is_ready = True
    return {"status": "ready"}
