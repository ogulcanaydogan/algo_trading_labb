"""
API Versioning Module.

Provides API versioning support with:
- URL path versioning (/api/v1/, /api/v2/)
- Header-based version selection (X-API-Version)
- Version deprecation warnings
- Backward compatibility layers

Usage:
    from api.versioning import VersionedAPIRouter, APIVersion

    v1_router = VersionedAPIRouter(version=APIVersion.V1, prefix="/api/v1")
    v2_router = VersionedAPIRouter(version=APIVersion.V2, prefix="/api/v2")

    @v1_router.get("/users")
    async def get_users_v1():
        return {"users": [...]}  # V1 format

    @v2_router.get("/users")
    async def get_users_v2():
        return {"data": {"users": [...]}, "meta": {...}}  # V2 format
"""

import functools
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar

from fastapi import APIRouter, Header, HTTPException, Request, Response
from fastapi.routing import APIRoute
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class APIVersion(str, Enum):
    """API version identifiers."""

    V1 = "v1"
    V2 = "v2"
    V3 = "v3"

    @classmethod
    def latest(cls) -> "APIVersion":
        """Return the latest stable API version."""
        return cls.V2

    @classmethod
    def all_versions(cls) -> List[str]:
        """Return all version strings."""
        return [v.value for v in cls]


# Version metadata
VERSION_INFO: Dict[str, Dict[str, Any]] = {
    "v1": {
        "released": "2024-01-01",
        "status": "deprecated",
        "sunset_date": "2025-06-01",
        "description": "Legacy API version",
        "breaking_changes": [],
    },
    "v2": {
        "released": "2024-06-01",
        "status": "stable",
        "sunset_date": None,
        "description": "Current stable API version with improved response format",
        "breaking_changes": [
            "Response format changed to {data: ..., meta: ...}",
            "Error format changed to RFC 7807",
            "Timestamps now use ISO 8601 with timezone",
        ],
    },
    "v3": {
        "released": "2025-01-01",
        "status": "beta",
        "sunset_date": None,
        "description": "Beta API version with new features",
        "breaking_changes": [
            "Pagination format changed to cursor-based",
            "Rate limiting headers updated",
        ],
    },
}


class VersionedRoute(APIRoute):
    """
    Custom route class that adds version headers to responses.
    """

    def __init__(self, *args, version: str = "v2", **kwargs):
        super().__init__(*args, **kwargs)
        self.api_version = version

    def get_route_handler(self) -> Callable:
        original_handler = super().get_route_handler()

        async def versioned_handler(request: Request) -> Response:
            response = await original_handler(request)

            # Add version headers
            response.headers["X-API-Version"] = self.api_version
            response.headers["X-API-Version-Status"] = VERSION_INFO.get(
                self.api_version, {}
            ).get("status", "unknown")

            # Add deprecation warning if applicable
            version_info = VERSION_INFO.get(self.api_version, {})
            if version_info.get("status") == "deprecated":
                sunset_date = version_info.get("sunset_date", "unknown")
                response.headers["Deprecation"] = "true"
                response.headers["Sunset"] = sunset_date
                response.headers["Link"] = f'</api/v2{request.url.path.replace(f"/api/{self.api_version}", "")}>; rel="successor-version"'

            return response

        return versioned_handler


class VersionedAPIRouter(APIRouter):
    """
    API router with built-in versioning support.
    """

    def __init__(
        self,
        version: APIVersion = APIVersion.V2,
        prefix: str = "",
        **kwargs,
    ):
        """
        Initialize a versioned API router.

        Args:
            version: API version for this router
            prefix: URL prefix (typically /api/v1, /api/v2, etc.)
            **kwargs: Additional arguments for APIRouter
        """
        # Set prefix if not provided
        if not prefix:
            prefix = f"/api/{version.value}"

        super().__init__(
            prefix=prefix,
            route_class=lambda *a, **kw: VersionedRoute(*a, version=version.value, **kw),
            **kwargs,
        )
        self.api_version = version

    def deprecated(
        self,
        successor_endpoint: Optional[str] = None,
        sunset_date: Optional[str] = None,
    ):
        """
        Mark an endpoint as deprecated.

        Args:
            successor_endpoint: URL of the new endpoint to use
            sunset_date: Date when this endpoint will be removed
        """

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                response = await func(*args, **kwargs)
                # Deprecation headers are handled by VersionedRoute
                return response

            # Add deprecation metadata
            wrapper._deprecated = True
            wrapper._successor = successor_endpoint
            wrapper._sunset = sunset_date

            return wrapper

        return decorator


class APIVersionMiddleware(BaseHTTPMiddleware):
    """
    Middleware for handling API version selection via headers.
    """

    def __init__(self, app, default_version: str = "v2"):
        super().__init__(app)
        self.default_version = default_version

    async def dispatch(self, request: Request, call_next):
        # Check for version header
        requested_version = request.headers.get("X-API-Version", self.default_version)

        # Validate version
        if requested_version not in APIVersion.all_versions():
            requested_version = self.default_version

        # Store version in request state
        request.state.api_version = requested_version

        # Call the route
        response = await call_next(request)

        # Add version info to response
        response.headers["X-API-Version-Requested"] = requested_version
        response.headers["X-API-Version-Latest"] = APIVersion.latest().value

        return response


def version_response(
    data: Any,
    version: str = "v2",
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Format response according to API version.

    Args:
        data: Response data
        version: API version
        meta: Optional metadata

    Returns:
        Formatted response dict
    """
    if version == "v1":
        # V1: Simple response format
        if isinstance(data, dict):
            return data
        return {"result": data}

    # V2+: Standardized response format
    response = {
        "data": data,
        "meta": {
            "version": version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **(meta or {}),
        },
    }

    return response


def version_error(
    status_code: int,
    detail: str,
    version: str = "v2",
    error_code: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Format error response according to API version.

    Args:
        status_code: HTTP status code
        detail: Error message
        version: API version
        error_code: Machine-readable error code
        extra: Additional error details

    Returns:
        Formatted error dict
    """
    if version == "v1":
        # V1: Simple error format
        return {"error": detail, "status": status_code}

    # V2+: RFC 7807 Problem Details
    return {
        "type": f"https://api.example.com/errors/{error_code or 'unknown'}",
        "title": error_code or "Error",
        "status": status_code,
        "detail": detail,
        "instance": None,  # Set by caller
        **(extra or {}),
    }


T = TypeVar("T")


def version_adapter(
    v1_transform: Optional[Callable[[T], Any]] = None,
    v2_transform: Optional[Callable[[T], Any]] = None,
    v3_transform: Optional[Callable[[T], Any]] = None,
):
    """
    Decorator to transform responses based on API version.

    Args:
        v1_transform: Transform function for v1
        v2_transform: Transform function for v2
        v3_transform: Transform function for v3
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            # Get response from original function
            result = await func(request, *args, **kwargs)

            # Get version from request
            version = getattr(request.state, "api_version", "v2")

            # Apply transform based on version
            if version == "v1" and v1_transform:
                return v1_transform(result)
            elif version == "v2" and v2_transform:
                return v2_transform(result)
            elif version == "v3" and v3_transform:
                return v3_transform(result)

            return result

        return wrapper

    return decorator


# Convenience functions for creating versioned routers
def create_v1_router(**kwargs) -> VersionedAPIRouter:
    """Create a V1 API router."""
    return VersionedAPIRouter(version=APIVersion.V1, **kwargs)


def create_v2_router(**kwargs) -> VersionedAPIRouter:
    """Create a V2 API router."""
    return VersionedAPIRouter(version=APIVersion.V2, **kwargs)


def create_v3_router(**kwargs) -> VersionedAPIRouter:
    """Create a V3 API router."""
    return VersionedAPIRouter(version=APIVersion.V3, **kwargs)


# API version info endpoint
version_info_router = APIRouter(prefix="/api", tags=["versioning"])


@version_info_router.get("/versions")
async def get_api_versions():
    """
    Get information about available API versions.

    Returns:
        List of available API versions with their status and details
    """
    versions = []
    for version_id, info in VERSION_INFO.items():
        versions.append(
            {
                "version": version_id,
                "status": info["status"],
                "released": info["released"],
                "sunset_date": info["sunset_date"],
                "description": info["description"],
                "breaking_changes": info.get("breaking_changes", []),
                "url_prefix": f"/api/{version_id}",
            }
        )

    return {
        "current_version": APIVersion.latest().value,
        "versions": versions,
    }


@version_info_router.get("/versions/{version_id}")
async def get_version_info(version_id: str):
    """
    Get information about a specific API version.

    Args:
        version_id: Version identifier (v1, v2, v3)

    Returns:
        Version details
    """
    if version_id not in VERSION_INFO:
        raise HTTPException(
            status_code=404,
            detail=f"Version '{version_id}' not found. Available versions: {list(VERSION_INFO.keys())}",
        )

    info = VERSION_INFO[version_id]
    return {
        "version": version_id,
        "is_current": version_id == APIVersion.latest().value,
        **info,
    }
