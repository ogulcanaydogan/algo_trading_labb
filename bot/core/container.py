"""
Dependency Injection Container.

Provides centralized dependency management for the trading system,
replacing scattered global singletons with a proper IoC container.

Usage:
    # Register a dependency
    container.register(MarketDataService, factory=lambda: MarketDataService())

    # Get a dependency
    service = container.get(MarketDataService)

    # Use with decorator
    @container.inject
    def my_function(service: MarketDataService):
        pass
"""

from __future__ import annotations

import inspect
import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Optional,
    Type,
    TypeVar,
    Union,
    get_type_hints,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Lifetime(Enum):
    """Dependency lifetime."""

    SINGLETON = "singleton"  # One instance for the entire application
    TRANSIENT = "transient"  # New instance on each request
    SCOPED = "scoped"        # One instance per scope (e.g., per request)


@dataclass
class Registration:
    """Dependency registration."""

    service_type: Type
    factory: Optional[Callable[[], Any]] = None
    instance: Optional[Any] = None
    lifetime: Lifetime = Lifetime.SINGLETON

    def create_instance(self, container: "Container") -> Any:
        """Create an instance of the service."""
        if self.instance is not None and self.lifetime == Lifetime.SINGLETON:
            return self.instance

        if self.factory is not None:
            instance = self.factory()
        else:
            # Try to instantiate with dependency injection
            instance = container._create_with_injection(self.service_type)

        if self.lifetime == Lifetime.SINGLETON:
            self.instance = instance

        return instance


class Container:
    """
    Dependency Injection Container.

    Thread-safe container for managing service dependencies.
    Supports singleton, transient, and scoped lifetimes.
    """

    _instance: Optional["Container"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._registrations: Dict[Type, Registration] = {}
        self._scopes: Dict[str, Dict[Type, Any]] = {}
        self._lock = threading.RLock()

    @classmethod
    def get_instance(cls) -> "Container":
        """Get or create the global container instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the global container (for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance._registrations.clear()
                cls._instance._scopes.clear()
            cls._instance = None

    def register(
        self,
        service_type: Type[T],
        factory: Optional[Callable[[], T]] = None,
        instance: Optional[T] = None,
        lifetime: Lifetime = Lifetime.SINGLETON,
    ) -> "Container":
        """
        Register a service with the container.

        Args:
            service_type: The type to register
            factory: Factory function to create instances
            instance: Pre-created instance (for singletons)
            lifetime: Service lifetime

        Returns:
            Self for chaining
        """
        with self._lock:
            self._registrations[service_type] = Registration(
                service_type=service_type,
                factory=factory,
                instance=instance,
                lifetime=lifetime,
            )
        return self

    def register_instance(self, service_type: Type[T], instance: T) -> "Container":
        """Register an existing instance as a singleton."""
        return self.register(
            service_type,
            instance=instance,
            lifetime=Lifetime.SINGLETON,
        )

    def register_factory(
        self,
        service_type: Type[T],
        factory: Callable[[], T],
        lifetime: Lifetime = Lifetime.SINGLETON,
    ) -> "Container":
        """Register a factory function."""
        return self.register(
            service_type,
            factory=factory,
            lifetime=lifetime,
        )

    def register_transient(
        self,
        service_type: Type[T],
        factory: Optional[Callable[[], T]] = None,
    ) -> "Container":
        """Register a transient service (new instance each time)."""
        return self.register(
            service_type,
            factory=factory,
            lifetime=Lifetime.TRANSIENT,
        )

    def get(self, service_type: Type[T], scope: Optional[str] = None) -> T:
        """
        Get a service instance.

        Args:
            service_type: The type to resolve
            scope: Optional scope name for scoped services

        Returns:
            Service instance

        Raises:
            KeyError: If service is not registered
        """
        with self._lock:
            if service_type not in self._registrations:
                raise KeyError(f"Service {service_type.__name__} is not registered")

            registration = self._registrations[service_type]

            # Handle scoped lifetime
            if registration.lifetime == Lifetime.SCOPED and scope:
                if scope not in self._scopes:
                    self._scopes[scope] = {}
                if service_type not in self._scopes[scope]:
                    self._scopes[scope][service_type] = registration.create_instance(self)
                return self._scopes[scope][service_type]

            return registration.create_instance(self)

    def try_get(self, service_type: Type[T]) -> Optional[T]:
        """Try to get a service, returning None if not registered."""
        try:
            return self.get(service_type)
        except KeyError:
            return None

    def is_registered(self, service_type: Type) -> bool:
        """Check if a service type is registered."""
        return service_type in self._registrations

    def create_scope(self, scope_name: str) -> "Scope":
        """Create a new scope for scoped services."""
        return Scope(self, scope_name)

    def dispose_scope(self, scope_name: str) -> None:
        """Dispose of a scope and its instances."""
        with self._lock:
            if scope_name in self._scopes:
                # Call dispose on instances that have it
                for instance in self._scopes[scope_name].values():
                    if hasattr(instance, "dispose"):
                        try:
                            instance.dispose()
                        except Exception as e:
                            logger.warning(f"Error disposing {type(instance)}: {e}")
                del self._scopes[scope_name]

    def _create_with_injection(self, service_type: Type[T]) -> T:
        """Create an instance with constructor injection."""
        try:
            # Get constructor signature
            sig = inspect.signature(service_type.__init__)
            hints = get_type_hints(service_type.__init__) if hasattr(service_type.__init__, "__annotations__") else {}

            kwargs = {}
            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue

                # Get type hint
                param_type = hints.get(param_name)
                if param_type and self.is_registered(param_type):
                    kwargs[param_name] = self.get(param_type)
                elif param.default is not inspect.Parameter.empty:
                    # Use default value
                    pass
                # Otherwise let it fail naturally

            return service_type(**kwargs)
        except Exception as e:
            logger.debug(f"Could not auto-inject for {service_type}: {e}")
            # Fallback to simple construction
            return service_type()

    def inject(self, func: Callable) -> Callable:
        """
        Decorator for automatic dependency injection.

        Usage:
            @container.inject
            def my_function(service: MarketDataService):
                pass
        """
        import functools

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get type hints
            hints = get_type_hints(func) if hasattr(func, "__annotations__") else {}
            sig = inspect.signature(func)

            # Inject missing dependencies
            for param_name, param_type in hints.items():
                if param_name in kwargs:
                    continue
                if param_name == "return":
                    continue
                if self.is_registered(param_type):
                    kwargs[param_name] = self.get(param_type)

            return func(*args, **kwargs)

        return wrapper


class Scope:
    """
    Scoped container context.

    Usage:
        with container.create_scope("request_123") as scope:
            service = scope.get(MyScopedService)
    """

    def __init__(self, container: Container, name: str):
        self._container = container
        self._name = name

    def __enter__(self) -> "Scope":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._container.dispose_scope(self._name)

    def get(self, service_type: Type[T]) -> T:
        """Get a service within this scope."""
        return self._container.get(service_type, scope=self._name)


# Global container instance
container = Container.get_instance()


def register(
    service_type: Type[T],
    factory: Optional[Callable[[], T]] = None,
    instance: Optional[T] = None,
    lifetime: Lifetime = Lifetime.SINGLETON,
) -> None:
    """Register a service with the global container."""
    container.register(service_type, factory, instance, lifetime)


def get(service_type: Type[T]) -> T:
    """Get a service from the global container."""
    return container.get(service_type)


def inject(func: Callable) -> Callable:
    """Decorator for automatic dependency injection using global container."""
    return container.inject(func)
