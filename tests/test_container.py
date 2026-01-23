"""Tests for dependency injection container."""

import pytest

from bot.core.container import (
    Container,
    Lifetime,
    container,
    get,
    inject,
    register,
)


class TestContainer:
    """Tests for Container class."""

    @pytest.fixture(autouse=True)
    def reset_container(self):
        """Reset container before each test."""
        Container.reset()
        yield
        Container.reset()

    def test_register_and_get_instance(self):
        """Test registering and retrieving an instance."""
        c = Container()

        class MyService:
            def __init__(self):
                self.value = 42

        c.register_instance(MyService, MyService())
        service = c.get(MyService)

        assert service.value == 42

    def test_singleton_returns_same_instance(self):
        """Test that singleton lifetime returns same instance."""
        c = Container()

        class MyService:
            pass

        c.register(MyService, factory=MyService)

        instance1 = c.get(MyService)
        instance2 = c.get(MyService)

        assert instance1 is instance2

    def test_transient_returns_new_instances(self):
        """Test that transient lifetime returns new instances."""
        c = Container()

        class MyService:
            pass

        c.register_transient(MyService, factory=MyService)

        instance1 = c.get(MyService)
        instance2 = c.get(MyService)

        assert instance1 is not instance2

    def test_factory_registration(self):
        """Test factory function registration."""
        c = Container()
        call_count = {"count": 0}

        class MyService:
            def __init__(self, value: int):
                self.value = value

        def factory():
            call_count["count"] += 1
            return MyService(call_count["count"])

        c.register_factory(MyService, factory)

        service = c.get(MyService)
        assert service.value == 1
        # Should return same instance (singleton)
        service2 = c.get(MyService)
        assert service2.value == 1
        assert call_count["count"] == 1

    def test_unregistered_service_raises_error(self):
        """Test that getting unregistered service raises KeyError."""
        c = Container()

        class MyService:
            pass

        with pytest.raises(KeyError):
            c.get(MyService)

    def test_try_get_returns_none_for_unregistered(self):
        """Test try_get returns None for unregistered service."""
        c = Container()

        class MyService:
            pass

        result = c.try_get(MyService)
        assert result is None

    def test_is_registered(self):
        """Test is_registered method."""
        c = Container()

        class MyService:
            pass

        assert c.is_registered(MyService) is False
        c.register_instance(MyService, MyService())
        assert c.is_registered(MyService) is True

    def test_scoped_lifetime(self):
        """Test scoped lifetime behavior."""
        c = Container()

        class MyService:
            pass

        c.register(MyService, factory=MyService, lifetime=Lifetime.SCOPED)

        # Different scopes should get different instances
        instance1 = c.get(MyService, scope="scope1")
        instance2 = c.get(MyService, scope="scope2")
        assert instance1 is not instance2

        # Same scope should get same instance
        instance3 = c.get(MyService, scope="scope1")
        assert instance1 is instance3

    def test_scope_context_manager(self):
        """Test scope as context manager."""
        c = Container()

        class MyService:
            pass

        c.register(MyService, factory=MyService, lifetime=Lifetime.SCOPED)

        with c.create_scope("test_scope") as scope:
            instance1 = scope.get(MyService)
            instance2 = scope.get(MyService)
            assert instance1 is instance2

        # Scope should be disposed
        assert "test_scope" not in c._scopes

    def test_constructor_injection(self):
        """Test automatic constructor injection."""
        c = Container()

        class Dependency:
            def __init__(self):
                self.value = "injected"

        class MyService:
            def __init__(self, dep: Dependency):
                self.dep = dep

        c.register(Dependency, factory=Dependency)
        c.register(MyService)

        service = c.get(MyService)
        assert service.dep.value == "injected"

    def test_inject_decorator(self):
        """Test @inject decorator."""
        c = Container()

        class MyService:
            def __init__(self):
                self.value = 123

        c.register(MyService, factory=MyService)

        @c.inject
        def my_function(service: MyService):
            return service.value

        result = my_function()
        assert result == 123

    def test_inject_decorator_with_existing_args(self):
        """Test @inject decorator preserves existing arguments."""
        c = Container()

        class MyService:
            pass

        c.register(MyService, factory=MyService)

        @c.inject
        def my_function(x: int, service: MyService):
            return x, service

        result = my_function(42)
        assert result[0] == 42
        assert isinstance(result[1], MyService)

    def test_chaining_registration(self):
        """Test method chaining for registration."""
        c = Container()

        class Service1:
            pass

        class Service2:
            pass

        c.register(Service1, factory=Service1).register(Service2, factory=Service2)

        assert c.is_registered(Service1)
        assert c.is_registered(Service2)

    def test_global_container_singleton(self):
        """Test global container is a singleton."""
        c1 = Container.get_instance()
        c2 = Container.get_instance()
        assert c1 is c2


class TestGlobalFunctions:
    """Tests for module-level convenience functions."""

    @pytest.fixture(autouse=True)
    def reset_container(self):
        """Reset container before each test."""
        Container.reset()
        yield
        Container.reset()

    def test_global_register_and_get(self):
        """Test global register and get functions."""

        class MyService:
            pass

        register(MyService, factory=MyService)
        service = get(MyService)
        assert isinstance(service, MyService)

    def test_global_inject(self):
        """Test global inject decorator."""

        class MyService:
            def __init__(self):
                self.value = "global"

        register(MyService, factory=MyService)

        @inject
        def my_function(service: MyService):
            return service.value

        assert my_function() == "global"
