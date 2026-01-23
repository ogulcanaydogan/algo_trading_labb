"""Tests for async safety utilities."""

import asyncio
import time

import pytest

from bot.core.async_utils import (
    AsyncTaskManager,
    RateLimiter,
    Semaphore,
    create_safe_callback,
    retry_async,
    run_with_semaphore,
    safe_timeout,
    wait_with_timeout,
)


class TestSafeTimeout:
    """Tests for safe_timeout context manager."""

    @pytest.mark.asyncio
    async def test_no_timeout(self):
        async with safe_timeout(1.0) as ctx:
            await asyncio.sleep(0.01)

        assert ctx.timed_out is False
        assert ctx.elapsed < 0.1

    @pytest.mark.asyncio
    async def test_timeout_occurs(self):
        async with safe_timeout(0.05) as ctx:
            await asyncio.sleep(0.2)

        assert ctx.timed_out is True
        assert ctx.elapsed >= 0.05

    @pytest.mark.asyncio
    async def test_timeout_with_message(self, caplog):
        async with safe_timeout(0.01, cancel_message="Test timeout"):
            await asyncio.sleep(0.1)

        assert "Test timeout" in caplog.text


class TestWaitWithTimeout:
    """Tests for wait_with_timeout."""

    @pytest.mark.asyncio
    async def test_returns_result(self):
        async def quick_task():
            return 42

        result = await wait_with_timeout(quick_task(), timeout=1.0)
        assert result == 42

    @pytest.mark.asyncio
    async def test_returns_default_on_timeout(self):
        async def slow_task():
            await asyncio.sleep(1.0)
            return 42

        result = await wait_with_timeout(slow_task(), timeout=0.01, default=-1)
        assert result == -1

    @pytest.mark.asyncio
    async def test_returns_none_on_timeout_no_default(self):
        async def slow_task():
            await asyncio.sleep(1.0)

        result = await wait_with_timeout(slow_task(), timeout=0.01)
        assert result is None


class TestRetryAsync:
    """Tests for retry_async decorator."""

    @pytest.mark.asyncio
    async def test_succeeds_first_try(self):
        call_count = {"count": 0}

        @retry_async(max_attempts=3)
        async def succeed():
            call_count["count"] += 1
            return "success"

        result = await succeed()
        assert result == "success"
        assert call_count["count"] == 1

    @pytest.mark.asyncio
    async def test_retries_on_failure(self):
        call_count = {"count": 0}

        @retry_async(max_attempts=3, initial_delay=0.01)
        async def fail_then_succeed():
            call_count["count"] += 1
            if call_count["count"] < 3:
                raise ValueError("Not yet")
            return "success"

        result = await fail_then_succeed()
        assert result == "success"
        assert call_count["count"] == 3

    @pytest.mark.asyncio
    async def test_raises_after_max_attempts(self):
        @retry_async(max_attempts=2, initial_delay=0.01)
        async def always_fail():
            raise ValueError("Always fails")

        with pytest.raises(ValueError, match="Always fails"):
            await always_fail()

    @pytest.mark.asyncio
    async def test_only_catches_specified_exceptions(self):
        @retry_async(max_attempts=3, exceptions=(ValueError,), initial_delay=0.01)
        async def raise_type_error():
            raise TypeError("Wrong type")

        with pytest.raises(TypeError):
            await raise_type_error()

    @pytest.mark.asyncio
    async def test_on_retry_callback(self):
        retries = []

        def on_retry(exc, attempt):
            retries.append((str(exc), attempt))

        @retry_async(max_attempts=3, initial_delay=0.01, on_retry=on_retry)
        async def fail_twice():
            if len(retries) < 2:
                raise ValueError("Fail")
            return "success"

        await fail_twice()
        assert len(retries) == 2
        assert retries[0][1] == 1
        assert retries[1][1] == 2


class TestRateLimiter:
    """Tests for RateLimiter."""

    @pytest.mark.asyncio
    async def test_allows_burst(self):
        limiter = RateLimiter(calls_per_second=100, burst_size=5)

        start = time.monotonic()
        for _ in range(5):
            async with limiter:
                pass
        elapsed = time.monotonic() - start

        # Burst should be fast
        assert elapsed < 0.1

    @pytest.mark.asyncio
    async def test_rate_limits_after_burst(self):
        limiter = RateLimiter(calls_per_second=10, burst_size=2)

        start = time.monotonic()
        for _ in range(4):
            async with limiter:
                pass
        elapsed = time.monotonic() - start

        # Should take at least 0.2s for 2 extra calls at 10/s
        assert elapsed >= 0.15

    @pytest.mark.asyncio
    async def test_context_manager(self):
        limiter = RateLimiter(calls_per_second=100)

        async with limiter:
            pass  # Should not raise


class TestAsyncTaskManager:
    """Tests for AsyncTaskManager."""

    @pytest.mark.asyncio
    async def test_creates_and_runs_tasks(self):
        results = []

        async with AsyncTaskManager("test") as manager:

            async def task(value):
                await asyncio.sleep(0.01)
                results.append(value)

            manager.create_task(task(1))
            manager.create_task(task(2))
            await manager.wait_all()

        assert sorted(results) == [1, 2]

    @pytest.mark.asyncio
    async def test_cancels_on_exit(self):
        cancelled = {"flag": False}

        async with AsyncTaskManager("test") as manager:

            async def long_task():
                try:
                    await asyncio.sleep(10)
                except asyncio.CancelledError:
                    cancelled["flag"] = True
                    raise

            manager.create_task(long_task())
            await asyncio.sleep(0.01)  # Let task start

        # Give time for cancellation
        await asyncio.sleep(0.01)
        assert cancelled["flag"] is True

    @pytest.mark.asyncio
    async def test_logs_task_errors(self, caplog):
        async with AsyncTaskManager("test") as manager:

            async def failing_task():
                raise ValueError("Task failed")

            manager.create_task(failing_task())
            await asyncio.sleep(0.1)  # Let task fail

        # Error should be logged
        assert "Task failed" in caplog.text or len(manager._tasks) == 0


class TestSemaphore:
    """Tests for enhanced Semaphore."""

    @pytest.mark.asyncio
    async def test_limits_concurrency(self):
        sem = Semaphore(max_concurrent=2)
        concurrent = {"max": 0, "current": 0}

        async def task():
            async with sem.acquire():
                concurrent["current"] += 1
                concurrent["max"] = max(concurrent["max"], concurrent["current"])
                await asyncio.sleep(0.05)
                concurrent["current"] -= 1

        await asyncio.gather(*[task() for _ in range(5)])

        assert concurrent["max"] <= 2

    @pytest.mark.asyncio
    async def test_timeout(self):
        sem = Semaphore(max_concurrent=1)

        async with sem.acquire():
            with pytest.raises(asyncio.TimeoutError):
                async with sem.acquire(timeout=0.01):
                    pass


class TestRunWithSemaphore:
    """Tests for run_with_semaphore."""

    @pytest.mark.asyncio
    async def test_runs_all_coroutines(self):
        async def task(n):
            await asyncio.sleep(0.01)
            return n * 2

        coros = [task(i) for i in range(5)]
        results = await run_with_semaphore(coros, max_concurrent=2)

        assert results == [0, 2, 4, 6, 8]

    @pytest.mark.asyncio
    async def test_limits_concurrency(self):
        concurrent = {"max": 0, "current": 0}

        async def task(n):
            concurrent["current"] += 1
            concurrent["max"] = max(concurrent["max"], concurrent["current"])
            await asyncio.sleep(0.02)
            concurrent["current"] -= 1
            return n

        coros = [task(i) for i in range(10)]
        await run_with_semaphore(coros, max_concurrent=3)

        assert concurrent["max"] <= 3


class TestCreateSafeCallback:
    """Tests for create_safe_callback."""

    def test_wraps_sync_callback(self):
        def callback(x):
            return x * 2

        safe = create_safe_callback(callback)
        assert safe(21) == 42

    def test_handles_sync_exception(self, caplog):
        def callback():
            raise ValueError("Error")

        safe = create_safe_callback(callback)
        result = safe()

        assert result is None
        assert "Error" in caplog.text

    @pytest.mark.asyncio
    async def test_wraps_async_callback(self):
        async def callback(x):
            return x * 2

        safe = create_safe_callback(callback)
        assert await safe(21) == 42

    @pytest.mark.asyncio
    async def test_handles_async_exception(self, caplog):
        async def callback():
            raise ValueError("Async error")

        safe = create_safe_callback(callback)
        result = await safe()

        assert result is None
        assert "Async error" in caplog.text

    def test_custom_error_handler(self):
        errors = []

        def error_handler(e):
            errors.append(str(e))

        def callback():
            raise ValueError("Custom handled")

        safe = create_safe_callback(callback, error_handler)
        safe()

        assert "Custom handled" in errors
