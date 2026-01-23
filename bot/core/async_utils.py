"""
Async Safety Utilities.

Provides safe async patterns for the trading system:
- Timeout handling
- Task cancellation
- Semaphore-based rate limiting
- Retry with exponential backoff
- Context managers for safe cleanup

Usage:
    from bot.core.async_utils import safe_timeout, retry_async, RateLimiter

    # With timeout
    async with safe_timeout(5.0) as ctx:
        result = await some_operation()
        if ctx.timed_out:
            handle_timeout()

    # With retry
    @retry_async(max_attempts=3, backoff_factor=2.0)
    async def fetch_data():
        ...

    # With rate limiting
    limiter = RateLimiter(calls_per_second=10)
    async with limiter:
        await api_call()
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class TimeoutContext:
    """Context for timeout operations."""

    timeout: float
    timed_out: bool = False
    elapsed: float = 0.0


@asynccontextmanager
async def safe_timeout(
    timeout: float,
    shield: bool = False,
    cancel_message: Optional[str] = None,
):
    """
    Context manager for safe timeout handling.

    Args:
        timeout: Timeout in seconds
        shield: If True, protect from cancellation
        cancel_message: Optional message on timeout

    Yields:
        TimeoutContext with timed_out flag

    Example:
        async with safe_timeout(5.0) as ctx:
            await some_operation()
        if ctx.timed_out:
            logger.warning("Operation timed out")
    """
    ctx = TimeoutContext(timeout=timeout)
    start = time.monotonic()

    try:
        if shield:
            async with asyncio.timeout(timeout):
                yield ctx
        else:
            async with asyncio.timeout(timeout):
                yield ctx
    except asyncio.TimeoutError:
        ctx.timed_out = True
        ctx.elapsed = time.monotonic() - start
        if cancel_message:
            logger.warning(f"{cancel_message} (after {ctx.elapsed:.2f}s)")
    finally:
        ctx.elapsed = time.monotonic() - start


async def wait_with_timeout(
    coro: Awaitable[T],
    timeout: float,
    default: Optional[T] = None,
) -> Optional[T]:
    """
    Wait for a coroutine with timeout, returning default on timeout.

    Args:
        coro: Coroutine to await
        timeout: Timeout in seconds
        default: Value to return on timeout

    Returns:
        Result or default value
    """
    try:
        async with asyncio.timeout(timeout):
            return await coro
    except asyncio.TimeoutError:
        return default


def retry_async(
    max_attempts: int = 3,
    backoff_factor: float = 2.0,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
):
    """
    Decorator for async retry with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts
        backoff_factor: Multiplier for delay between retries
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exceptions: Tuple of exceptions to catch
        on_retry: Callback called on each retry (exception, attempt_number)

    Example:
        @retry_async(max_attempts=3, backoff_factor=2.0)
        async def fetch_data():
            return await api.get_data()
    """

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            delay = initial_delay
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_attempts:
                        raise

                    if on_retry:
                        on_retry(e, attempt)

                    logger.debug(f"Retry {attempt}/{max_attempts} for {func.__name__}: {e}")
                    await asyncio.sleep(delay)
                    delay = min(delay * backoff_factor, max_delay)

            raise last_exception  # Should not reach here

        return wrapper

    return decorator


class RateLimiter:
    """
    Async rate limiter using token bucket algorithm.

    Thread-safe rate limiting for API calls.

    Example:
        limiter = RateLimiter(calls_per_second=10)

        async with limiter:
            await api_call()

        # Or with await
        await limiter.acquire()
        try:
            await api_call()
        finally:
            limiter.release()
    """

    def __init__(
        self,
        calls_per_second: float = 10.0,
        burst_size: Optional[int] = None,
    ):
        """
        Initialize rate limiter.

        Args:
            calls_per_second: Maximum calls per second
            burst_size: Maximum burst size (defaults to calls_per_second)
        """
        self.calls_per_second = calls_per_second
        self.burst_size = burst_size or int(calls_per_second)
        self._tokens = float(self.burst_size)
        self._last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire a token, waiting if necessary."""
        async with self._lock:
            await self._wait_for_token()
            self._tokens -= 1

    async def _wait_for_token(self) -> None:
        """Wait until a token is available."""
        while True:
            self._refill_tokens()
            if self._tokens >= 1:
                return

            # Calculate wait time for next token
            tokens_needed = 1 - self._tokens
            wait_time = tokens_needed / self.calls_per_second
            await asyncio.sleep(wait_time)

    def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_update
        self._last_update = now

        new_tokens = elapsed * self.calls_per_second
        self._tokens = min(self._tokens + new_tokens, self.burst_size)

    def release(self) -> None:
        """Release is a no-op for token bucket (tokens auto-refill)."""
        pass

    async def __aenter__(self) -> "RateLimiter":
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


class AsyncTaskManager:
    """
    Manages async tasks with proper cancellation and cleanup.

    Example:
        manager = AsyncTaskManager()

        async with manager:
            manager.create_task(some_coroutine())
            manager.create_task(another_coroutine())
            # Tasks run concurrently
        # All tasks cancelled and cleaned up on exit
    """

    def __init__(self, name: str = "TaskManager"):
        self.name = name
        self._tasks: Set[asyncio.Task] = set()
        self._lock = asyncio.Lock()
        self._shutdown = False

    def create_task(
        self,
        coro: Awaitable[T],
        name: Optional[str] = None,
    ) -> asyncio.Task[T]:
        """
        Create and track a task.

        Args:
            coro: Coroutine to run
            name: Optional task name

        Returns:
            Created task
        """
        if self._shutdown:
            raise RuntimeError(f"{self.name} is shutting down")

        task = asyncio.create_task(coro, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._task_done)
        return task

    def _task_done(self, task: asyncio.Task) -> None:
        """Callback when task completes."""
        self._tasks.discard(task)

        # Log exceptions from tasks
        if not task.cancelled():
            exc = task.exception()
            if exc:
                logger.error(
                    f"Task {task.get_name()} failed: {exc}",
                    exc_info=exc,
                )

    async def cancel_all(self, timeout: float = 5.0) -> None:
        """
        Cancel all running tasks.

        Args:
            timeout: Timeout for task cancellation
        """
        self._shutdown = True

        for task in list(self._tasks):
            if not task.done():
                task.cancel()

        if self._tasks:
            try:
                async with asyncio.timeout(timeout):
                    await asyncio.gather(*self._tasks, return_exceptions=True)
            except asyncio.TimeoutError:
                logger.warning(f"{self.name}: Some tasks didn't cancel in time")

    async def wait_all(self, timeout: Optional[float] = None) -> List[Any]:
        """
        Wait for all tasks to complete.

        Args:
            timeout: Optional timeout

        Returns:
            List of results
        """
        if not self._tasks:
            return []

        try:
            if timeout:
                async with asyncio.timeout(timeout):
                    return await asyncio.gather(*self._tasks, return_exceptions=True)
            else:
                return await asyncio.gather(*self._tasks, return_exceptions=True)
        except asyncio.TimeoutError:
            await self.cancel_all()
            return []

    async def __aenter__(self) -> "AsyncTaskManager":
        self._shutdown = False
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.cancel_all()


class Semaphore:
    """
    Enhanced async semaphore with timeout support.

    Example:
        sem = Semaphore(max_concurrent=5)

        async with sem.acquire(timeout=10.0):
            await do_work()
    """

    def __init__(self, max_concurrent: int = 10):
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._max = max_concurrent

    @asynccontextmanager
    async def acquire(self, timeout: Optional[float] = None):
        """
        Acquire semaphore with optional timeout.

        Args:
            timeout: Optional timeout in seconds

        Raises:
            asyncio.TimeoutError: If timeout expires
        """
        if timeout:
            async with asyncio.timeout(timeout):
                async with self._semaphore:
                    yield
        else:
            async with self._semaphore:
                yield

    @property
    def available(self) -> int:
        """Number of available slots."""
        # Note: This is approximate due to async nature
        return self._semaphore._value


async def run_with_semaphore(
    coros: List[Awaitable[T]],
    max_concurrent: int = 10,
) -> List[T]:
    """
    Run coroutines with limited concurrency.

    Args:
        coros: List of coroutines
        max_concurrent: Maximum concurrent tasks

    Returns:
        List of results in original order
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    results = []

    async def bounded_coro(index: int, coro: Awaitable[T]) -> tuple:
        async with semaphore:
            result = await coro
            return (index, result)

    tasks = [asyncio.create_task(bounded_coro(i, coro)) for i, coro in enumerate(coros)]

    completed = await asyncio.gather(*tasks, return_exceptions=True)

    # Sort by original index
    sorted_results = sorted(
        [(idx, res) for idx, res in completed if not isinstance(res, Exception)],
        key=lambda x: x[0],
    )

    return [res for _, res in sorted_results]


# =============================================================================
# Async File I/O Utilities
# =============================================================================


async def async_read_json(
    file_path: Union[str, "Path"],
    default: Optional[Any] = None,
) -> Any:
    """
    Read JSON file asynchronously (non-blocking).

    Args:
        file_path: Path to JSON file
        default: Value to return if file doesn't exist or is invalid

    Returns:
        Parsed JSON data or default value

    Example:
        data = await async_read_json("data/state.json", default={})
    """
    import json
    from pathlib import Path

    path = Path(file_path) if isinstance(file_path, str) else file_path

    def _read():
        if not path.exists():
            return default
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to read {path}: {e}")
            return default

    return await asyncio.to_thread(_read)


async def async_write_json(
    file_path: Union[str, "Path"],
    data: Any,
    indent: int = 2,
    ensure_dir: bool = True,
) -> bool:
    """
    Write JSON file asynchronously (non-blocking).

    Args:
        file_path: Path to JSON file
        data: Data to serialize
        indent: JSON indentation
        ensure_dir: Create parent directories if needed

    Returns:
        True if successful, False otherwise

    Example:
        success = await async_write_json("data/state.json", {"balance": 1000})
    """
    import json
    from pathlib import Path

    path = Path(file_path) if isinstance(file_path, str) else file_path

    def _write():
        try:
            if ensure_dir:
                path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(data, f, indent=indent, default=str)
            return True
        except (OSError, TypeError) as e:
            logger.error(f"Failed to write {path}: {e}")
            return False

    return await asyncio.to_thread(_write)


async def async_read_text(
    file_path: Union[str, "Path"],
    default: Optional[str] = None,
) -> Optional[str]:
    """
    Read text file asynchronously (non-blocking).

    Args:
        file_path: Path to text file
        default: Value to return if file doesn't exist

    Returns:
        File contents or default value
    """
    from pathlib import Path

    path = Path(file_path) if isinstance(file_path, str) else file_path

    def _read():
        if not path.exists():
            return default
        try:
            with open(path, "r") as f:
                return f.read()
        except OSError as e:
            logger.warning(f"Failed to read {path}: {e}")
            return default

    return await asyncio.to_thread(_read)


async def async_write_text(
    file_path: Union[str, "Path"],
    content: str,
    ensure_dir: bool = True,
) -> bool:
    """
    Write text file asynchronously (non-blocking).

    Args:
        file_path: Path to text file
        content: Text content to write
        ensure_dir: Create parent directories if needed

    Returns:
        True if successful, False otherwise
    """
    from pathlib import Path

    path = Path(file_path) if isinstance(file_path, str) else file_path

    def _write():
        try:
            if ensure_dir:
                path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                f.write(content)
            return True
        except OSError as e:
            logger.error(f"Failed to write {path}: {e}")
            return False

    return await asyncio.to_thread(_write)


async def async_append_json(
    file_path: Union[str, "Path"],
    item: Any,
    max_items: Optional[int] = None,
) -> bool:
    """
    Append item to JSON array file asynchronously.

    Args:
        file_path: Path to JSON file (should contain array)
        item: Item to append
        max_items: Optional max items to keep (removes oldest)

    Returns:
        True if successful, False otherwise

    Example:
        await async_append_json("data/trades.json", trade_data, max_items=1000)
    """
    import json
    from pathlib import Path

    path = Path(file_path) if isinstance(file_path, str) else file_path

    def _append():
        try:
            # Read existing
            data = []
            if path.exists():
                try:
                    with open(path, "r") as f:
                        data = json.load(f)
                except (json.JSONDecodeError, OSError):
                    data = []

            # Append
            if not isinstance(data, list):
                data = []
            data.append(item)

            # Trim if needed
            if max_items and len(data) > max_items:
                data = data[-max_items:]

            # Write back
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)
            return True
        except (OSError, TypeError) as e:
            logger.error(f"Failed to append to {path}: {e}")
            return False

    return await asyncio.to_thread(_append)


async def async_file_exists(file_path: Union[str, "Path"]) -> bool:
    """Check if file exists asynchronously."""
    from pathlib import Path

    path = Path(file_path) if isinstance(file_path, str) else file_path
    return await asyncio.to_thread(path.exists)


def create_safe_callback(
    callback: Callable,
    error_handler: Optional[Callable[[Exception], None]] = None,
) -> Callable:
    """
    Wrap a callback to handle exceptions safely.

    Args:
        callback: Original callback
        error_handler: Optional error handler

    Returns:
        Safe wrapper function
    """
    if asyncio.iscoroutinefunction(callback):

        @functools.wraps(callback)
        async def async_wrapper(*args, **kwargs):
            try:
                return await callback(*args, **kwargs)
            except Exception as e:
                if error_handler:
                    error_handler(e)
                else:
                    logger.exception(f"Callback error: {e}")
                return None

        return async_wrapper
    else:

        @functools.wraps(callback)
        def sync_wrapper(*args, **kwargs):
            try:
                return callback(*args, **kwargs)
            except Exception as e:
                if error_handler:
                    error_handler(e)
                else:
                    logger.exception(f"Callback error: {e}")
                return None

        return sync_wrapper
