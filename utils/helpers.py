"""General-purpose helper utilities."""

import functools
import math
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

# ---------------------------------------------------------------------------
# ID / time formatting
# ---------------------------------------------------------------------------


def generate_id() -> str:
    """Return a unique 32-character hex identifier."""
    return uuid.uuid4().hex


def format_timestamp(
    dt: Optional[datetime] = None,
    fmt: str = "%Y-%m-%dT%H:%M:%S%z",
) -> str:
    """Format a datetime as an ISO-style string.

    Args:
        dt: Datetime to format. Defaults to current UTC time.
        fmt: strftime format string.

    Returns:
        Formatted string.
    """
    if dt is None:
        dt = datetime.now(timezone.utc)
    return dt.strftime(fmt)


def format_filesize(size_bytes: int) -> str:
    """Convert a byte count into a human-readable string.

    Examples::

        >>> format_filesize(1536)
        '1.50 KB'
        >>> format_filesize(1048576)
        '1.00 MB'
    """
    if size_bytes < 0:
        raise ValueError("size_bytes must be non-negative")
    if size_bytes == 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    exp = int(math.log(size_bytes, 1024))
    exp = min(exp, len(units) - 1)
    value = size_bytes / (1024 ** exp)
    return f"{value:.2f} {units[exp]}"


# ---------------------------------------------------------------------------
# Filename sanitisation
# ---------------------------------------------------------------------------

_UNSAFE_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def sanitize_filename(name: str, replacement: str = "_") -> str:
    """Remove or replace characters unsafe for file paths.

    Args:
        name: Original filename.
        replacement: Character to substitute for unsafe characters.

    Returns:
        A safe filename string.
    """
    safe = _UNSAFE_RE.sub(replacement, name)
    safe = safe.strip(". ")
    return safe or "unnamed"


# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------


def retry(
    func: Optional[F] = None,
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Callable[[F], F]:
    """Retry a function with exponential back-off.

    Args:
        func: Function to decorate.
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay in seconds.
        max_delay: Upper-bound delay in seconds.
        exceptions: Exception types that trigger a retry.

    Usage::

        @retry(max_retries=5, base_delay=0.5)
        def flaky_call(): ...
    """

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Optional[BaseException] = None
            for attempt in range(max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt == max_retries:
                        break
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    time.sleep(delay)
            raise last_exc  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    if func is not None:
        return decorator(func)
    return decorator


# ---------------------------------------------------------------------------
# Chunk iterator
# ---------------------------------------------------------------------------


def chunk_iterator(
    data: Any,
    chunk_size: int = 1024,
) -> Iterator[Any]:
    """Yield successive chunks from *data*.

    Works with ``bytes``, ``list``, ``numpy.ndarray``, ``str``, and anything
    implementing ``__getitem__`` and ``__len__``.

    Args:
        data: Source collection.
        chunk_size: Maximum size of each chunk.

    Yields:
        Sub-sequences of *data*.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    length = len(data)
    for start in range(0, length, chunk_size):
        yield data[start : start + chunk_size]
