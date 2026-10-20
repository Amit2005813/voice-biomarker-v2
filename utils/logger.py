"""Structured logging utilities for the Voice Biomarker Platform."""

import functools
import json
import logging
import time
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

# ---------------------------------------------------------------------------
# Request-ID context variable
# ---------------------------------------------------------------------------

_request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")


def get_request_id() -> str:
    """Return the current request ID (auto-generated if absent)."""
    rid = _request_id_ctx.get()
    if rid == "-":
        rid = uuid.uuid4().hex[:12]
        _request_id_ctx.set(rid)
    return rid


def set_request_id(request_id: str) -> None:
    """Explicitly set the request ID for the current context."""
    _request_id_ctx.set(request_id)


# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------


class JSONFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": get_request_id(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        # Merge any extra fields attached via ``extra=`` in the call.
        for key in ("duration_ms", "status_code", "endpoint", "function"):
            val = getattr(record, key, None)
            if val is not None:
                log_entry[key] = val
        return json.dumps(log_entry, default=str)


# ---------------------------------------------------------------------------
# Logger factory
# ---------------------------------------------------------------------------

_loggers: dict[str, logging.Logger] = {}
_configured = False


def _configure_root() -> None:
    global _configured  # noqa: PLW0603
    if _configured:
        return
    root = logging.getLogger("voicebiomarker")
    root.setLevel(logging.DEBUG)

    # Console handler (readable format)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(console)

    # JSON file handler
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    fh = logging.FileHandler(log_dir / "app.jsonl", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(JSONFormatter())
    root.addHandler(fh)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return (and cache) a named logger under the ``voicebiomarker`` namespace.

    Args:
        name: Logger name (e.g. ``"api.analyzer"``).

    Returns:
        A configured ``logging.Logger`` instance.
    """
    _configure_root()
    full_name = f"voicebiomarker.{name}"
    if full_name not in _loggers:
        _loggers[full_name] = logging.getLogger(full_name)
    return _loggers[full_name]


# ---------------------------------------------------------------------------
# Performance logging decorator
# ---------------------------------------------------------------------------


def log_performance(
    func: Optional[F] = None,
    *,
    logger_name: str = "performance",
) -> Callable[[F], F]:
    """Decorator that logs function execution time.

    Can be used with or without arguments::

        @log_performance
        def heavy_task(): ...

        @log_performance(logger_name="api")
        def api_call(): ...
    """

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = get_logger(logger_name)
            request_id = get_request_id()
            start = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                elapsed = (time.perf_counter() - start) * 1000
                logger.info(
                    "Completed %s in %.2f ms",
                    fn.__qualname__,
                    elapsed,
                    extra={
                        "duration_ms": round(elapsed, 2),
                        "function": fn.__qualname__,
                    },
                )
                return result
            except Exception:
                elapsed = (time.perf_counter() - start) * 1000
                logger.exception(
                    "Failed %s after %.2f ms",
                    fn.__qualname__,
                    elapsed,
                    extra={
                        "duration_ms": round(elapsed, 2),
                        "function": fn.__qualname__,
                    },
                )
                raise

        return wrapper  # type: ignore[return-value]

    if func is not None:
        return decorator(func)
    return decorator
