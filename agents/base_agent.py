"""Abstract base agent for Voice Biomarker Platform v2."""

from __future__ import annotations

import logging
import time
import traceback
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class AgentStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class BaseAgent(ABC):
    """Abstract base class for all analysis agents.

    Provides common logging, error handling, and status tracking.
    Subclasses must implement :meth:`run`.

    Parameters
    ----------
    name : str
        Human-readable agent name used in logs and reports.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.logger = logging.getLogger(f"agents.{name}")
        self.status: AgentStatus = AgentStatus.IDLE
        self.last_error: Optional[str] = None
        self._run_count: int = 0
        self._total_duration: float = 0.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def average_duration(self) -> float:
        """Average wall-clock seconds per run."""
        if self._run_count == 0:
            return 0.0
        return self._total_duration / self._run_count

    # ------------------------------------------------------------------
    # Template method
    # ------------------------------------------------------------------

    def run(self, **kwargs: Any) -> Dict[str, Any]:
        """Execute the agent with logging, timing, and error handling.

        Calls :meth:`_execute` internally. Subclasses should override
        ``_execute`` rather than ``run``.
        """
        self.status = AgentStatus.RUNNING
        self.last_error = None
        start = time.perf_counter()

        self.logger.info("[%s] Starting execution", self.name)
        try:
            result = self._execute(**kwargs)
            elapsed = time.perf_counter() - start
            self._run_count += 1
            self._total_duration += elapsed
            self.status = AgentStatus.COMPLETED
            self.logger.info(
                "[%s] Completed in %.3fs", self.name, elapsed,
            )
            return result
        except Exception as exc:
            elapsed = time.perf_counter() - start
            self._run_count += 1
            self._total_duration += elapsed
            self.status = AgentStatus.FAILED
            self.last_error = str(exc)
            self.logger.error(
                "[%s] Failed after %.3fs: %s\n%s",
                self.name, elapsed, exc, traceback.format_exc(),
            )
            raise

    @abstractmethod
    def _execute(self, **kwargs: Any) -> Dict[str, Any]:
        """Agent-specific logic. Must be implemented by subclasses."""
        ...

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_info(self) -> Dict[str, Any]:
        """Return agent metadata for reporting."""
        return {
            "name": self.name,
            "status": self.status.value,
            "run_count": self._run_count,
            "average_duration": round(self.average_duration, 4),
            "last_error": self.last_error,
        }

    def reset(self) -> None:
        """Reset status to idle and clear error state."""
        self.status = AgentStatus.IDLE
        self.last_error = None

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r} status={self.status.value}>"
