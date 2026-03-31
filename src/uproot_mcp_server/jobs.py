"""Thread-safe async job store for long-running dataset kernel jobs.

Jobs are submitted to a background :class:`~concurrent.futures.ThreadPoolExecutor`
and tracked in an in-memory store with LRU eviction (max 20 completed jobs).
"""

from __future__ import annotations

import threading
import uuid
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Literal


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Job:
    """Represents a single async dataset job.

    Parameters
    ----------
    job_id:
        Unique identifier (UUID4 string).
    status:
        Current lifecycle state.
    submitted_at:
        UTC timestamp when the job was created.
    started_at:
        UTC timestamp when the background thread began executing (or ``None``).
    finished_at:
        UTC timestamp when the job reached a terminal state (or ``None``).
    result:
        Final reduced result set when ``status == "done"``; ``None`` otherwise.
    error:
        Error message string when ``status == "failed"``; ``None`` otherwise.
    """

    job_id: str
    status: Literal["pending", "running", "done", "failed", "cancelled"]
    submitted_at: str
    started_at: str | None = None
    finished_at: str | None = None
    result: Any = field(default=None, repr=False)
    error: str | None = None
    _future: Future | None = field(default=None, repr=False)  # type: ignore[type-arg]

    def to_status_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable status dict (no result payload)."""
        return {
            "job_id": self.job_id,
            "status": self.status,
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

_TERMINAL_STATES: frozenset[str] = frozenset({"done", "failed", "cancelled"})
_MAX_COMPLETED: int = 20


class JobStore:
    """Thread-safe in-memory store for async dataset jobs.

    Completed / failed / cancelled jobs are retained up to *max_completed*
    entries (LRU eviction removes the oldest on overflow).

    Parameters
    ----------
    max_workers:
        Maximum number of threads in the background pool (default 4).
    max_completed:
        Maximum number of completed jobs to retain (default 20).
    """

    def __init__(
        self,
        max_workers: int = 4,
        max_completed: int = _MAX_COMPLETED,
    ) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._completed_ids: deque[str] = deque()
        self._max_completed = max_completed
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
        """Submit *fn(*args, **kwargs)* for background execution.

        Parameters
        ----------
        fn:
            Callable to execute in a background thread.
        *args, **kwargs:
            Passed verbatim to *fn*.

        Returns
        -------
        str
            UUID4 job identifier.
        """
        job_id = str(uuid.uuid4())
        now = _utcnow()
        job = Job(
            job_id=job_id,
            status="pending",
            submitted_at=now,
        )
        with self._lock:
            self._jobs[job_id] = job
            future = self._executor.submit(self._run, job_id, fn, args, kwargs)
            job._future = future
        return job_id

    def status(self, job_id: str) -> dict[str, Any]:
        """Return the current status of a job.

        Parameters
        ----------
        job_id:
            Job identifier returned by :meth:`submit`.

        Returns
        -------
        dict with keys ``job_id``, ``status``, ``submitted_at``,
        ``started_at``, ``finished_at``, ``error``.

        Raises
        ------
        KeyError
            If *job_id* is not found.
        """
        with self._lock:
            job = self._jobs[job_id]
            return job.to_status_dict()

    def result(self, job_id: str) -> Any:
        """Return the result of a completed job.

        Parameters
        ----------
        job_id:
            Job identifier returned by :meth:`submit`.

        Returns
        -------
        Any
            The return value of the submitted callable.

        Raises
        ------
        KeyError
            If *job_id* is not found.
        ValueError
            If the job is not yet done (pending/running/cancelled) or failed.
        """
        with self._lock:
            job = self._jobs[job_id]
        if job.status == "failed":
            raise ValueError(f"Job {job_id} failed: {job.error}")
        if job.status != "done":
            raise KeyError(
                f"Job {job_id} is not done yet (status: {job.status})"
            )
        return job.result

    def cancel(self, job_id: str) -> bool:
        """Attempt to cancel a pending job.

        Parameters
        ----------
        job_id:
            Job identifier returned by :meth:`submit`.

        Returns
        -------
        bool
            ``True`` if the job was successfully cancelled (it was still
            pending); ``False`` if it is already running, done, failed, or
            cancelled.

        Raises
        ------
        KeyError
            If *job_id* is not found.
        """
        with self._lock:
            job = self._jobs[job_id]
            if job.status != "pending":
                return False
            # Try to cancel via the future; if the future is already running
            # Future.cancel() returns False.
            if job._future is not None and job._future.cancel():
                job.status = "cancelled"
                job.finished_at = _utcnow()
                self._record_completed(job_id)
                return True
            # The future may have started between the status check and the
            # cancel call; in that case we cannot stop it.
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run(
        self,
        job_id: str,
        fn: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        """Background wrapper: updates job status around *fn* execution."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status == "cancelled":
                return
            job.status = "running"
            job.started_at = _utcnow()

        try:
            result = fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                job = self._jobs.get(job_id)
                if job is not None:
                    job.status = "failed"
                    job.error = str(exc)
                    job.finished_at = _utcnow()
                    self._record_completed(job_id)
            return

        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.status = "done"
                job.result = result
                job.finished_at = _utcnow()
                self._record_completed(job_id)

    def _record_completed(self, job_id: str) -> None:
        """Register *job_id* in the LRU deque and evict old entries.

        Must be called with :attr:`_lock` held.
        """
        self._completed_ids.append(job_id)
        while len(self._completed_ids) > self._max_completed:
            oldest = self._completed_ids.popleft()
            self._jobs.pop(oldest, None)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _utcnow() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(tz=timezone.utc).isoformat()
