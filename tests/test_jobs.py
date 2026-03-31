"""Tests for the JobStore in uproot_mcp_server.jobs."""

from __future__ import annotations

import time

import pytest

from uproot_mcp_server.jobs import JobStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slow_fn(delay: float = 0.05) -> str:
    """Simple function that sleeps and returns a string."""
    time.sleep(delay)
    return "done"


def _failing_fn() -> None:
    """Function that always raises."""
    raise ValueError("intentional failure")


def _fast_fn(value: int = 42) -> int:
    """Fast function returning an integer."""
    return value


def _job_exists(store: JobStore, job_id: str) -> bool:
    """Return True if *job_id* is still tracked in *store*."""
    try:
        store.status(job_id)
        return True
    except KeyError:
        return False


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestJobStore:
    def test_submit_and_wait(self) -> None:
        """Submit a job, poll until done, and retrieve the result."""
        store = JobStore(max_workers=2)
        job_id = store.submit(_fast_fn, 99)

        # Wait for completion (up to 5 s)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            s = store.status(job_id)
            if s["status"] in ("done", "failed", "cancelled"):
                break
            time.sleep(0.01)

        s = store.status(job_id)
        assert s["status"] == "done"
        assert store.result(job_id) == 99

    def test_cancel_pending(self) -> None:
        """Cancel a job before it starts running."""
        # Use a single-worker store with a blocking job in front so our
        # target job stays pending long enough to cancel.
        store = JobStore(max_workers=1)
        # Occupy the single worker thread
        _blocker = store.submit(_slow_fn, 0.5)
        # Now submit the job we want to cancel
        job_id = store.submit(_fast_fn, 1)

        cancelled = store.cancel(job_id)
        assert cancelled is True
        s = store.status(job_id)
        assert s["status"] == "cancelled"

        # Clean up: wait for blocker
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if store.status(_blocker)["status"] != "running":
                break
            time.sleep(0.05)

    def test_cancel_running(self) -> None:
        """cancel() returns False when the job is already running."""
        store = JobStore(max_workers=2)
        job_id = store.submit(_slow_fn, 0.3)

        # Wait until it starts
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if store.status(job_id)["status"] == "running":
                break
            time.sleep(0.01)

        assert store.cancel(job_id) is False

        # Let it finish
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if store.status(job_id)["status"] == "done":
                break
            time.sleep(0.05)

    def test_lru_eviction(self) -> None:
        """After 21 completed jobs the oldest is evicted from the store."""
        # Use a single worker so jobs complete in submission order, making
        # eviction order deterministic.
        store = JobStore(max_workers=1, max_completed=20)
        ids: list[str] = []
        for i in range(21):
            ids.append(store.submit(_fast_fn, i))

        # Wait for all to finish or be evicted
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            alive = sum(1 for jid in ids if _job_exists(store, jid))
            if alive == 20:
                break
            time.sleep(0.05)

        # Exactly one job should have been evicted; 20 remain
        alive_ids = [jid for jid in ids if _job_exists(store, jid)]
        evicted_ids = [jid for jid in ids if not _job_exists(store, jid)]
        assert len(evicted_ids) == 1
        assert len(alive_ids) == 20
        # With a single worker the first submitted job completes first, so it
        # is the one evicted when the 21st job completes.
        assert evicted_ids[0] == ids[0]

    def test_failed_job(self) -> None:
        """A failing function sets status=failed; result() raises ValueError."""
        store = JobStore(max_workers=2)
        job_id = store.submit(_failing_fn)

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            s = store.status(job_id)
            if s["status"] in ("failed", "done", "cancelled"):
                break
            time.sleep(0.01)

        s = store.status(job_id)
        assert s["status"] == "failed"
        assert "intentional failure" in (s["error"] or "")

        with pytest.raises(ValueError, match="intentional failure"):
            store.result(job_id)

    def test_status_fields(self) -> None:
        """submitted_at, started_at, finished_at are all present after completion."""
        store = JobStore(max_workers=2)
        job_id = store.submit(_fast_fn)

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            s = store.status(job_id)
            if s["status"] == "done":
                break
            time.sleep(0.01)

        s = store.status(job_id)
        assert s["submitted_at"] is not None
        assert s["started_at"] is not None
        assert s["finished_at"] is not None
        assert s["error"] is None

    def test_result_before_done_raises(self) -> None:
        """result() raises ValueError when the job is still pending/running."""
        store = JobStore(max_workers=1)
        # Occupy the worker
        _blocker = store.submit(_slow_fn, 0.5)
        job_id = store.submit(_fast_fn)

        with pytest.raises(ValueError):
            store.result(job_id)

    def test_unknown_job_raises(self) -> None:
        """Accessing an unknown job_id raises KeyError."""
        store = JobStore()
        with pytest.raises(KeyError):
            store.status("nonexistent-id")
        with pytest.raises(KeyError):
            store.result("nonexistent-id")
        with pytest.raises(KeyError):
            store.cancel("nonexistent-id")
