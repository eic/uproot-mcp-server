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
        _blocker = store.submit(_slow_fn, 2.0)
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
        job_id = store.submit(_slow_fn, 1.0)

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
        store = JobStore(max_workers=4, max_completed=20)
        ids: list[str] = []
        for i in range(21):
            ids.append(store.submit(_fast_fn, i))

        # Wait for all to finish
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            statuses = [store.status(jid)["status"] for jid in ids if jid in store._jobs]
            if all(s == "done" for s in statuses):
                break
            time.sleep(0.05)

        # The first job should have been evicted
        with pytest.raises(KeyError):
            store.status(ids[0])

        # Jobs 1-20 should still be present
        for jid in ids[1:]:
            s = store.status(jid)
            assert s["status"] == "done"

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
        """result() raises KeyError when the job is still pending/running."""
        store = JobStore(max_workers=1)
        # Occupy the worker
        _blocker = store.submit(_slow_fn, 2.0)
        job_id = store.submit(_fast_fn)

        with pytest.raises(KeyError):
            store.result(job_id)

        # Clean up
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                if store.status(_blocker)["status"] == "done":
                    break
            except KeyError:
                break
            time.sleep(0.05)

    def test_unknown_job_raises(self) -> None:
        """Accessing an unknown job_id raises KeyError."""
        store = JobStore()
        with pytest.raises(KeyError):
            store.status("nonexistent-id")
        with pytest.raises(KeyError):
            store.result("nonexistent-id")
        with pytest.raises(KeyError):
            store.cancel("nonexistent-id")
