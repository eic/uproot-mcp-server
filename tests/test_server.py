"""Integration tests for the MCP server tool handlers.

These tests call the server tool functions directly (bypassing MCP transport)
to verify that:
- Tool functions return JSON-serialisable dicts
- Error cases are handled gracefully (no unhandled exceptions; error key present)
- Results are consistent with the underlying analysis module
"""

from __future__ import annotations

import json
import pathlib

import pytest

from uproot_mcp_server import server

FIXTURE_DIR = pathlib.Path(__file__).parent / "fixtures"
LOCAL_FILE = str(FIXTURE_DIR / "test_eic.root")


def _is_json_serialisable(obj: object) -> bool:
    try:
        json.dumps(obj)
        return True
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# get_file_structure
# ---------------------------------------------------------------------------


class TestServerGetFileStructure:
    def test_returns_json_serialisable(self):
        result = server.get_file_structure(LOCAL_FILE)
        assert _is_json_serialisable(result)

    def test_valid_file(self):
        result = server.get_file_structure(LOCAL_FILE)
        assert "error" not in result
        assert "keys" in result
        assert "trees" in result

    def test_invalid_file_returns_error_key(self):
        result = server.get_file_structure("/no/such/file.root")
        assert "error" in result
        assert isinstance(result["error"], str)

    def test_elapsed_s(self):
        result = server.get_file_structure(LOCAL_FILE)
        assert "elapsed_s" in result
        assert isinstance(result["elapsed_s"], float)
        assert result["elapsed_s"] >= 0.0


# ---------------------------------------------------------------------------
# get_tree_info
# ---------------------------------------------------------------------------


class TestServerGetTreeInfo:
    def test_returns_json_serialisable(self):
        result = server.get_tree_info(LOCAL_FILE, "events")
        assert _is_json_serialisable(result)

    def test_valid_tree(self):
        result = server.get_tree_info(LOCAL_FILE, "events")
        assert "error" not in result
        assert result["num_entries"] == 1000

    def test_invalid_tree_returns_error_key(self):
        result = server.get_tree_info(LOCAL_FILE, "no_such_tree")
        assert "error" in result


# ---------------------------------------------------------------------------
# get_branch_statistics
# ---------------------------------------------------------------------------


class TestServerGetBranchStatistics:
    def test_returns_json_serialisable(self):
        result = server.get_branch_statistics(LOCAL_FILE, "events", "px")
        assert _is_json_serialisable(result)

    def test_valid_branch(self):
        result = server.get_branch_statistics(LOCAL_FILE, "events", "px")
        assert "error" not in result
        assert result["count"] == 1000

    def test_with_cut(self):
        result = server.get_branch_statistics(
            LOCAL_FILE, "events", "px", cut="charge != 0"
        )
        assert "error" not in result
        assert 0 < result["count"] < 1000

    def test_invalid_branch_returns_error_key(self):
        result = server.get_branch_statistics(LOCAL_FILE, "events", "no_branch")
        assert "error" in result

    def test_elapsed_s(self):
        result = server.get_branch_statistics(LOCAL_FILE, "events", "px")
        assert "elapsed_s" in result
        assert isinstance(result["elapsed_s"], float)
        assert result["elapsed_s"] >= 0.0

    def test_no_inf_or_nan_in_output(self):
        result = server.get_branch_statistics(LOCAL_FILE, "events", "px")
        json_str = json.dumps(result)
        assert "Infinity" not in json_str
        assert "NaN" not in json_str


# ---------------------------------------------------------------------------
# histogram_branch
# ---------------------------------------------------------------------------


class TestServerHistogramBranch:
    def test_returns_json_serialisable(self):
        result = server.histogram_branch(LOCAL_FILE, "events", "px")
        assert _is_json_serialisable(result)

    def test_valid_branch(self):
        result = server.histogram_branch(LOCAL_FILE, "events", "px", bins=50)
        assert "error" not in result
        assert len(result["counts"]) == 50
        assert len(result["edges"]) == 51

    def test_with_cut(self):
        result = server.histogram_branch(
            LOCAL_FILE, "events", "px", bins=50, cut="charge != 0"
        )
        assert "error" not in result
        assert 0 < result["entries"] < 1000

    def test_invalid_branch_returns_error_key(self):
        result = server.histogram_branch(LOCAL_FILE, "events", "no_branch")
        assert "error" in result

    def test_invalid_range_returns_error_key(self):
        # Only one of range_min/range_max provided → ValueError in analysis
        result = server.histogram_branch(
            LOCAL_FILE, "events", "px", range_min=0.0
        )
        assert "error" in result

    def test_entries_conserved(self):
        result = server.histogram_branch(LOCAL_FILE, "events", "px", bins=100)
        total = sum(result["counts"]) + result["underflow"] + result["overflow"]
        assert total == result["entries"]

    def test_elapsed_s(self):
        result = server.histogram_branch(LOCAL_FILE, "events", "px")
        assert "elapsed_s" in result
        assert isinstance(result["elapsed_s"], float)
        assert result["elapsed_s"] >= 0.0

    def test_no_inf_or_nan_in_output(self):
        result = server.histogram_branch(LOCAL_FILE, "events", "px", bins=100)
        json_str = json.dumps(result)
        assert "Infinity" not in json_str
        assert "NaN" not in json_str
