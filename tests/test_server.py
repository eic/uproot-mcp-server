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

    def test_no_inf_or_nan_in_output(self):
        result = server.histogram_branch(LOCAL_FILE, "events", "px", bins=100)
        json_str = json.dumps(result)
        assert "Infinity" not in json_str
        assert "NaN" not in json_str


# ---------------------------------------------------------------------------
# TestServerGetDatasetFileList
# ---------------------------------------------------------------------------

class TestServerGetDatasetFileList:
    def test_returns_dict(self):
        result = server.get_dataset_file_list(
            str(FIXTURE_DIR / "*.root"), "events", workers=1
        )
        assert isinstance(result, dict)

    def test_json_serialisable(self):
        result = server.get_dataset_file_list(
            str(FIXTURE_DIR / "*.root"), "events", workers=1
        )
        json.dumps(result)  # must not raise

    def test_finds_fixture_file(self):
        result = server.get_dataset_file_list(
            str(FIXTURE_DIR / "*.root"), "events", workers=1
        )
        assert result["n_files"] >= 1

    def test_error_on_xrootd_without_pyxrootd(self):
        # Without pyxrootd installed, should return an error dict
        import sys
        # Only test if XRootD is not available
        if "XRootD" not in sys.modules:
            result = server.get_dataset_file_list(
                "root://nonexistent//path/*.root", "events", workers=1
            )
            assert "error" in result


# ---------------------------------------------------------------------------
# TestServerValidateDatasetSchema
# ---------------------------------------------------------------------------

class TestServerValidateDatasetSchema:
    def test_returns_dict(self):
        result = server.validate_dataset_schema(
            [LOCAL_FILE], "events", ["px", "py"], workers=1
        )
        assert isinstance(result, dict)

    def test_json_serialisable(self):
        result = server.validate_dataset_schema(
            [LOCAL_FILE], "events", ["px", "py"], workers=1
        )
        json.dumps(result)  # must not raise

    def test_compatible(self):
        result = server.validate_dataset_schema(
            [LOCAL_FILE], "events", ["px", "py", "pz"], workers=1
        )
        assert result["compatible"] is True

    def test_missing_branch(self):
        result = server.validate_dataset_schema(
            [LOCAL_FILE], "events", ["px", "nonexistent_xyz"], workers=1
        )
        assert result["compatible"] is False
        assert "nonexistent_xyz" in result["missing_branch_files"]

    def test_bad_tree(self):
        result = server.validate_dataset_schema(
            [LOCAL_FILE], "nonexistent_tree", ["px"], workers=1
        )
        assert result["n_files_failed"] == 1
        assert result["compatible"] is False
