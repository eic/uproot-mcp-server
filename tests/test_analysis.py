"""Tests for uproot_mcp_server.analysis module.

These tests run against:
1. A local synthetic ROOT file in tests/fixtures/test_eic.root
2. (Optional) A remote XRootD file at dtn-eic.jlab.org if the
   UPROOT_TEST_REMOTE_FILE environment variable is set.

The local tests always run.  Remote tests are skipped when the environment
variable is absent.
"""

from __future__ import annotations

import math
import os
import pathlib

import pytest

from uproot_mcp_server import analysis

# ---------------------------------------------------------------------------
# Fixture paths
# ---------------------------------------------------------------------------

FIXTURE_DIR = pathlib.Path(__file__).parent / "fixtures"
LOCAL_FILE = str(FIXTURE_DIR / "test_eic.root")
NESTED_FILE = str(FIXTURE_DIR / "test_eic_nested.root")

# Remote file: set via environment variable (skipped if not set)
REMOTE_FILE = os.environ.get(
    "UPROOT_TEST_REMOTE_FILE",
    "",  # empty -> skip
)

# ---------------------------------------------------------------------------
# Expected field names for key return dicts
# ---------------------------------------------------------------------------

REQUIRED_TREE_INFO_FIELDS = (
    "file_path", "tree_name", "title", "num_entries", "num_branches", "branches",
)

REQUIRED_STATS_FIELDS = (
    "count", "mean", "std", "min", "max", "p25", "p50", "p75",
    "num_nan", "num_inf", "file_path", "tree_name", "branch_name",
)

REQUIRED_HISTOGRAM_FIELDS = (
    "edges", "counts", "underflow", "overflow", "entries",
    "mean", "std", "range_min", "range_max", "bins",
    "file_path", "tree_name", "branch_name",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_finite_or_none(v: object) -> bool:
    """Return True if *v* is None or a finite float/int."""
    if v is None:
        return True
    if isinstance(v, (int, float)):
        return math.isfinite(float(v))
    return False


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

_remote_only = pytest.mark.skipif(
    not REMOTE_FILE, reason="UPROOT_TEST_REMOTE_FILE not set"
)


# ===========================================================================
# get_file_structure
# ===========================================================================


class TestGetFileStructure:
    def test_returns_dict(self):
        result = analysis.get_file_structure(LOCAL_FILE)
        assert isinstance(result, dict)

    def test_file_path_echoed(self):
        result = analysis.get_file_structure(LOCAL_FILE)
        assert result["file_path"] == LOCAL_FILE

    def test_has_keys_and_trees(self):
        result = analysis.get_file_structure(LOCAL_FILE)
        assert "keys" in result
        assert "trees" in result
        assert isinstance(result["keys"], list)
        assert isinstance(result["trees"], list)

    def test_finds_events_tree(self):
        result = analysis.get_file_structure(LOCAL_FILE)
        names = [k["name"] for k in result["keys"]]
        assert "events" in names

    def test_tree_summary_fields(self):
        result = analysis.get_file_structure(LOCAL_FILE)
        trees = result["trees"]
        assert len(trees) >= 1
        tree = trees[0]
        assert "name" in tree
        assert "key_name" in tree
        assert "cycle" in tree
        assert "num_entries" in tree
        assert "num_branches" in tree
        assert "branches" in tree

    def test_tree_cycle_and_key_name(self):
        result = analysis.get_file_structure(LOCAL_FILE)
        tree = result["trees"][0]
        assert isinstance(tree["cycle"], int)
        assert tree["cycle"] >= 1
        assert ";" in tree["key_name"]
        assert tree["key_name"].startswith(tree["name"])

    def test_num_entries_positive(self):
        result = analysis.get_file_structure(LOCAL_FILE)
        assert result["trees"][0]["num_entries"] > 0

    def test_branch_list_not_empty(self):
        result = analysis.get_file_structure(LOCAL_FILE)
        assert result["trees"][0]["num_branches"] > 0

    def test_error_on_missing_file(self):
        # analysis.get_file_structure raises on missing file;
        # the server layer (server.py) wraps the exception into an error dict.
        with pytest.raises(Exception):
            analysis.get_file_structure("/nonexistent/path/file.root")

    def test_elapsed_s(self):
        result = analysis.get_file_structure(LOCAL_FILE)
        assert "elapsed_s" in result
        assert isinstance(result["elapsed_s"], float)
        assert result["elapsed_s"] >= 0.0

    @_remote_only
    def test_remote_file_structure(self):
        result = analysis.get_file_structure(REMOTE_FILE)
        assert isinstance(result, dict)
        assert "keys" in result
        assert len(result["keys"]) >= 1


# ===========================================================================
# get_tree_info
# ===========================================================================


class TestGetTreeInfo:
    def test_returns_dict(self):
        result = analysis.get_tree_info(LOCAL_FILE, "events")
        assert isinstance(result, dict)

    def test_required_fields(self):
        result = analysis.get_tree_info(LOCAL_FILE, "events")
        for field in REQUIRED_TREE_INFO_FIELDS:
            assert field in result, f"Missing field: {field}"

    def test_tree_name(self):
        result = analysis.get_tree_info(LOCAL_FILE, "events")
        assert result["tree_name"] == "events"

    def test_num_entries(self):
        result = analysis.get_tree_info(LOCAL_FILE, "events")
        assert result["num_entries"] == 1000

    def test_branches_list(self):
        result = analysis.get_tree_info(LOCAL_FILE, "events")
        branches = result["branches"]
        assert isinstance(branches, list)
        assert len(branches) == 5

    def test_branch_fields(self):
        result = analysis.get_tree_info(LOCAL_FILE, "events")
        for b in result["branches"]:
            assert "name" in b
            assert "typename" in b
            assert "num_entries" in b

    def test_known_branches_present(self):
        result = analysis.get_tree_info(LOCAL_FILE, "events")
        names = {b["name"] for b in result["branches"]}
        assert {"px", "py", "pz", "charge", "energy"}.issubset(names)

    def test_raises_on_missing_tree(self):
        with pytest.raises(Exception):
            analysis.get_tree_info(LOCAL_FILE, "no_such_tree")

    def test_elapsed_s(self):
        result = analysis.get_tree_info(LOCAL_FILE, "events")
        assert "elapsed_s" in result
        assert isinstance(result["elapsed_s"], float)
        assert result["elapsed_s"] >= 0.0

    @_remote_only
    def test_remote_tree_info(self):
        structure = analysis.get_file_structure(REMOTE_FILE)
        if not structure.get("trees"):
            pytest.skip("No TTrees found in remote file")
        tree_name = structure["trees"][0]["name"]
        result = analysis.get_tree_info(REMOTE_FILE, tree_name)
        assert result["num_entries"] >= 0


# ===========================================================================
# get_branch_statistics
# ===========================================================================


class TestGetBranchStatistics:
    def test_returns_dict(self):
        result = analysis.get_branch_statistics(LOCAL_FILE, "events", "px")
        assert isinstance(result, dict)

    def test_required_fields(self):
        result = analysis.get_branch_statistics(LOCAL_FILE, "events", "px")
        for field in REQUIRED_STATS_FIELDS:
            assert field in result, f"Missing field: {field}"

    def test_count_matches_entries(self):
        result = analysis.get_branch_statistics(LOCAL_FILE, "events", "px")
        assert result["count"] == 1000

    def test_finite_statistics(self):
        result = analysis.get_branch_statistics(LOCAL_FILE, "events", "px")
        for key in ("mean", "std", "min", "max", "p25", "p50", "p75"):
            assert _is_finite_or_none(result[key]), f"{key} is not finite: {result[key]}"

    def test_min_le_p25_le_p50_le_p75_le_max(self):
        result = analysis.get_branch_statistics(LOCAL_FILE, "events", "px")
        assert result["min"] <= result["p25"]
        assert result["p25"] <= result["p50"]
        assert result["p50"] <= result["p75"]
        assert result["p75"] <= result["max"]

    def test_std_non_negative(self):
        result = analysis.get_branch_statistics(LOCAL_FILE, "events", "px")
        assert result["std"] >= 0

    def test_no_nan_or_inf(self):
        result = analysis.get_branch_statistics(LOCAL_FILE, "events", "px")
        assert result["num_nan"] == 0
        assert result["num_inf"] == 0

    def test_with_selection_cut(self):
        result_all = analysis.get_branch_statistics(LOCAL_FILE, "events", "px")
        result_cut = analysis.get_branch_statistics(
            LOCAL_FILE, "events", "px", cut="charge != 0"
        )
        assert result_cut["count"] < result_all["count"]
        assert result_cut["count"] > 0

    def test_cut_echoed_in_result(self):
        cut = "charge != 0"
        result = analysis.get_branch_statistics(LOCAL_FILE, "events", "px", cut=cut)
        assert result["cut"] == cut

    def test_no_cut_is_none(self):
        result = analysis.get_branch_statistics(LOCAL_FILE, "events", "px")
        assert result["cut"] is None

    def test_entry_range(self):
        result_full = analysis.get_branch_statistics(LOCAL_FILE, "events", "px")
        result_half = analysis.get_branch_statistics(
            LOCAL_FILE, "events", "px", entry_start=0, entry_stop=500
        )
        assert result_half["count"] == 500
        assert result_full["count"] == 1000

    def test_integer_branch(self):
        result = analysis.get_branch_statistics(LOCAL_FILE, "events", "charge")
        assert result["count"] == 1000
        # charge is in {-1, 0, 1}
        assert result["min"] >= -1
        assert result["max"] <= 1

    def test_elapsed_s(self):
        result = analysis.get_branch_statistics(LOCAL_FILE, "events", "px")
        assert "elapsed_s" in result
        assert isinstance(result["elapsed_s"], float)
        assert result["elapsed_s"] >= 0.0

    @_remote_only
    def test_remote_branch_statistics(self):
        structure = analysis.get_file_structure(REMOTE_FILE)
        if not structure.get("trees"):
            pytest.skip("No TTrees found in remote file")
        tree_name = structure["trees"][0]["name"]
        branch_name = structure["trees"][0]["branches"][0]["name"]
        result = analysis.get_branch_statistics(REMOTE_FILE, tree_name, branch_name)
        assert isinstance(result, dict)
        assert "count" in result


# ===========================================================================
# histogram_branch
# ===========================================================================


class TestHistogramBranch:
    def test_returns_dict(self):
        result = analysis.histogram_branch(LOCAL_FILE, "events", "px")
        assert isinstance(result, dict)

    def test_required_fields(self):
        result = analysis.histogram_branch(LOCAL_FILE, "events", "px")
        for field in REQUIRED_HISTOGRAM_FIELDS:
            assert field in result, f"Missing field: {field}"

    def test_edge_count(self):
        result = analysis.histogram_branch(LOCAL_FILE, "events", "px", bins=20)
        assert len(result["edges"]) == 21  # bins + 1

    def test_count_length(self):
        result = analysis.histogram_branch(LOCAL_FILE, "events", "px", bins=20)
        assert len(result["counts"]) == 20

    def test_total_entries_conserved(self):
        result = analysis.histogram_branch(LOCAL_FILE, "events", "px", bins=100)
        total = sum(result["counts"]) + result["underflow"] + result["overflow"]
        assert total == result["entries"]

    def test_entries_match_tree(self):
        result = analysis.histogram_branch(LOCAL_FILE, "events", "px", bins=100)
        assert result["entries"] == 1000

    def test_edges_monotonically_increasing(self):
        result = analysis.histogram_branch(LOCAL_FILE, "events", "px", bins=50)
        edges = result["edges"]
        for i in range(len(edges) - 1):
            assert edges[i] < edges[i + 1]

    def test_explicit_range(self):
        result = analysis.histogram_branch(
            LOCAL_FILE, "events", "px", bins=50, range_min=-2.0, range_max=2.0
        )
        assert result["range_min"] == pytest.approx(-2.0)
        assert result["range_max"] == pytest.approx(2.0)
        assert result["underflow"] >= 0
        assert result["overflow"] >= 0

    def test_partial_range_raises(self):
        with pytest.raises(ValueError, match="range_min and range_max"):
            analysis.histogram_branch(LOCAL_FILE, "events", "px", range_min=0.0)

    def test_partial_range_max_only_raises(self):
        with pytest.raises(ValueError, match="range_min and range_max"):
            analysis.histogram_branch(LOCAL_FILE, "events", "px", range_max=1.0)

    def test_bins_must_be_positive(self):
        with pytest.raises(ValueError, match="bins"):
            analysis.histogram_branch(LOCAL_FILE, "events", "px", bins=0)

    def test_with_cut(self):
        result_all = analysis.histogram_branch(LOCAL_FILE, "events", "px", bins=50)
        result_cut = analysis.histogram_branch(
            LOCAL_FILE, "events", "px", bins=50, cut="charge != 0"
        )
        assert result_cut["entries"] < result_all["entries"]
        assert result_cut["entries"] > 0

    def test_cut_echoed(self):
        cut = "energy > 3"
        result = analysis.histogram_branch(LOCAL_FILE, "events", "px", cut=cut)
        assert result["cut"] == cut

    def test_mean_finite(self):
        result = analysis.histogram_branch(LOCAL_FILE, "events", "px")
        assert result["mean"] is None or math.isfinite(result["mean"])

    def test_std_non_negative(self):
        result = analysis.histogram_branch(LOCAL_FILE, "events", "px")
        assert result["std"] is None or result["std"] >= 0

    def test_entry_range(self):
        result = analysis.histogram_branch(
            LOCAL_FILE, "events", "px", bins=50, entry_start=0, entry_stop=500
        )
        assert result["entries"] == 500

    def test_elapsed_s(self):
        result = analysis.histogram_branch(LOCAL_FILE, "events", "px")
        assert "elapsed_s" in result
        assert isinstance(result["elapsed_s"], float)
        assert result["elapsed_s"] >= 0.0

    @_remote_only
    def test_remote_histogram(self):
        structure = analysis.get_file_structure(REMOTE_FILE)
        if not structure.get("trees"):
            pytest.skip("No TTrees found in remote file")
        tree_name = structure["trees"][0]["name"]
        branch_name = structure["trees"][0]["branches"][0]["name"]
        result = analysis.histogram_branch(
            REMOTE_FILE, tree_name, branch_name, bins=50
        )
        assert isinstance(result, dict)
        assert "counts" in result
        assert len(result["counts"]) == 50


# ---------------------------------------------------------------------------
# TestRunKernel
# ---------------------------------------------------------------------------

REQUIRED_KERNEL_ARRAY_FIELDS = (
    "result_type", "data", "total", "page", "page_size", "page_count",
    "has_more", "file_path", "tree_name", "branches",
)

REQUIRED_KERNEL_SCALAR_FIELDS = (
    "result_type", "data", "file_path", "tree_name", "branches",
)


class TestRunKernel:
    def test_single_branch_passthrough(self):
        code = "def kernel(events):\n    return events['px']\n"
        result = analysis.run_kernel(LOCAL_FILE, "events", code, ["px"])
        assert isinstance(result, dict)
        for field in REQUIRED_KERNEL_ARRAY_FIELDS:
            assert field in result, f"missing field: {field}"
        assert result["result_type"] == "array"
        assert result["total"] == 1000

    def test_multi_branch_derived_quantity(self):
        """Kernel computing magnitude of 3-momentum."""
        code = (
            "def kernel(events):\n"
            "    px = events['px']\n"
            "    py = events['py']\n"
            "    pz = events['pz']\n"
            "    return np.sqrt(px**2 + py**2 + pz**2)\n"
        )
        result = analysis.run_kernel(
            LOCAL_FILE, "events", code, ["px", "py", "pz"]
        )
        assert result["result_type"] == "array"
        assert result["total"] == 1000
        # All momenta magnitudes must be non-negative
        assert all(v >= 0 for v in result["data"])

    def test_scalar_return(self):
        code = "def kernel(events):\n    return float(np.mean(events['px']))\n"
        result = analysis.run_kernel(LOCAL_FILE, "events", code, ["px"])
        assert result["result_type"] == "scalar"
        assert isinstance(result["data"], float)

    def test_dict_return(self):
        code = (
            "def kernel(events):\n"
            "    return {'mean': float(np.mean(events['px'])), 'n': len(events['px'])}\n"
        )
        result = analysis.run_kernel(LOCAL_FILE, "events", code, ["px"])
        assert result["result_type"] == "dict"
        assert result["data"]["n"] == 1000

    def test_pagination_first_page(self):
        code = "def kernel(events):\n    return events['px']\n"
        result = analysis.run_kernel(
            LOCAL_FILE, "events", code, ["px"], page=0, page_size=100
        )
        assert result["page"] == 0
        assert result["page_size"] == 100
        assert len(result["data"]) == 100
        assert result["total"] == 1000
        assert result["page_count"] == 10
        assert result["has_more"] is True

    def test_pagination_last_page(self):
        code = "def kernel(events):\n    return events['px']\n"
        result = analysis.run_kernel(
            LOCAL_FILE, "events", code, ["px"], page=9, page_size=100
        )
        assert result["page"] == 9
        assert len(result["data"]) == 100
        assert result["has_more"] is False

    def test_pagination_out_of_range_raises(self):
        code = "def kernel(events):\n    return events['px']\n"
        with pytest.raises(ValueError, match="out of range"):
            analysis.run_kernel(
                LOCAL_FILE, "events", code, ["px"], page=99, page_size=100
            )

    def test_entry_range(self):
        code = "def kernel(events):\n    return events['px']\n"
        result = analysis.run_kernel(
            LOCAL_FILE, "events", code, ["px"],
            entry_start=0, entry_stop=200, page_size=500,
        )
        assert result["total"] == 200

    def test_invalid_branch_raises(self):
        code = "def kernel(events):\n    return events['nonexistent']\n"
        with pytest.raises(ValueError, match="not found"):
            analysis.run_kernel(
                LOCAL_FILE, "events", code, ["nonexistent"]
            )

    def test_invalid_kernel_raises(self):
        from uproot_mcp_server.sandbox import KernelError
        code = "def kernel(events):\n    import os\n    return os.getcwd()\n"
        with pytest.raises(KernelError):
            analysis.run_kernel(LOCAL_FILE, "events", code, ["px"])

    def test_json_serialisable(self):
        import json
        code = "def kernel(events):\n    return events['px']\n"
        result = analysis.run_kernel(LOCAL_FILE, "events", code, ["px"])
        json.dumps(result)  # must not raise

    def test_elapsed_s_and_kernel_elapsed_s(self):
        code = "def kernel(events):\n    return events['px']\n"
        result = analysis.run_kernel(LOCAL_FILE, "events", code, ["px"])
        assert "elapsed_s" in result
        assert "kernel_elapsed_s" in result
        assert isinstance(result["elapsed_s"], float)
        assert isinstance(result["kernel_elapsed_s"], float)
        assert result["elapsed_s"] >= result["kernel_elapsed_s"] >= 0.0


# ---------------------------------------------------------------------------
# TestGetDatasetFileList
# ---------------------------------------------------------------------------

class TestGetDatasetFileList:
    def test_returns_dict(self):
        result = analysis.get_dataset_file_list(
            str(FIXTURE_DIR / "*.root"), "events", workers=1
        )
        assert isinstance(result, dict)

    def test_finds_fixture_file(self):
        result = analysis.get_dataset_file_list(
            str(FIXTURE_DIR / "*.root"), "events", workers=1
        )
        assert result["n_files"] >= 1
        assert any("test_eic.root" in p for p in result["file_paths"])

    def test_missing_tree(self):
        result = analysis.get_dataset_file_list(
            str(FIXTURE_DIR / "*.root"), "nonexistent_tree", workers=1
        )
        assert result["n_files_missing_tree"] >= 1

    def test_elapsed_s(self):
        result = analysis.get_dataset_file_list(
            str(FIXTURE_DIR / "*.root"), "events", workers=1
        )
        assert "elapsed_s" in result
        assert result["elapsed_s"] >= 0.0

    def test_result_keys(self):
        result = analysis.get_dataset_file_list(
            str(FIXTURE_DIR / "*.root"), "events", workers=1
        )
        for key in ("path", "tree_name", "file_paths", "n_files",
                    "n_files_missing_tree", "missing_tree_files",
                    "n_files_failed", "failed_files", "elapsed_s"):
            assert key in result

    def test_directory_path(self):
        result = analysis.get_dataset_file_list(
            str(FIXTURE_DIR), "events", workers=1
        )
        assert result["n_files"] >= 1


# ---------------------------------------------------------------------------
# TestValidateDatasetSchema
# ---------------------------------------------------------------------------

class TestValidateDatasetSchema:
    def test_valid_schema(self):
        result = analysis.validate_dataset_schema(
            [LOCAL_FILE], "events", ["px", "py", "pz"], workers=1
        )
        assert isinstance(result, dict)
        assert result["n_files"] == 1
        assert result["n_files_ok"] == 1
        assert result["compatible"] is True
        assert result["total_entries"] > 0

    def test_missing_branch(self):
        result = analysis.validate_dataset_schema(
            [LOCAL_FILE], "events", ["px", "nonexistent_branch_xyz"], workers=1
        )
        assert result["compatible"] is False
        assert "nonexistent_branch_xyz" in result["missing_branch_files"]

    def test_bad_tree(self):
        result = analysis.validate_dataset_schema(
            [LOCAL_FILE], "nonexistent_tree", ["px"], workers=1
        )
        assert result["n_files_failed"] == 1
        assert result["compatible"] is False

    def test_elapsed_s(self):
        result = analysis.validate_dataset_schema(
            [LOCAL_FILE], "events", ["px"], workers=1
        )
        assert "elapsed_s" in result
        assert result["elapsed_s"] >= 0.0

    def test_result_keys(self):
        result = analysis.validate_dataset_schema(
            [LOCAL_FILE], "events", ["px"], workers=1
        )
        for key in ("compatible", "n_files", "n_files_ok", "n_files_failed",
                    "total_entries", "missing_branch_files", "failed_files",
                    "elapsed_s"):
            assert key in result

    def test_json_serialisable(self):
        import json
        result = analysis.validate_dataset_schema(
            [LOCAL_FILE], "events", ["px", "py"], workers=1
        )
        json.dumps(result)  # must not raise


# ---------------------------------------------------------------------------
# Branch name resolution (podio/edm4eic regression)
# ---------------------------------------------------------------------------


class TestNestedBranchNameResolution:
    """Regression: branch validation must use ``name in tree`` rather than
    ``name in set(tree.keys())``.

    On podio/edm4eic files, ``tree.keys()`` reports full paths like
    ``Particles/Particles.momentum.x`` so the dotted leaf name (which uproot
    *can* resolve via ``tree[name]``) is absent from that set.  The synthetic
    fixture reproduces the same mismatch with a record-typed branch where
    leaf-only names like ``"x"`` are resolvable but missing from default keys.
    """

    def test_precondition_keys_mismatch(self):
        """Sanity check: without this, the tests below could silently pass
        for the wrong reason if uproot's behavior changes."""
        import uproot
        with uproot.open(NESTED_FILE) as f:
            tree = f["events"]
            assert "x" not in set(tree.keys())
            assert "x" in tree

    def test_run_kernel_does_not_reject_unlisted_leaf_name(self):
        """Pre-fix, ``run_kernel`` short-circuited with
        ``ValueError("Branch 'x' not found in tree 'events'")`` because the
        membership check used ``set(tree.keys())``.  Downstream array
        unpacking on this synthetic fixture still fails (``tree.arrays(['x'])``
        returns a record nested under ``hits``), but that is unrelated; the
        regression-specific assertion is that the *not-found* error is gone.
        """
        code = "def kernel(events):\n    return events['x']\n"
        try:
            analysis.run_kernel(NESTED_FILE, "events", code, ["x"])
        except ValueError as exc:
            assert "not found" not in str(exc), (
                f"Pre-fix not-found error has resurfaced: {exc!r}"
            )
        except Exception:
            pass  # any other downstream failure is fine for this regression

    def test_validate_dataset_schema_accepts_unlisted_leaf_name(self):
        result = analysis.validate_dataset_schema(
            [NESTED_FILE], "events", ["x", "y"], workers=1
        )
        assert result["compatible"] is True
        assert result["n_files_ok"] == 1
        assert result["missing_branch_files"] == {}


# ===========================================================================
# histogram_dataset
# ===========================================================================

REQUIRED_DATASET_HISTOGRAM_FIELDS = (
    "edges", "counts", "underflow", "overflow", "entries",
    "mean", "std", "range_min", "range_max", "bins",
    "file_paths", "tree_name", "branch_name", "cut",
    "n_files", "n_files_ok", "n_files_failed", "failed_files", "elapsed_s",
)


class TestHistogramDataset:
    def test_returns_dict(self):
        result = analysis.histogram_dataset(
            [LOCAL_FILE], "events", "px", range_min=-5.0, range_max=5.0
        )
        assert isinstance(result, dict)

    def test_required_fields(self):
        result = analysis.histogram_dataset(
            [LOCAL_FILE], "events", "px", range_min=-5.0, range_max=5.0
        )
        for field in REQUIRED_DATASET_HISTOGRAM_FIELDS:
            assert field in result, f"Missing field: {field}"

    def test_edge_count(self):
        result = analysis.histogram_dataset(
            [LOCAL_FILE], "events", "px", bins=20, range_min=-5.0, range_max=5.0
        )
        assert len(result["edges"]) == 21

    def test_count_length(self):
        result = analysis.histogram_dataset(
            [LOCAL_FILE], "events", "px", bins=20, range_min=-5.0, range_max=5.0
        )
        assert len(result["counts"]) == 20

    def test_entries_match_single_file(self):
        result = analysis.histogram_dataset(
            [LOCAL_FILE], "events", "px", bins=50, range_min=-5.0, range_max=5.0
        )
        assert result["entries"] == 1000

    def test_counts_additive_two_copies(self, tmp_path):
        """histogram_dataset([f, f]) counts == 2 * histogram_branch(f) for same range."""
        import shutil
        copy = str(tmp_path / "copy.root")
        shutil.copy(LOCAL_FILE, copy)
        single = analysis.histogram_branch(
            LOCAL_FILE, "events", "px", bins=50, range_min=-5.0, range_max=5.0
        )
        double = analysis.histogram_dataset(
            [LOCAL_FILE, copy], "events", "px", bins=50, range_min=-5.0, range_max=5.0
        )
        assert double["entries"] == 2 * single["entries"]
        for i, (c_d, c_s) in enumerate(zip(double["counts"], single["counts"])):
            assert c_d == 2 * c_s, f"bin {i}: {c_d} != 2 * {c_s}"

    def test_n_files_ok_all_good(self):
        result = analysis.histogram_dataset(
            [LOCAL_FILE], "events", "px", range_min=-5.0, range_max=5.0
        )
        assert result["n_files"] == 1
        assert result["n_files_ok"] == 1
        assert result["n_files_failed"] == 0
        assert result["failed_files"] == []

    def test_failed_file_tracked(self):
        result = analysis.histogram_dataset(
            [LOCAL_FILE, "/no/such/file.root"],
            "events", "px", range_min=-5.0, range_max=5.0,
        )
        assert result["n_files"] == 2
        assert result["n_files_failed"] == 1
        assert len(result["failed_files"]) == 1
        assert result["failed_files"][0]["file"] == "/no/such/file.root"
        assert isinstance(result["failed_files"][0]["error"], str)

    def test_elapsed_s_non_negative(self):
        result = analysis.histogram_dataset(
            [LOCAL_FILE], "events", "px", range_min=-5.0, range_max=5.0
        )
        assert result["elapsed_s"] >= 0.0

    def test_range_echoed(self):
        result = analysis.histogram_dataset(
            [LOCAL_FILE], "events", "px", range_min=-3.0, range_max=3.0
        )
        assert result["range_min"] == pytest.approx(-3.0)
        assert result["range_max"] == pytest.approx(3.0)

    def test_bins_must_be_positive(self):
        with pytest.raises(ValueError, match="bins"):
            analysis.histogram_dataset(
                [LOCAL_FILE], "events", "px", bins=0, range_min=-5.0, range_max=5.0
            )

    def test_invalid_range_raises(self):
        with pytest.raises(ValueError):
            analysis.histogram_dataset(
                [LOCAL_FILE], "events", "px", range_min=5.0, range_max=-5.0
            )

    def test_entries_per_file_limits_count(self):
        result = analysis.histogram_dataset(
            [LOCAL_FILE], "events", "px",
            range_min=-5.0, range_max=5.0, entries_per_file=200,
        )
        assert result["entries"] <= 200

    def test_json_serialisable(self):
        import json
        result = analysis.histogram_dataset(
            [LOCAL_FILE], "events", "px", range_min=-5.0, range_max=5.0
        )
        json.dumps(result)

    def test_mean_finite_or_none(self):
        result = analysis.histogram_dataset(
            [LOCAL_FILE], "events", "px", range_min=-5.0, range_max=5.0
        )
        assert result["mean"] is None or math.isfinite(result["mean"])

    def test_std_non_negative(self):
        result = analysis.histogram_dataset(
            [LOCAL_FILE], "events", "px", range_min=-5.0, range_max=5.0
        )
        assert result["std"] is None or result["std"] >= 0.0

    def test_cut_echoed(self):
        cut = "charge != 0"
        result = analysis.histogram_dataset(
            [LOCAL_FILE], "events", "px", range_min=-5.0, range_max=5.0, cut=cut
        )
        assert result["cut"] == cut


# ===========================================================================
# get_dataset_statistics
# ===========================================================================

REQUIRED_DATASET_STATS_FIELDS = (
    "count", "mean", "std", "min", "max", "p25", "p50", "p75",
    "num_nan", "num_inf",
    "file_paths", "tree_name", "branch_name", "cut",
    "n_files", "n_files_ok", "n_files_failed", "failed_files", "elapsed_s",
)


class TestGetDatasetStatistics:
    def test_returns_dict(self):
        result = analysis.get_dataset_statistics([LOCAL_FILE], "events", "px")
        assert isinstance(result, dict)

    def test_required_fields(self):
        result = analysis.get_dataset_statistics([LOCAL_FILE], "events", "px")
        for field in REQUIRED_DATASET_STATS_FIELDS:
            assert field in result, f"Missing field: {field}"

    def test_count_matches_single_file(self):
        result = analysis.get_dataset_statistics([LOCAL_FILE], "events", "px")
        assert result["count"] == 1000

    def test_percentiles_are_none(self):
        result = analysis.get_dataset_statistics([LOCAL_FILE], "events", "px")
        assert result["p25"] is None
        assert result["p50"] is None
        assert result["p75"] is None

    def test_mean_matches_single_file(self):
        single = analysis.get_branch_statistics(LOCAL_FILE, "events", "px")
        dataset = analysis.get_dataset_statistics([LOCAL_FILE], "events", "px")
        assert dataset["mean"] == pytest.approx(single["mean"], rel=1e-6)

    def test_std_matches_single_file(self):
        single = analysis.get_branch_statistics(LOCAL_FILE, "events", "px")
        dataset = analysis.get_dataset_statistics([LOCAL_FILE], "events", "px")
        assert dataset["std"] == pytest.approx(single["std"], rel=1e-6)

    def test_two_copies_same_mean_std(self, tmp_path):
        """Statistics on [f, f] should give same mean/std as on f alone."""
        import shutil
        copy = str(tmp_path / "copy.root")
        shutil.copy(LOCAL_FILE, copy)
        single = analysis.get_dataset_statistics([LOCAL_FILE], "events", "px")
        double = analysis.get_dataset_statistics([LOCAL_FILE, copy], "events", "px")
        assert double["count"] == 2 * single["count"]
        assert double["mean"] == pytest.approx(single["mean"], rel=1e-6)
        assert double["std"] == pytest.approx(single["std"], rel=1e-6)

    def test_n_files_ok_all_good(self):
        result = analysis.get_dataset_statistics([LOCAL_FILE], "events", "px")
        assert result["n_files"] == 1
        assert result["n_files_ok"] == 1
        assert result["n_files_failed"] == 0
        assert result["failed_files"] == []

    def test_failed_file_tracked(self):
        result = analysis.get_dataset_statistics(
            [LOCAL_FILE, "/no/such/file.root"], "events", "px"
        )
        assert result["n_files"] == 2
        assert result["n_files_failed"] == 1
        assert len(result["failed_files"]) == 1

    def test_elapsed_s_non_negative(self):
        result = analysis.get_dataset_statistics([LOCAL_FILE], "events", "px")
        assert result["elapsed_s"] >= 0.0

    def test_std_non_negative(self):
        result = analysis.get_dataset_statistics([LOCAL_FILE], "events", "px")
        assert result["std"] is None or result["std"] >= 0.0

    def test_entries_per_file_limits_count(self):
        result = analysis.get_dataset_statistics(
            [LOCAL_FILE], "events", "px", entries_per_file=200
        )
        assert result["count"] <= 200

    def test_cut_echoed(self):
        cut = "charge != 0"
        result = analysis.get_dataset_statistics(
            [LOCAL_FILE], "events", "px", cut=cut
        )
        assert result["cut"] == cut

    def test_json_serialisable(self):
        import json
        result = analysis.get_dataset_statistics([LOCAL_FILE], "events", "px")
        json.dumps(result)

    def test_finite_statistics(self):
        result = analysis.get_dataset_statistics([LOCAL_FILE], "events", "px")
        for key in ("mean", "std", "min", "max"):
            assert _is_finite_or_none(result[key]), f"{key} not finite: {result[key]}"

# ---------------------------------------------------------------------------
# run_kernel_dataset
# ---------------------------------------------------------------------------

MOMENTUM_KERNEL = (
    "def kernel(events):\n"
    "    px = events['px']\n"
    "    py = events['py']\n"
    "    pz = events['pz']\n"
    "    return np.sqrt(px**2 + py**2 + pz**2)\n"
)

SCALAR_KERNEL = "def kernel(events):\n    return float(np.mean(events['px']))\n"

HISTOGRAM_KERNEL = (
    "def kernel(events):\n"
    "    counts, _ = np.histogram(events['px'], bins=100, range=(-5.0, 5.0))\n"
    "    return {'edges': np.linspace(-5.0, 5.0, 101).tolist(), 'counts': counts.tolist()}\n"
)


class TestRunKernelDataset:
    def test_array_result_two_files(self):
        """Using the same file twice should give 2× total entries."""
        result = analysis.run_kernel_dataset(
            [LOCAL_FILE, LOCAL_FILE], "events", MOMENTUM_KERNEL,
            ["px", "py", "pz"],
        )
        assert result["result_type"] == "array"
        assert result["total"] == 2000  # 1000 entries × 2 files
        assert result["n_files_ok"] == 2
        assert result["n_files_failed"] == 0

    def test_scalar_result_per_file_list(self):
        result = analysis.run_kernel_dataset(
            [LOCAL_FILE, LOCAL_FILE], "events", SCALAR_KERNEL, ["px"],
        )
        assert result["result_type"] == "scalar"
        assert isinstance(result["data"], list)
        assert len(result["data"]) == 2
        assert "sum" in result
        assert "mean" in result
        assert isinstance(result["mean"], float)

    def test_histogram_dict_auto_reduce(self):
        result = analysis.run_kernel_dataset(
            [LOCAL_FILE, LOCAL_FILE], "events", HISTOGRAM_KERNEL, ["px"],
        )
        assert result["result_type"] == "dict"
        data = result["data"]
        assert "edges" in data
        assert "counts" in data
        assert len(data["edges"]) == 101
        assert len(data["counts"]) == 100
        # Summed over two identical files: counts should be 2× single file
        single = analysis.run_kernel(LOCAL_FILE, "events", HISTOGRAM_KERNEL, ["px"])
        single_counts = single["data"]["counts"]
        for got, expected in zip(data["counts"], single_counts):
            assert got == 2 * expected

    def test_entries_per_file_limits_results(self):
        result = analysis.run_kernel_dataset(
            [LOCAL_FILE, LOCAL_FILE], "events", MOMENTUM_KERNEL,
            ["px", "py", "pz"], entries_per_file=10,
        )
        assert result["result_type"] == "array"
        assert result["total"] == 20  # 10 entries × 2 files

    def test_failed_file_recorded_and_continues(self):
        result = analysis.run_kernel_dataset(
            [LOCAL_FILE, "/nonexistent/file.root", LOCAL_FILE],
            "events", MOMENTUM_KERNEL, ["px", "py", "pz"],
        )
        assert result["n_files_failed"] == 1
        assert "/nonexistent/file.root" in result["failed_files"]
        assert result["n_files_ok"] == 2
        assert result["result_type"] == "array"

    def test_elapsed_s_non_negative(self):
        result = analysis.run_kernel_dataset(
            [LOCAL_FILE], "events", SCALAR_KERNEL, ["px"],
        )
        assert result["elapsed_s"] >= 0.0
        assert isinstance(result["elapsed_s"], float)
        assert len(result["per_file_elapsed_s"]) == 1
        assert result["per_file_elapsed_s"][0] >= 0.0

    def test_all_files_fail_returns_empty(self):
        result = analysis.run_kernel_dataset(
            ["/no/file.root"], "events", SCALAR_KERNEL, ["px"],
        )
        assert result["result_type"] == "empty"
        assert result["n_files_failed"] == 1

    def test_reduce_code_applied(self):
        """reduce_code is applied as a left fold: scalar + scalar = scalar sum."""
        reduce_code = "def reduce(a, b):\n    return a + b\n"
        single = analysis.run_kernel(LOCAL_FILE, "events", SCALAR_KERNEL, ["px"])
        result = analysis.run_kernel_dataset(
            [LOCAL_FILE, LOCAL_FILE], "events", SCALAR_KERNEL,
            ["px"], reduce_code=reduce_code,
        )
        # Sum of two identical per-file means = 2 × single mean
        assert result["result_type"] == "scalar"
        assert isinstance(result["data"], float)
        assert abs(result["data"] - 2 * single["data"]) < 1e-6

    def test_json_serialisable(self):
        import json as _json
        result = analysis.run_kernel_dataset(
            [LOCAL_FILE, LOCAL_FILE], "events", SCALAR_KERNEL, ["px"],
        )
        _json.dumps(result)  # must not raise


# ---------------------------------------------------------------------------
# estimate_dataset_cost
# ---------------------------------------------------------------------------


class TestEstimateDatasetCost:
    def test_returns_expected_keys(self):
        result = analysis.estimate_dataset_cost(
            [LOCAL_FILE, LOCAL_FILE, LOCAL_FILE],
            "events", SCALAR_KERNEL, ["px"],
        )
        for key in (
            "n_files", "total_entries", "sample_files_used",
            "entries_per_second", "estimated_total_seconds",
            "recommended_prototype_entries_per_file",
            "sample_elapsed_s", "elapsed_s",
        ):
            assert key in result, f"missing key: {key}"

    def test_estimated_total_seconds_positive(self):
        result = analysis.estimate_dataset_cost(
            [LOCAL_FILE, LOCAL_FILE, LOCAL_FILE],
            "events", SCALAR_KERNEL, ["px"],
        )
        assert result["estimated_total_seconds"] > 0

    def test_entries_per_second_positive(self):
        result = analysis.estimate_dataset_cost(
            [LOCAL_FILE, LOCAL_FILE, LOCAL_FILE],
            "events", SCALAR_KERNEL, ["px"],
        )
        assert result["entries_per_second"] > 0

    def test_elapsed_s_non_negative(self):
        result = analysis.estimate_dataset_cost(
            [LOCAL_FILE, LOCAL_FILE, LOCAL_FILE],
            "events", SCALAR_KERNEL, ["px"],
        )
        assert result["elapsed_s"] >= 0.0
        assert result["sample_elapsed_s"] >= 0.0

    def test_n_files_and_total_entries(self):
        result = analysis.estimate_dataset_cost(
            [LOCAL_FILE, LOCAL_FILE],
            "events", SCALAR_KERNEL, ["px"],
        )
        assert result["n_files"] == 2
        assert result["total_entries"] == 2000  # 1000 × 2

    def test_sample_files_capped_at_n_files(self):
        result = analysis.estimate_dataset_cost(
            [LOCAL_FILE],
            "events", SCALAR_KERNEL, ["px"],
            sample_files=10,
        )
        assert result["sample_files_used"] == 1
