"""Core ROOT file analysis logic using uproot.

This module provides functions for reading ROOT files (local or via XRootD),
querying their structure, computing branch statistics, and producing histograms.
All public functions return plain Python dicts suitable for JSON serialization.
"""

from __future__ import annotations

import math
import time
from typing import Any

import awkward as ak
import numpy as np
import uproot


# ---------------------------------------------------------------------------
# Type aliases for return values
# ---------------------------------------------------------------------------

FileStructure = dict[str, Any]
BranchStatistics = dict[str, Any]
HistogramResult = dict[str, Any]
KernelResult = dict[str, Any]
TreeInfo = dict[str, Any]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _flatten_to_float(data: Any) -> np.ndarray:
    """Flatten any uproot-returned array to a 1-D float64 numpy array.

    Handles both regular numpy arrays and jagged/variable-length Awkward
    arrays by using ``ak.flatten`` before converting to numpy.
    """
    if isinstance(data, ak.Array):
        return ak.to_numpy(ak.flatten(data, axis=None)).astype(float)
    arr = np.asarray(data)
    if arr.dtype == object:
        # Object array likely contains sub-arrays (jagged); flatten via awkward
        return ak.to_numpy(ak.flatten(ak.Array(arr), axis=None)).astype(float)
    return arr.ravel().astype(float)


def _open_file(file_path: str) -> uproot.ReadOnlyFile:
    """Open a ROOT file at *file_path*.

    *file_path* may be a local path or an XRootD URL
    (``root://server//path/to/file.root``).
    """
    return uproot.open(file_path)


def _branch_info(branch: Any, *, include_leaves: bool = True) -> dict[str, Any]:
    """Return a summary dict for a single branch."""
    info: dict[str, Any] = {
        "name": branch.name,
        "typename": branch.typename if hasattr(branch, "typename") else str(type(branch)),
        "num_entries": int(branch.num_entries) if hasattr(branch, "num_entries") else None,
        "compression_ratio": None,
    }

    # Compression info is stored on TBranch objects
    try:
        tot = int(branch.member("fTotBytes"))
        zip_ = int(branch.member("fZipBytes"))
        info["uncompressed_bytes"] = tot
        info["compressed_bytes"] = zip_
        info["compression_ratio"] = round(tot / zip_, 3) if zip_ > 0 else None
    except Exception:
        pass

    if include_leaves:
        leaves = []
        try:
            for leaf in branch.leaves:
                leaf_info: dict[str, Any] = {
                    "name": leaf.name,
                    "typename": leaf.typename if hasattr(leaf, "typename") else str(type(leaf)),
                }
                leaves.append(leaf_info)
        except Exception:
            pass
        info["leaves"] = leaves

    return info


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_file_structure(file_path: str) -> FileStructure:
    """Return the high-level structure of a ROOT file.

    Returns a dict with:
    - ``file_path``: the path as given
    - ``keys``: list of top-level key dicts (name, classname, cycle)
    - ``trees``: list of tree summary dicts for each TTree found at the top level
    - ``elapsed_s``: wall-clock seconds for the operation
    """
    t0 = time.perf_counter()
    with _open_file(file_path) as f:
        keys: list[dict[str, Any]] = []
        trees: list[dict[str, Any]] = []

        for key_name, key_class in f.classnames().items():
            # key_name includes cycle, e.g. "events;1"
            base_name = key_name.split(";")[0]
            cycle_str = key_name.split(";")[1] if ";" in key_name else "1"
            keys.append(
                {
                    "name": base_name,
                    "classname": key_class,
                    "cycle": int(cycle_str),
                }
            )

            if key_class in ("TTree", "TNtuple", "TNtupleD"):
                try:
                    tree = f[key_name]
                    trees.append(
                        {
                            "name": base_name,
                            "key_name": key_name,
                            "cycle": int(cycle_str),
                            "classname": key_class,
                            "num_entries": int(tree.num_entries),
                            "num_branches": len(tree.keys()),
                            "branches": [
                                _branch_info(tree[b], include_leaves=False)
                                for b in tree.keys()
                            ],
                        }
                    )
                except Exception as exc:
                    trees.append(
                        {
                            "name": base_name,
                            "key_name": key_name,
                            "cycle": int(cycle_str),
                            "classname": key_class,
                            "error": str(exc),
                        }
                    )

        result: FileStructure = {
            "file_path": file_path,
            "keys": keys,
            "trees": trees,
        }
    result["elapsed_s"] = round(time.perf_counter() - t0, 6)
    return result


def get_tree_info(file_path: str, tree_name: str) -> TreeInfo:
    """Return detailed metadata for a single TTree.

    Returns a dict with:
    - ``file_path``, ``tree_name``
    - ``num_entries``: total number of entries
    - ``num_branches``: number of top-level branches
    - ``branches``: list of branch dicts (name, typename, num_entries,
      uncompressed_bytes, compressed_bytes, compression_ratio, leaves)
    - ``title``: tree title if available
    - ``elapsed_s``: wall-clock seconds for the operation
    """
    t0 = time.perf_counter()
    with _open_file(file_path) as f:
        tree = f[tree_name]

        title = ""
        try:
            title = tree.title
        except Exception:
            pass

        branches = [_branch_info(tree[b]) for b in tree.keys()]

        result: TreeInfo = {
            "file_path": file_path,
            "tree_name": tree.name,
            "title": title,
            "num_entries": int(tree.num_entries),
            "num_branches": len(branches),
            "branches": branches,
        }
    result["elapsed_s"] = round(time.perf_counter() - t0, 6)
    return result


def get_branch_statistics(
    file_path: str,
    tree_name: str,
    branch_name: str,
    *,
    cut: str | None = None,
    entry_start: int | None = None,
    entry_stop: int | None = None,
) -> BranchStatistics:
    """Compute summary statistics for a single branch.

    Parameters
    ----------
    file_path:
        Path or XRootD URL to the ROOT file.
    tree_name:
        Name of the TTree (optionally with cycle, e.g. ``"events;1"``).
    branch_name:
        Fully-qualified branch name (e.g. ``"MCParticles.momentum.x"``).
    cut:
        Optional boolean expression string applied entry-wise (NumPy style
        referencing other branches in the same tree).  Only entries where the
        expression evaluates to True are included in the statistics.
    entry_start, entry_stop:
        Slice the tree to a sub-range of entries.

    Returns a dict with:
    - ``count``, ``mean``, ``std``, ``min``, ``max``, ``p25``, ``p50``, ``p75``
    - ``num_nan``, ``num_inf``  (counts of non-finite values)
    - metadata: ``file_path``, ``tree_name``, ``branch_name``, ``cut``
    - ``elapsed_s``: wall-clock seconds for the operation
    """
    t0 = time.perf_counter()
    with _open_file(file_path) as f:
        tree = f[tree_name]

        read_kwargs: dict[str, Any] = {}
        if entry_start is not None:
            read_kwargs["entry_start"] = entry_start
        if entry_stop is not None:
            read_kwargs["entry_stop"] = entry_stop

        if cut:
            # Read the branch of interest and all branches needed for the cut
            arrays = tree.arrays(
                [branch_name],
                cut=cut,
                library="ak",
                **read_kwargs,
            )
            data_raw: Any = arrays[branch_name]
        else:
            data_raw = tree[branch_name].array(library="ak", **read_kwargs)

        # Flatten to 1-D float64 (handles both fixed-size and jagged branches)
        data = _flatten_to_float(data_raw)

        num_nan = int(np.sum(np.isnan(data)))
        num_inf = int(np.sum(np.isinf(data)))
        finite = data[np.isfinite(data)]

        if len(finite) == 0:
            result: BranchStatistics = {
                "file_path": file_path,
                "tree_name": tree_name,
                "branch_name": branch_name,
                "cut": cut,
                "count": 0,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "p25": None,
                "p50": None,
                "p75": None,
                "num_nan": num_nan,
                "num_inf": num_inf,
            }
            result["elapsed_s"] = round(time.perf_counter() - t0, 6)
            return result

        percentiles = np.percentile(finite, [25, 50, 75])

        def _safe(v: Any) -> Any:
            if isinstance(v, float) and not math.isfinite(v):
                return None
            return float(v)

        result = {
            "file_path": file_path,
            "tree_name": tree_name,
            "branch_name": branch_name,
            "cut": cut,
            "count": int(len(finite)),
            "mean": _safe(float(np.mean(finite))),
            "std": _safe(float(np.std(finite))),
            "min": _safe(float(np.min(finite))),
            "max": _safe(float(np.max(finite))),
            "p25": _safe(float(percentiles[0])),
            "p50": _safe(float(percentiles[1])),
            "p75": _safe(float(percentiles[2])),
            "num_nan": num_nan,
            "num_inf": num_inf,
        }
        result["elapsed_s"] = round(time.perf_counter() - t0, 6)
        return result


def histogram_branch(
    file_path: str,
    tree_name: str,
    branch_name: str,
    *,
    bins: int = 100,
    range_min: float | None = None,
    range_max: float | None = None,
    cut: str | None = None,
    entry_start: int | None = None,
    entry_stop: int | None = None,
) -> HistogramResult:
    """Histogram a branch, optionally applying a selection cut.

    Parameters
    ----------
    file_path:
        Path or XRootD URL to the ROOT file.
    tree_name:
        Name of the TTree.
    branch_name:
        Branch to histogram.
    bins:
        Number of bins (default 100).
    range_min, range_max:
        Explicit histogram range.  If either is ``None`` the range is
        determined from the data (both must be given or both omitted).
    cut:
        Optional boolean selection expression.
    entry_start, entry_stop:
        Slice the tree to a sub-range of entries.

    Returns a dict with:
    - ``edges``: list of bin-edge values (length ``bins + 1``)
    - ``counts``: list of bin counts (length ``bins``)
    - ``underflow``, ``overflow``: entries outside the range
    - ``entries``, ``mean``, ``std``
    - metadata: ``file_path``, ``tree_name``, ``branch_name``, ``cut``,
      ``bins``, ``range_min``, ``range_max``
    """
    if bins < 1:
        raise ValueError(f"bins must be >= 1, got {bins}")

    has_min = range_min is not None
    has_max = range_max is not None
    if has_min != has_max:
        raise ValueError(
            "range_min and range_max must both be provided or both omitted "
            f"(got range_min={range_min}, range_max={range_max})"
        )

    t0 = time.perf_counter()
    with _open_file(file_path) as f:
        tree = f[tree_name]

        read_kwargs: dict[str, Any] = {}
        if entry_start is not None:
            read_kwargs["entry_start"] = entry_start
        if entry_stop is not None:
            read_kwargs["entry_stop"] = entry_stop

        if cut:
            arrays = tree.arrays(
                [branch_name],
                cut=cut,
                library="ak",
                **read_kwargs,
            )
            data_raw: Any = arrays[branch_name]
        else:
            data_raw = tree[branch_name].array(library="ak", **read_kwargs)

        # Flatten to 1-D float64 (handles both fixed-size and jagged branches)
        data = _flatten_to_float(data_raw)
        finite = data[np.isfinite(data)]

        if len(finite) == 0:
            # Return empty histogram
            if has_min and has_max:
                edges_arr = np.linspace(range_min, range_max, bins + 1)  # type: ignore[arg-type]
            else:
                edges_arr = np.linspace(0.0, 1.0, bins + 1)
            result: HistogramResult = {
                "file_path": file_path,
                "tree_name": tree_name,
                "branch_name": branch_name,
                "cut": cut,
                "bins": bins,
                "range_min": float(edges_arr[0]),
                "range_max": float(edges_arr[-1]),
                "edges": edges_arr.tolist(),
                "counts": [0] * bins,
                "underflow": 0,
                "overflow": 0,
                "entries": 0,
                "mean": None,
                "std": None,
            }
            result["elapsed_s"] = round(time.perf_counter() - t0, 6)
            return result

        if has_min and has_max:
            hist_range = (float(range_min), float(range_max))  # type: ignore[arg-type]
        else:
            hist_range = (float(np.min(finite)), float(np.max(finite)))
            # Protect against degenerate range (all values identical)
            if hist_range[0] == hist_range[1]:
                hist_range = (hist_range[0] - 0.5, hist_range[1] + 0.5)

        counts_arr, edges_arr = np.histogram(finite, bins=bins, range=hist_range)

        underflow = int(np.sum(finite < hist_range[0]))
        overflow = int(np.sum(finite > hist_range[1]))

        mean = float(np.mean(finite))
        std = float(np.std(finite))

        result = {
            "file_path": file_path,
            "tree_name": tree_name,
            "branch_name": branch_name,
            "cut": cut,
            "bins": bins,
            "range_min": hist_range[0],
            "range_max": hist_range[1],
            "edges": edges_arr.tolist(),
            "counts": counts_arr.tolist(),
            "underflow": underflow,
            "overflow": overflow,
            "entries": int(len(finite)),
            "mean": mean if math.isfinite(mean) else None,
            "std": std if math.isfinite(std) else None,
        }
    result["elapsed_s"] = round(time.perf_counter() - t0, 6)
    return result


# ---------------------------------------------------------------------------
# Kernel execution
# ---------------------------------------------------------------------------


def _normalize_json(v: Any) -> Any:
    """Recursively convert *v* to a JSON-serialisable Python type.

    Handles ``numpy`` scalars and arrays, ``awkward`` arrays, non-finite floats
    (mapped to ``None``), and nested ``dict`` / ``list`` / ``tuple`` containers.
    """
    if isinstance(v, dict):
        return {k: _normalize_json(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_normalize_json(item) for item in v]
    if isinstance(v, np.generic):
        val = v.item()
        if isinstance(val, float) and not math.isfinite(val):
            return None
        return val
    if isinstance(v, np.ndarray):
        return _normalize_json(v.tolist())
    if isinstance(v, ak.Array):
        return _normalize_json(ak.to_list(v))
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def _paginate(
    data: Any,
    page: int,
    page_size: int,
    result_type: str,
    meta: dict[str, Any],
) -> KernelResult:
    """Slice *data* to the requested page and return a paginated result dict.

    *data* may be an ``ak.Array``, ``np.ndarray``, ``list``, or ``tuple``.
    Only the requested page slice is materialised to a Python list, avoiding
    the overhead of converting the entire result before slicing.
    """
    total = len(data)
    page_count = max(1, math.ceil(total / page_size))

    if page >= page_count and total > 0:
        raise ValueError(
            f"page {page} is out of range "
            f"(total pages: {page_count}, total items: {total})"
        )

    start = page * page_size
    end = start + page_size
    page_slice = data[start:end]

    if isinstance(page_slice, ak.Array):
        page_data: list[Any] = ak.to_list(page_slice)
    elif isinstance(page_slice, np.ndarray):
        page_data = page_slice.tolist()
    else:
        page_data = list(page_slice)

    return {
        "result_type": result_type,
        "data": page_data,
        "total": total,
        "page": page,
        "page_size": page_size,
        "page_count": page_count,
        "has_more": end < total,
        **meta,
    }


def run_kernel(
    file_path: str,
    tree_name: str,
    kernel_code: str,
    branches: list[str],
    *,
    cut: str | None = None,
    entry_start: int | None = None,
    entry_stop: int | None = None,
    page: int = 0,
    page_size: int = 1000,
) -> KernelResult:
    """Load branches from a ROOT file and execute a sandboxed kernel.

    The kernel is a Python function ``def kernel(events): ...`` where
    ``events`` is a ``dict[str, array]`` keyed by the names in *branches*.
    Only ``np`` (numpy) and ``ak`` (awkward) are available inside the kernel;
    imports, file I/O, and dangerous builtins are blocked.

    Parameters
    ----------
    file_path:
        Path or XRootD URL to the ROOT file.
    tree_name:
        Name of the TTree.
    kernel_code:
        Python source defining ``def kernel(events): ...``.
    branches:
        Branch names to load and inject into ``events``.
    cut:
        Optional boolean selection expression applied before loading.
    entry_start, entry_stop:
        Slice the tree to a sub-range of entries.
    page:
        0-indexed page number for array-like results (default 0).
    page_size:
        Elements per page for array-like results (default 1000).

    Returns
    -------
    dict
        Always contains ``result_type`` (``"array"``, ``"scalar"``, or
        ``"dict"``), ``data``, and request metadata.  Array results also
        include ``total``, ``page``, ``page_size``, ``page_count``,
        ``has_more``.
    """
    if page < 0:
        raise ValueError(f"page must be >= 0, got {page}")
    if page_size < 1:
        raise ValueError(f"page_size must be >= 1, got {page_size}")

    t0 = time.perf_counter()

    # Compile first — fast, catches errors before opening the (potentially
    # remote) file.
    from uproot_mcp_server.sandbox import compile_kernel  # noqa: PLC0415
    from uproot_mcp_server.sandbox import execute_kernel as _execute_kernel  # noqa: PLC0415

    code_obj = compile_kernel(kernel_code)

    with _open_file(file_path) as f:
        tree = f[tree_name]

        available = set(tree.keys())
        for b in branches:
            if b not in available:
                raise ValueError(
                    f"Branch '{b}' not found in tree '{tree_name}'"
                )

        read_kwargs: dict[str, Any] = {}
        if entry_start is not None:
            read_kwargs["entry_start"] = entry_start
        if entry_stop is not None:
            read_kwargs["entry_stop"] = entry_stop

        arrays_kwargs: dict[str, Any] = dict(read_kwargs)
        if cut:
            arrays_kwargs["cut"] = cut

        arrays_ak = tree.arrays(branches, library="ak", **arrays_kwargs)
        branches_data: dict[str, Any] = {b: arrays_ak[b] for b in branches}

    t1 = time.perf_counter()
    result = _execute_kernel(code_obj, branches_data)
    t2 = time.perf_counter()

    kernel_elapsed_s = round(t2 - t1, 6)

    meta: dict[str, Any] = {
        "file_path": file_path,
        "tree_name": tree_name,
        "branches": branches,
        "cut": cut,
        "entry_start": entry_start,
        "entry_stop": entry_stop,
        "page": page,
        "page_size": page_size,
        "kernel_elapsed_s": kernel_elapsed_s,
    }

    if isinstance(result, ak.Array):
        out = _paginate(result, page, page_size, "array", meta)
    elif isinstance(result, np.ndarray):
        out = _paginate(result.ravel(), page, page_size, "array", meta)
    elif isinstance(result, (list, tuple)):
        out = _paginate(result, page, page_size, "array", meta)
    elif isinstance(result, dict):
        out = {"result_type": "dict", "data": _normalize_json(result), **meta}
    else:
        # Scalar: convert numpy scalars to plain Python types
        scalar: Any = _normalize_json(result)
        out = {"result_type": "scalar", "data": scalar, **meta}

    out["elapsed_s"] = round(time.perf_counter() - t0, 6)
    return out
