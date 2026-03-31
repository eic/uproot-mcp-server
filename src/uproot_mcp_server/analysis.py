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
    """
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

        return {
            "file_path": file_path,
            "keys": keys,
            "trees": trees,
        }


def get_tree_info(file_path: str, tree_name: str) -> TreeInfo:
    """Return detailed metadata for a single TTree.

    Returns a dict with:
    - ``file_path``, ``tree_name``
    - ``num_entries``: total number of entries
    - ``num_branches``: number of top-level branches
    - ``branches``: list of branch dicts (name, typename, num_entries,
      uncompressed_bytes, compressed_bytes, compression_ratio, leaves)
    - ``title``: tree title if available
    """
    with _open_file(file_path) as f:
        tree = f[tree_name]

        title = ""
        try:
            title = tree.title
        except Exception:
            pass

        branches = [_branch_info(tree[b]) for b in tree.keys()]

        return {
            "file_path": file_path,
            "tree_name": tree.name,
            "title": title,
            "num_entries": int(tree.num_entries),
            "num_branches": len(branches),
            "branches": branches,
        }


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
    """
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
            return {
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

        percentiles = np.percentile(finite, [25, 50, 75])

        def _safe(v: Any) -> Any:
            if isinstance(v, float) and not math.isfinite(v):
                return None
            return float(v)

        return {
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
            return {
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

        return {
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

    result = _execute_kernel(code_obj, branches_data)

    meta: dict[str, Any] = {
        "file_path": file_path,
        "tree_name": tree_name,
        "branches": branches,
        "cut": cut,
        "entry_start": entry_start,
        "entry_stop": entry_stop,
        "page": page,
        "page_size": page_size,
    }

    if isinstance(result, ak.Array):
        return _paginate(result, page, page_size, "array", meta)
    if isinstance(result, np.ndarray):
        return _paginate(result.ravel(), page, page_size, "array", meta)
    if isinstance(result, (list, tuple)):
        return _paginate(result, page, page_size, "array", meta)
    if isinstance(result, dict):
        return {"result_type": "dict", "data": _normalize_json(result), **meta}
    # Scalar: convert numpy scalars to plain Python types
    scalar: Any = _normalize_json(result)
    return {"result_type": "scalar", "data": scalar, **meta}


# ---------------------------------------------------------------------------
# Dataset kernel execution helpers
# ---------------------------------------------------------------------------


def _is_histogram_dict(result: Any) -> bool:
    """Return True if *result* looks like a histogram dict ``{"edges", "counts"}``."""
    if not isinstance(result, dict):
        return False
    if "edges" not in result or "counts" not in result:
        return False
    counts = result["counts"]
    edges = result["edges"]
    if not hasattr(counts, "__len__") or not hasattr(edges, "__len__"):
        return False
    if len(edges) != len(counts) + 1:
        return False
    # Verify counts contains numbers
    if isinstance(counts, np.ndarray):
        return True
    if isinstance(counts, (list, tuple)) and counts:
        return isinstance(counts[0], (int, float))
    return False


def run_kernel_dataset(
    file_paths: list[str],
    tree_name: str,
    kernel_code: str,
    branches: list[str],
    *,
    reduce_code: str | None = None,
    cut: str | None = None,
    entries_per_file: int | None = None,
    workers: int = 4,
    page: int = 0,
    page_size: int = 1000,
) -> KernelResult:
    """Run a sandboxed kernel over multiple files and auto-reduce the results.

    The kernel is compiled once, then executed per-file in a RestrictedPython
    subprocess.  Partial results are accumulated and reduced automatically
    based on the result type of the first successful file.

    Parameters
    ----------
    file_paths:
        List of local paths or XRootD URLs to ROOT files.
    tree_name:
        Name of the TTree in each file.
    kernel_code:
        Python source defining ``def kernel(events): ...``.
    branches:
        Branch names to load and pass into ``events``.
    reduce_code:
        Optional Python source defining ``def reduce(a, b): ...``.
        Applied as a left fold over partial results.  If omitted, an
        automatic reduction strategy is chosen based on the result type.
    cut:
        Optional boolean selection expression applied per file.
    entries_per_file:
        Maximum entries to read per file (``None`` reads all).
    workers:
        Reserved for future parallel execution (currently unused).
    page:
        0-indexed page for array results (default 0).
    page_size:
        Elements per page for array results (default 1000).

    Returns
    -------
    dict
        Contains ``result_type``, ``data``, and dataset-level metadata:
        ``n_files``, ``n_files_ok``, ``n_files_failed``, ``failed_files``,
        ``per_file_elapsed_s``, ``elapsed_s``.
        Array results additionally include pagination fields.
        Scalar results include ``sum`` and ``mean`` over per-file values.
    """
    if page < 0:
        raise ValueError(f"page must be >= 0, got {page}")
    if page_size < 1:
        raise ValueError(f"page_size must be >= 1, got {page_size}")

    from uproot_mcp_server.sandbox import compile_kernel  # noqa: PLC0415
    from uproot_mcp_server.sandbox import execute_kernel as _execute_kernel  # noqa: PLC0415

    # Compile once — catches errors before touching any files
    code_obj = compile_kernel(kernel_code)

    reduce_code_obj = None
    if reduce_code is not None:
        reduce_code_obj = compile_kernel(reduce_code)

    t_total_start = time.perf_counter()
    partial_results: list[Any] = []
    per_file_elapsed_s: list[float] = []
    failed_files: list[str] = []

    for fp in file_paths:
        t_file = time.perf_counter()
        try:
            with _open_file(fp) as f:
                tree = f[tree_name]
                read_kwargs: dict[str, Any] = {}
                if entries_per_file is not None:
                    read_kwargs["entry_stop"] = entries_per_file
                arrays_kwargs: dict[str, Any] = dict(read_kwargs)
                if cut:
                    arrays_kwargs["cut"] = cut
                arrays_ak = tree.arrays(branches, library="ak", **arrays_kwargs)
                branches_data: dict[str, Any] = {b: arrays_ak[b] for b in branches}
            result = _execute_kernel(code_obj, branches_data)
            partial_results.append(result)
        except Exception:  # noqa: BLE001
            failed_files.append(fp)
        per_file_elapsed_s.append(time.perf_counter() - t_file)

    elapsed_s = time.perf_counter() - t_total_start

    meta: dict[str, Any] = {
        "file_paths": file_paths,
        "tree_name": tree_name,
        "branches": branches,
        "cut": cut,
        "entries_per_file": entries_per_file,
        "n_files": len(file_paths),
        "n_files_ok": len(partial_results),
        "n_files_failed": len(failed_files),
        "failed_files": failed_files,
        "per_file_elapsed_s": per_file_elapsed_s,
        "elapsed_s": elapsed_s,
    }

    if not partial_results:
        return {"result_type": "empty", "data": None, **meta}

    # --- Custom reduce_code: left fold ---
    if reduce_code_obj is not None:
        from uproot_mcp_server.sandbox import execute_reduce as _execute_reduce  # noqa: PLC0415
        reduced: Any = partial_results[0]
        for r in partial_results[1:]:
            reduced = _execute_reduce(reduce_code_obj, reduced, r)
        # Format the single reduced result
        if isinstance(reduced, (ak.Array, np.ndarray)):
            flat = (
                ak.to_numpy(ak.flatten(reduced, axis=None)).astype(float)
                if isinstance(reduced, ak.Array)
                else reduced.ravel().astype(float)
            )
            return _paginate(flat, page, page_size, "array", meta)
        if isinstance(reduced, (list, tuple)):
            return _paginate(list(reduced), page, page_size, "array", meta)
        if isinstance(reduced, dict):
            return {"result_type": "dict", "data": _normalize_json(reduced), **meta}
        return {"result_type": "scalar", "data": _normalize_json(reduced), **meta}

    # --- Auto-reduce based on first result type ---
    first = partial_results[0]

    if isinstance(first, (ak.Array, np.ndarray)):
        arrays: list[np.ndarray] = []
        for r in partial_results:
            if isinstance(r, ak.Array):
                arrays.append(
                    ak.to_numpy(ak.flatten(r, axis=None)).astype(float)
                )
            elif isinstance(r, np.ndarray):
                arrays.append(r.ravel().astype(float))
            else:
                arrays.append(np.asarray(r, dtype=float).ravel())
        combined = np.concatenate(arrays)
        return _paginate(combined, page, page_size, "array", meta)

    if isinstance(first, (list, tuple)):
        combined_list: list[Any] = []
        for r in partial_results:
            combined_list.extend(r)
        return _paginate(combined_list, page, page_size, "array", meta)

    if isinstance(first, dict) and _is_histogram_dict(first):
        edges = (
            first["edges"].tolist()
            if isinstance(first["edges"], np.ndarray)
            else list(first["edges"])
        )
        total_counts = np.zeros(len(first["counts"]), dtype=np.int64)
        for r in partial_results:
            c = r["counts"]
            total_counts += np.asarray(c, dtype=np.int64)
        return {
            "result_type": "dict",
            "data": {"edges": edges, "counts": total_counts.tolist()},
            **meta,
        }

    if isinstance(first, dict):
        return {
            "result_type": "dict",
            "data": [_normalize_json(r) for r in partial_results],
            **meta,
        }

    # Scalar (int/float/numpy scalar)
    scalars = [_normalize_json(r) for r in partial_results]
    finite_scalars = [s for s in scalars if isinstance(s, (int, float))]
    scalar_sum: Any = sum(finite_scalars) if finite_scalars else None
    scalar_mean: Any = scalar_sum / len(finite_scalars) if finite_scalars else None
    return {
        "result_type": "scalar",
        "data": scalars,
        "sum": scalar_sum,
        "mean": scalar_mean,
        **meta,
    }


def estimate_dataset_cost(
    file_paths: list[str],
    tree_name: str,
    kernel_code: str,
    branches: list[str],
    *,
    sample_files: int = 3,
    entries_per_file: int = 1000,
) -> dict[str, Any]:
    """Measure kernel performance on a sample and extrapolate to the full dataset.

    Opens each file metadata-only to count total entries, then times the kernel
    on a small sample to produce a cost estimate.

    Parameters
    ----------
    file_paths:
        List of local paths or XRootD URLs to ROOT files.
    tree_name:
        Name of the TTree in each file.
    kernel_code:
        Python source defining ``def kernel(events): ...``.
    branches:
        Branch names required by the kernel.
    sample_files:
        Number of files to sample (default 3).
    entries_per_file:
        Maximum entries per sample file (default 1000).

    Returns
    -------
    dict with keys:

    - ``n_files``: total number of files in the dataset
    - ``total_entries``: sum of all ``num_entries`` across files
    - ``sample_files_used``: number of files actually sampled
    - ``entries_per_second``: throughput measured on the sample
    - ``estimated_total_seconds``: extrapolated wall time for the full dataset
    - ``recommended_prototype_entries_per_file``: entries per file that keeps a
      full-dataset run under 30 s
    - ``sample_elapsed_s``: wall time for all sample kernel executions
    - ``elapsed_s``: total wall time including metadata reads
    """
    t_start = time.perf_counter()

    # Step 1: metadata-only total entry count
    total_entries = 0
    for fp in file_paths:
        with _open_file(fp) as f:
            total_entries += int(f[tree_name].num_entries)

    n_files = len(file_paths)
    n_sample = min(sample_files, n_files)
    sample_file_paths = file_paths[:n_sample]

    # Step 2: time kernel on sample files
    sample_entries = 0
    t_sample_start = time.perf_counter()
    for fp in sample_file_paths:
        with _open_file(fp) as f:
            actual = int(f[tree_name].num_entries)
        entries_this_file = min(entries_per_file, actual)
        sample_entries += entries_this_file
        run_kernel(fp, tree_name, kernel_code, branches, entry_stop=entries_per_file)
    sample_elapsed_s = time.perf_counter() - t_sample_start

    elapsed_s = time.perf_counter() - t_start

    entries_per_second = (
        sample_entries / sample_elapsed_s if sample_elapsed_s > 0 else float("inf")
    )
    estimated_total_seconds = (
        total_entries / entries_per_second if entries_per_second > 0 else float("inf")
    )
    recommended = (
        max(1, int(30 * entries_per_second / n_files))
        if n_files > 0 and math.isfinite(entries_per_second)
        else 1
    )

    return {
        "n_files": n_files,
        "total_entries": total_entries,
        "sample_files_used": n_sample,
        "entries_per_second": entries_per_second,
        "estimated_total_seconds": estimated_total_seconds,
        "recommended_prototype_entries_per_file": recommended,
        "sample_elapsed_s": sample_elapsed_s,
        "elapsed_s": elapsed_s,
    }
