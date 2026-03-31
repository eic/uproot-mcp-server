#!/usr/bin/env python3
"""uproot MCP server.

Exposes uproot-based ROOT file analysis tools over the Model Context Protocol
(MCP).  Supports local files and XRootD URLs (``root://server//path``).

Run with::

    python -m uproot_mcp_server.server        # stdio transport (default)
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from uproot_mcp_server import analysis

# ---------------------------------------------------------------------------
# Server setup
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="uproot-mcp-server",
    instructions=(
        "An MCP server for analysing ROOT files with uproot. "
        "Supports local paths and XRootD URLs (root://server//path). "
        "All tools return JSON-serialisable dictionaries."
    ),
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _json_safe(obj: Any) -> Any:
    """Recursively convert non-JSON-serialisable values to safe equivalents."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        import math
        if math.isnan(obj) or math.isinf(obj):
            return None
    return obj


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def get_file_structure(file_path: str) -> dict[str, Any]:
    """Return the top-level structure of a ROOT file.

    Lists all keys stored in the file and provides a summary of every TTree
    (number of entries, branches).  Useful as a first step when exploring an
    unfamiliar file.

    Parameters
    ----------
    file_path:
        Local filesystem path **or** XRootD URL, e.g.
        ``root://dtn-eic.jlab.org//path/to/file.root``.

    Returns
    -------
    dict with keys:

    - ``file_path``: echoed input path
    - ``keys``: list of dicts with ``name``, ``classname``, ``cycle``
    - ``trees``: list of TTree summary dicts containing ``name``,
      ``num_entries``, ``num_branches``, and a ``branches`` list with
      basic per-branch metadata
    """
    try:
        result = analysis.get_file_structure(file_path)
        return _json_safe(result)
    except Exception as exc:
        return {"error": str(exc), "file_path": file_path}


@mcp.tool()
def get_tree_info(file_path: str, tree_name: str) -> dict[str, Any]:
    """Return detailed metadata for a single TTree including all branches and leaves.

    Parameters
    ----------
    file_path:
        Local path or XRootD URL to the ROOT file.
    tree_name:
        Name of the TTree to inspect (e.g. ``"events"`` or ``"events;1"``).

    Returns
    -------
    dict with keys:

    - ``file_path``, ``tree_name``, ``title``
    - ``num_entries``: total number of entries in the tree
    - ``num_branches``: number of top-level branches
    - ``branches``: list of branch dicts, each containing:
        - ``name``, ``typename``
        - ``num_entries``
        - ``uncompressed_bytes``, ``compressed_bytes``, ``compression_ratio``
        - ``leaves``: list of leaf dicts (``name``, ``typename``)
    """
    try:
        result = analysis.get_tree_info(file_path, tree_name)
        return _json_safe(result)
    except Exception as exc:
        return {"error": str(exc), "file_path": file_path, "tree_name": tree_name}


@mcp.tool()
def get_branch_statistics(
    file_path: str,
    tree_name: str,
    branch_name: str,
    cut: str | None = None,
    entry_start: int | None = None,
    entry_stop: int | None = None,
) -> dict[str, Any]:
    """Compute summary statistics for a single branch.

    The branch values are read into a flat array (variable-length branches
    are flattened) and summary statistics are computed over finite values.

    Parameters
    ----------
    file_path:
        Local path or XRootD URL to the ROOT file.
    tree_name:
        Name of the TTree.
    branch_name:
        Fully-qualified branch name (e.g. ``"MCParticles.momentum.x"``).
    cut:
        Optional selection expression evaluated per entry
        (e.g. ``"MCParticles.charge != 0"``).
        Only entries where the expression is True are included.
    entry_start, entry_stop:
        Integer indices to slice the tree (Python-style).

    Returns
    -------
    dict with keys:

    - ``count``: number of finite values
    - ``mean``, ``std``, ``min``, ``max``
    - ``p25``, ``p50``, ``p75``: 25th, 50th, 75th percentiles
    - ``num_nan``, ``num_inf``: counts of non-finite values
    - ``file_path``, ``tree_name``, ``branch_name``, ``cut``
    """
    try:
        result = analysis.get_branch_statistics(
            file_path,
            tree_name,
            branch_name,
            cut=cut,
            entry_start=entry_start,
            entry_stop=entry_stop,
        )
        return _json_safe(result)
    except Exception as exc:
        return {
            "error": str(exc),
            "file_path": file_path,
            "tree_name": tree_name,
            "branch_name": branch_name,
        }


@mcp.tool()
def histogram_branch(
    file_path: str,
    tree_name: str,
    branch_name: str,
    bins: int = 100,
    range_min: float | None = None,
    range_max: float | None = None,
    cut: str | None = None,
    entry_start: int | None = None,
    entry_stop: int | None = None,
) -> dict[str, Any]:
    """Histogram a branch with an optional selection cut.

    Produces a 1-D histogram of the requested branch values.  The result can
    be used directly for plotting or further analysis by the client.

    Parameters
    ----------
    file_path:
        Local path or XRootD URL to the ROOT file.
    tree_name:
        Name of the TTree.
    branch_name:
        Branch to histogram (e.g. ``"MCParticles.momentum.x"``).
    bins:
        Number of histogram bins (default 100, must be >= 1).
    range_min, range_max:
        Explicit histogram range.  Both must be given together or both omitted
        (auto-range from data).
    cut:
        Optional boolean selection expression
        (e.g. ``"MCParticles.charge != 0"``).
    entry_start, entry_stop:
        Integer indices to slice the tree.

    Returns
    -------
    dict with keys:

    - ``edges``: list of ``bins + 1`` bin-edge values
    - ``counts``: list of ``bins`` bin counts
    - ``underflow``, ``overflow``: entries outside the histogram range
    - ``entries``: total finite entries histogrammed
    - ``mean``, ``std``: statistics of histogrammed values
    - ``range_min``, ``range_max``: actual histogram range used
    - ``bins``, ``file_path``, ``tree_name``, ``branch_name``, ``cut``
    """
    try:
        result = analysis.histogram_branch(
            file_path,
            tree_name,
            branch_name,
            bins=bins,
            range_min=range_min,
            range_max=range_max,
            cut=cut,
            entry_start=entry_start,
            entry_stop=entry_stop,
        )
        return _json_safe(result)
    except Exception as exc:
        return {
            "error": str(exc),
            "file_path": file_path,
            "tree_name": tree_name,
            "branch_name": branch_name,
        }


@mcp.tool()
def execute_kernel(
    file_path: str,
    tree_name: str,
    kernel_code: str,
    branches: list[str],
    cut: str | None = None,
    entry_start: int | None = None,
    entry_stop: int | None = None,
    page: int = 0,
    page_size: int = 1000,
) -> dict[str, Any]:
    """Execute a sandboxed Python kernel over ROOT file branch data.

    The kernel is a Python function that receives a dict of branch arrays and
    returns a computed result.  Imports, file I/O, ``exec``, ``eval``, and
    dangerous builtins are all blocked.  Only ``np`` (numpy) and ``ak``
    (awkward-array) are available as named packages.

    Parameters
    ----------
    file_path:
        Local path or XRootD URL to the ROOT file.
    tree_name:
        Name of the TTree.
    kernel_code:
        Python source defining ``def kernel(events): ...``.
        ``events`` is a ``dict[str, array]`` keyed by the names in
        *branches*.  Example::

            def kernel(events):
                px = events["ReconstructedParticles.momentum.x"]
                py = events["ReconstructedParticles.momentum.y"]
                pz = events["ReconstructedParticles.momentum.z"]
                return np.sqrt(px**2 + py**2 + pz**2)

    branches:
        List of branch names to load and pass into ``events``.
    cut:
        Optional boolean selection expression (same semantics as other tools).
    entry_start, entry_stop:
        Integer indices to slice the tree (Python-style).
    page:
        0-indexed page number for array-like results (default 0).
    page_size:
        Number of elements per page for array-like results (default 1000).

    Returns
    -------
    dict with keys:

    - ``result_type``: ``"array"``, ``"scalar"``, or ``"dict"``
    - ``data``: the result (a page slice for array results)
    - For array results: ``total``, ``page``, ``page_size``, ``page_count``,
      ``has_more``
    - Metadata: ``file_path``, ``tree_name``, ``branches``, ``cut``,
      ``entry_start``, ``entry_stop``
    """
    try:
        result = analysis.run_kernel(
            file_path,
            tree_name,
            kernel_code,
            branches,
            cut=cut,
            entry_start=entry_start,
            entry_stop=entry_stop,
            page=page,
            page_size=page_size,
        )
        return _json_safe(result)
    except Exception as exc:
        return {
            "error": str(exc),
            "file_path": file_path,
            "tree_name": tree_name,
            "branches": branches,
        }


@mcp.tool()
def execute_kernel_dataset(
    file_paths: list[str],
    tree_name: str,
    kernel_code: str,
    branches: list[str],
    reduce_code: str | None = None,
    cut: str | None = None,
    entries_per_file: int | None = None,
    workers: int = 4,
    page: int = 0,
    page_size: int = 1000,
) -> dict[str, Any]:
    """Run a sandboxed kernel over multiple ROOT files and auto-reduce the results.

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
        ``events`` is a ``dict[str, array]`` keyed by the names in *branches*.
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
    dict with keys:

    - ``result_type``: ``"array"``, ``"scalar"``, ``"dict"``, or ``"empty"``
    - ``data``: reduced result (paginated slice for arrays)
    - ``n_files``, ``n_files_ok``, ``n_files_failed``, ``failed_files``
    - ``per_file_elapsed_s``: list of per-file wall times
    - ``elapsed_s``: total wall time
    - Array results additionally include: ``total``, ``page``, ``page_size``,
      ``page_count``, ``has_more``
    - Scalar results additionally include: ``sum``, ``mean``
    """
    try:
        result = analysis.run_kernel_dataset(
            file_paths,
            tree_name,
            kernel_code,
            branches,
            reduce_code=reduce_code,
            cut=cut,
            entries_per_file=entries_per_file,
            workers=workers,
            page=page,
            page_size=page_size,
        )
        return _json_safe(result)
    except Exception as exc:
        return {
            "error": str(exc),
            "file_paths": file_paths,
            "tree_name": tree_name,
            "branches": branches,
        }


@mcp.tool()
def estimate_dataset_cost(
    file_paths: list[str],
    tree_name: str,
    kernel_code: str,
    branches: list[str],
    sample_files: int = 3,
    entries_per_file: int = 1000,
) -> dict[str, Any]:
    """Measure kernel performance on a sample and extrapolate to the full dataset.

    Opens each file metadata-only to count total entries, then times the kernel
    on a small sample to produce a wall-time cost estimate for the full dataset.

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
    - ``recommended_prototype_entries_per_file``: entries per file keeping a
      full-dataset run under 30 s
    - ``sample_elapsed_s``: wall time for all sample kernel executions
    - ``elapsed_s``: total wall time including metadata reads
    """
    try:
        result = analysis.estimate_dataset_cost(
            file_paths,
            tree_name,
            kernel_code,
            branches,
            sample_files=sample_files,
            entries_per_file=entries_per_file,
        )
        return _json_safe(result)
    except Exception as exc:
        return {
            "error": str(exc),
            "file_paths": file_paths,
            "tree_name": tree_name,
            "branches": branches,
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the MCP server using stdio transport."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
