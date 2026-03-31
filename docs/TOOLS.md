# Tool Reference

The uproot MCP Server exposes four tools for reading and analysing ROOT files.
All tools return JSON-serialisable dictionaries.  Non-finite floating-point
values (`NaN`, `±Inf`) are normalised to `null` before serialisation.

---

## `get_file_structure`

Returns the top-level structure of a ROOT file.

Lists every key stored in the file and provides a summary of each TTree.
Use this as the first step when exploring an unfamiliar file.

### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `file_path` | string | ✓ | Local path or XRootD URL to the ROOT file |

### Example request

```json
{
  "file_path": "/data/events.root"
}
```

### Example response

```json
{
  "file_path": "/data/events.root",
  "keys": [
    { "name": "events", "classname": "TTree", "cycle": 1 },
    { "name": "metadata", "classname": "TTree", "cycle": 1 }
  ],
  "trees": [
    {
      "name": "events",
      "num_entries": 10000,
      "num_branches": 42,
      "branches": [
        { "name": "MCParticles.momentum.x", "typename": "float32[]" }
      ]
    }
  ]
}
```

---

## `get_tree_info`

Returns detailed metadata for a single TTree including all branches and leaves.

### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `file_path` | string | ✓ | Local path or XRootD URL to the ROOT file |
| `tree_name` | string | ✓ | Name of the TTree, e.g. `"events"` or `"events;1"` |

### Example request

```json
{
  "file_path": "/data/events.root",
  "tree_name": "events"
}
```

### Example response

```json
{
  "file_path": "/data/events.root",
  "tree_name": "events",
  "title": "",
  "num_entries": 10000,
  "num_branches": 3,
  "branches": [
    {
      "name": "MCParticles.momentum.x",
      "typename": "float32[]",
      "num_entries": 10000,
      "uncompressed_bytes": 400000,
      "compressed_bytes": 180000,
      "compression_ratio": 2.22,
      "leaves": [
        { "name": "MCParticles.momentum.x", "typename": "float" }
      ]
    }
  ]
}
```

---

## `get_branch_statistics`

Computes summary statistics for a single branch.

Branch values are read into a flat array (variable-length branches are
flattened automatically) and summary statistics are computed over finite values.
Non-finite values are counted separately.

### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `file_path` | string | ✓ | Local path or XRootD URL to the ROOT file |
| `tree_name` | string | ✓ | Name of the TTree |
| `branch_name` | string | ✓ | Fully-qualified branch name, e.g. `"MCParticles.momentum.x"` |
| `cut` | string | — | Boolean selection expression evaluated per entry, e.g. `"MCParticles.charge != 0"` |
| `entry_start` | integer | — | First entry index (Python-style, inclusive) |
| `entry_stop` | integer | — | Last entry index (Python-style, exclusive) |

### Example request

```json
{
  "file_path": "/data/events.root",
  "tree_name": "events",
  "branch_name": "MCParticles.momentum.x",
  "cut": "MCParticles.charge != 0",
  "entry_start": 0,
  "entry_stop": 10000
}
```

### Example response

```json
{
  "file_path": "/data/events.root",
  "tree_name": "events",
  "branch_name": "MCParticles.momentum.x",
  "cut": "MCParticles.charge != 0",
  "count": 52341,
  "mean": 0.0312,
  "std": 1.843,
  "min": -9.98,
  "max": 9.97,
  "p25": -1.21,
  "p50": 0.03,
  "p75": 1.27,
  "num_nan": 0,
  "num_inf": 0
}
```

---

## `histogram_branch`

Produces a 1-D histogram of a branch with an optional selection cut.

The result contains bin edges, counts, under/overflow, and basic statistics.
It can be used directly by the client for plotting or further analysis.

### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `file_path` | string | ✓ | Local path or XRootD URL to the ROOT file |
| `tree_name` | string | ✓ | Name of the TTree |
| `branch_name` | string | ✓ | Branch to histogram, e.g. `"MCParticles.momentum.x"` |
| `bins` | integer | — | Number of histogram bins (default `100`, must be ≥ 1) |
| `range_min` | float | — | Lower histogram range (both `range_min` and `range_max` must be given together, or both omitted for auto-range) |
| `range_max` | float | — | Upper histogram range |
| `cut` | string | — | Boolean selection expression, e.g. `"MCParticles.charge != 0"` |
| `entry_start` | integer | — | First entry index |
| `entry_stop` | integer | — | Last entry index |

### Example request

```json
{
  "file_path": "root://dtn-eic.jlab.org//path/to/file.root",
  "tree_name": "events",
  "branch_name": "MCParticles.momentum.x",
  "bins": 100,
  "range_min": -5.0,
  "range_max": 5.0,
  "cut": "MCParticles.charge != 0"
}
```

### Example response

```json
{
  "file_path": "root://dtn-eic.jlab.org//path/to/file.root",
  "tree_name": "events",
  "branch_name": "MCParticles.momentum.x",
  "cut": "MCParticles.charge != 0",
  "bins": 100,
  "range_min": -5.0,
  "range_max": 5.0,
  "edges": [-5.0, -4.9, "..."],
  "counts": [12, 34, "..."],
  "underflow": 3,
  "overflow": 5,
  "entries": 52341,
  "mean": 0.0312,
  "std": 1.843
}
```

---

## `execute_kernel`

Execute an arbitrary Python computation over one or more branches in a **sandboxed environment**.

Use this when existing tools (`get_branch_statistics`, `histogram_branch`) are not expressive
enough — for example to compute derived quantities from multiple branches, apply custom
aggregations, or return structured results.

### Sandbox restrictions

The kernel runs under [RestrictedPython](https://restrictedpython.readthedocs.io/).
The following are **blocked**:

| Blocked | Mechanism |
|---|---|
| `import` / `__import__` | Not present in restricted builtins |
| `exec`, `eval`, `open`, `compile` | Not present in restricted builtins |
| Dunder attribute access (`obj.__class__`, etc.) | Rejected at AST compile time |
| Writes to `np` / `ak` modules | `_write_` guard raises `AttributeError` |
| Infinite loops / long-running code | 30-second wall-clock timeout; subprocess is forcefully terminated (SIGTERM then SIGKILL) |

Available inside kernels: `np` (numpy), `ak` (awkward-array), and a safe subset of
built-ins: `len`, `range`, `list`, `dict`, `tuple`, `set`, `int`, `float`, `bool`, `str`,
`abs`, `min`, `max`, `sum`, `round`, `zip`, `enumerate`, `map`, `filter`, `sorted`,
`reversed`, `isinstance`, `print`, and standard exception types.

### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `file_path` | string | ✓ | Local path or XRootD URL to the ROOT file |
| `tree_name` | string | ✓ | Name of the TTree |
| `kernel_code` | string | ✓ | Python source defining `def kernel(events): ...` |
| `branches` | array of strings | ✓ | Branch names to load and pass as `events` |
| `cut` | string | — | Optional boolean selection expression |
| `entry_start` | integer | — | First entry index |
| `entry_stop` | integer | — | Last entry index |
| `page` | integer | — | 0-indexed page number for array results (default `0`) |
| `page_size` | integer | — | Elements per page for array results (default `1000`) |

### Kernel interface

The kernel receives `events` — a `dict[str, array]` keyed by the names in `branches`.
It may return:

- An **array** (numpy ndarray, awkward Array, or list) — paginated automatically
- A **scalar** (int, float, bool) — returned as-is
- A **dict** — returned as-is (must be JSON-serialisable)

### Example request

```json
{
  "file_path": "root://dtn-eic.jlab.org//path/to/file.root",
  "tree_name": "events",
  "kernel_code": "def kernel(events):\n    px = events['ReconstructedParticles.momentum.x']\n    py = events['ReconstructedParticles.momentum.y']\n    pz = events['ReconstructedParticles.momentum.z']\n    return np.sqrt(px**2 + py**2 + pz**2)\n",
  "branches": [
    "ReconstructedParticles.momentum.x",
    "ReconstructedParticles.momentum.y",
    "ReconstructedParticles.momentum.z"
  ],
  "page": 0,
  "page_size": 1000
}
```

### Example response (array)

```json
{
  "result_type": "array",
  "data": [1.23, 0.87, 10.41, "..."],
  "total": 9698,
  "page": 0,
  "page_size": 1000,
  "page_count": 10,
  "has_more": true,
  "file_path": "root://dtn-eic.jlab.org//path/to/file.root",
  "tree_name": "events",
  "branches": ["ReconstructedParticles.momentum.x", "..."],
  "cut": null,
  "entry_start": null,
  "entry_stop": null
}
```

For scalar or dict results, `result_type` is `"scalar"` or `"dict"` and pagination
fields (`total`, `page_count`, `has_more`) are absent.

---

## Error handling

All tools catch exceptions and return an error dictionary rather than raising.
Clients should check for the presence of the `"error"` key in the response:

```json
{
  "error": "File not found: /data/missing.root",
  "file_path": "/data/missing.root"
}
```
