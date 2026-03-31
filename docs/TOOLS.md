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

## Error handling

All tools catch exceptions and return an error dictionary rather than raising.
Clients should check for the presence of the `"error"` key in the response:

```json
{
  "error": "File not found: /data/missing.root",
  "file_path": "/data/missing.root"
}
```
