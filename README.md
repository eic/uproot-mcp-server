# uproot-mcp-server

An MCP server for performing ROOT file analysis using [uproot](https://uproot.readthedocs.io/).
It acts as a computational backend for EIC data analysis workflows, reading large ROOT datasets
and returning compact, JSON-serialisable summaries suitable for display, tabulation, or
visualisation by MCP clients.

This server is a partner project to [xrootd-mcp-server](https://github.com/eic/xrootd-mcp-server)
and follows a similar structure, using Python as the primary language to enable native uproot
integration.

## Features

- **File structure inspection**: List all keys, TTrees, branches, and leaves in a ROOT file
- **Tree metadata**: Detailed per-branch information including type, entry count, and compression
- **Summary statistics**: Mean, std, min, max, percentiles (p25/p50/p75), and non-finite value counts for any branch
- **Histogramming with selection**: 1-D histograms with configurable bins, explicit range, and boolean cut expressions
- **Sandboxed kernel execution**: Run arbitrary multi-branch Python computations in a restricted sandbox (no imports, no file I/O)
- **Local and remote files**: Works with local paths and XRootD URLs (`root://server//path/to/file.root`)
- **JSON output**: All results are returned as JSON-serialisable dicts for easy client consumption

## Installation

### Using Docker (Recommended)

Pull and run the latest image published to the GitHub Container Registry:

```bash
docker run -i --rm \
  ghcr.io/eic/uproot-mcp-server:latest
```

### Using Docker Compose with Watchtower

The bundled `docker-compose.yml` starts the MCP server alongside
[Watchtower](https://containrrr.dev/watchtower/), which automatically pulls
and restarts the container whenever a new image is published to `ghcr.io`:

```bash
docker compose up -d
```

Watchtower polls for updates every hour by default.  Override the interval (in
seconds) with the `WATCHTOWER_POLL_INTERVAL` environment variable:

```bash
WATCHTOWER_POLL_INTERVAL=1800 docker compose up -d
```

### From source

```bash
git clone https://github.com/eic/uproot-mcp-server.git
cd uproot-mcp-server
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For XRootD remote file access (``root://server//path`` URLs), install the optional
XRootD client:

```bash
pip install -e ".[xrootd]"
```

### Dependencies

- Python ≥ 3.10
- `uproot` ≥ 5.0
- `numpy` ≥ 1.24
- `awkward` ≥ 2.0
- `mcp` ≥ 1.0
- `RestrictedPython` ≥ 7.0
- *(optional)* `xrootd` ≥ 5.4 — required for `root://` URLs

## Usage

### Starting the server (stdio transport)

```bash
source .venv/bin/activate
python -m uproot_mcp_server.server
# or, after pip install:
uproot-mcp-server
```

### MCP client configuration

#### Local installation

Add to your MCP client configuration file (e.g. Claude Desktop), using the full path
to the executable inside the virtual environment:

```json
{
  "mcpServers": {
    "uproot": {
      "command": "/path/to/uproot-mcp-server/.venv/bin/uproot-mcp-server"
    }
  }
}
```

Replace `/path/to/uproot-mcp-server` with the absolute path to the cloned repository.

#### Docker

```json
{
  "mcpServers": {
    "uproot": {
      "command": "docker",
      "args": ["run", "-i", "--rm", "ghcr.io/eic/uproot-mcp-server:latest"]
    }
  }
}
```

### Available tools

#### `get_file_structure`

Returns the top-level structure of a ROOT file (all keys and a summary of each TTree).

```json
{
  "file_path": "/data/events.root"
}
```

#### `get_tree_info`

Returns detailed metadata for a single TTree including all branches and leaves.

```json
{
  "file_path": "/data/events.root",
  "tree_name": "events"
}
```

#### `get_branch_statistics`

Computes summary statistics (mean, std, min, max, percentiles) for a single branch.
Supports an optional boolean selection cut and entry range.

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

#### `histogram_branch`

Produces a 1-D histogram of a branch with optional selection, configurable bins, and explicit range.

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

Response includes `edges` (bin boundaries), `counts`, `underflow`, `overflow`, `entries`, `mean`, and `std`.

#### `execute_kernel`

Execute an arbitrary Python computation over one or more branches in a **sandboxed environment**.
The kernel receives a `dict[branch_name, array]` and may return an array, scalar, or dict.

```json
{
  "file_path": "/data/events.root",
  "tree_name": "events",
  "kernel_code": "def kernel(events):\n    px = events['px']\n    py = events['py']\n    pz = events['pz']\n    return np.sqrt(px**2 + py**2 + pz**2)\n",
  "branches": ["px", "py", "pz"],
  "page": 0,
  "page_size": 1000
}
```

**Sandbox restrictions** — the following are blocked inside kernels:

| Blocked | Reason |
|---|---|
| `import` / `__import__` | No access to system modules |
| `exec`, `eval`, `compile` | No dynamic code execution |
| `open` | No file system access |
| Dunder attribute access (`obj.__class__`, etc.) | Blocked at AST compile time by RestrictedPython |
| Writes to `np` / `ak` modules | Prevented by `_write_` guard |

**Available in kernels:**

- `np` — the full numpy package
- `ak` — the full awkward-array package
- Safe built-ins: `len`, `range`, `list`, `dict`, `tuple`, `set`, `int`, `float`, `bool`, `str`, `abs`, `min`, `max`, `sum`, `round`, `zip`, `enumerate`, `map`, `filter`, `sorted`, `reversed`, `isinstance`, `print`, and standard exception types

**Response** (array result):

```json
{
  "result_type": "array",
  "data": [1.2, 3.4, ...],
  "total": 9698,
  "page": 0,
  "page_size": 1000,
  "page_count": 10,
  "has_more": true,
  "file_path": "...",
  "tree_name": "events",
  "branches": ["px", "py", "pz"]
}
```

For scalar or dict results, `result_type` is `"scalar"` or `"dict"` and pagination fields are absent.

**Execution timeout:** kernels run in an isolated subprocess that is forcefully terminated (SIGTERM then SIGKILL) after 30 seconds of wall-clock time.

## Development

### Running tests

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v
```

### Remote integration tests

Set the `UPROOT_TEST_REMOTE_FILE` environment variable to an XRootD URL to enable
remote file tests:

```bash
export UPROOT_TEST_REMOTE_FILE="root://dtn-eic.jlab.org//work/eic2/EPIC/RECO/24.07.0/epic_craterlake/DIS/NC/18x275/q2_0.001_1.0/pythia8NCDIS_18x275_minQ2=0.001_beamEffects_xAngle=-0.025_hiDiv_1.0000.eicrecon.tree.edm4eic.root"
pytest tests/ -v
```

## License

MIT
