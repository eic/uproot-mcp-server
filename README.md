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
- **Local and remote files**: Works with local paths and XRootD URLs (`root://server//path/to/file.root`)
- **JSON output**: All results are returned as JSON-serialisable dicts for easy client consumption

## Installation

### From source

```bash
git clone https://github.com/eic/uproot-mcp-server.git
cd uproot-mcp-server
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
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
- *(optional)* `xrootd` ≥ 5.4 — required for `root://` URLs

## Usage

### Starting the server (stdio transport)

```bash
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
python -m uproot_mcp_server.server
# or, after pip install:
uproot-mcp-server
```

### MCP client configuration

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
On Windows use `.venv\Scripts\uproot-mcp-server.exe` instead.

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

## Development

### Running tests

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
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
