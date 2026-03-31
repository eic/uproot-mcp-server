# Usage Guide

This guide explains how to install, configure, and run the uproot MCP Server.

## Installation

### From source

```bash
git clone https://github.com/eic/uproot-mcp-server.git
cd uproot-mcp-server
pip install -e .
```

For XRootD remote file access (`root://server//path` URLs), install the optional
XRootD client:

```bash
pip install -e ".[xrootd]"
```

### Dependencies

| Package | Version | Notes |
|---------|---------|-------|
| Python | ≥ 3.10 | |
| `uproot` | ≥ 5.0 | ROOT file I/O |
| `numpy` | ≥ 1.24 | Numerical arrays |
| `awkward` | ≥ 2.0 | Jagged array support |
| `mcp` | ≥ 1.0 | MCP framework |
| `xrootd` | ≥ 5.4 | *(optional)* Required for `root://` URLs |

## Starting the server

The server communicates over stdio, which is the standard transport for MCP
clients such as Claude Desktop and VS Code Copilot.

```bash
# Run directly with Python
python -m uproot_mcp_server.server

# Or use the installed entry-point (after pip install)
uproot-mcp-server
```

## MCP client configuration

### Claude Desktop

Add the following block to your Claude Desktop configuration file
(`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS,
`%APPDATA%\Claude\claude_desktop_config.json` on Windows):

```json
{
  "mcpServers": {
    "uproot": {
      "command": "uproot-mcp-server"
    }
  }
}
```

### VS Code (GitHub Copilot)

Add the server to your `.vscode/mcp.json` workspace file:

```json
{
  "servers": {
    "uproot": {
      "type": "stdio",
      "command": "uproot-mcp-server"
    }
  }
}
```

## Working with files

### Local files

Pass an absolute or relative filesystem path:

```json
{ "file_path": "/data/events.root" }
```

### Remote files via XRootD

Pass an XRootD URL.  The optional `xrootd` dependency must be installed:

```json
{ "file_path": "root://dtn-eic.jlab.org//work/eic2/EPIC/RECO/24.07.0/epic_craterlake/DIS/NC/18x275/q2_0.001_1.0/pythia8NCDIS_18x275.root" }
```

## Example workflow

A typical analysis session with an AI assistant might look like:

1. **Explore the file** — ask the assistant to list what trees and branches are
   available:

   > "What trees and branches are in `/data/events.root`?"

   The assistant calls `get_file_structure` and returns an overview of every key
   and TTree.

2. **Inspect a tree** — get detailed branch metadata:

   > "Show me the branches of the `events` tree."

   The assistant calls `get_tree_info` and returns per-branch information
   including types, entry counts, and compression.

3. **Compute statistics** — get a numerical summary of a branch:

   > "What are the statistics for `MCParticles.momentum.x` for charged
   > particles only?"

   The assistant calls `get_branch_statistics` with
   `cut="MCParticles.charge != 0"`.

4. **Histogram a distribution** — produce a histogram for plotting:

   > "Make a 50-bin histogram of `MCParticles.momentum.x` from −5 to 5 GeV."

   The assistant calls `histogram_branch` and returns edges, counts, and
   summary statistics ready for display.

## Development

### Running tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

### Remote integration tests

Set `UPROOT_TEST_REMOTE_FILE` to an XRootD URL to enable remote-file tests:

```bash
export UPROOT_TEST_REMOTE_FILE="root://dtn-eic.jlab.org//work/eic2/EPIC/RECO/24.07.0/..."
pytest tests/ -v
```
