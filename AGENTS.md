# AI Agent Instructions for uproot MCP Server

This document provides guidance for AI agents (GitHub Copilot, Claude, ChatGPT, etc.) working on this repository.

## Project Overview

**uproot MCP Server** - A Model Context Protocol server that provides LLMs access to ROOT file analysis via [uproot](https://uproot.readthedocs.io/) for the EIC (Electron-Ion Collider) project.

- **Language**: Python (≥ 3.10)
- **MCP framework**: `mcp>=1.0.0` (FastMCP)
- **Core libraries**: `uproot>=5.0`, `numpy>=1.24`, `awkward>=2.0`
- **Purpose**: ROOT file structure inspection, branch statistics, and histogramming for scientific computing
- **Target**: HEP (High Energy Physics) / Nuclear Physics data analysis workflows
- **Partner project**: [xrootd-mcp-server](https://github.com/eic/xrootd-mcp-server) for file discovery

## Architecture

```
src/uproot_mcp_server/
├── __init__.py       # Package marker
├── analysis.py       # Core ROOT file analysis logic (uproot, numpy, awkward)
└── server.py         # FastMCP server, tool definitions and entry point

tests/
├── fixtures/
│   ├── create_fixture.py  # Script that generates tests/fixtures/test_eic.root
│   └── test_eic.root      # Synthetic ROOT file used by local tests
├── test_analysis.py  # Tests for analysis.py public API
└── test_server.py    # Tests for server-level tool wrappers
```

## Key Principles

### 1. Analysis Layer Separation
- **`analysis.py`**: Pure computation — opens files, reads arrays, returns plain Python dicts.
  No MCP-specific code here.
- **`server.py`**: Thin MCP wrapper — calls `analysis.*` functions, applies `_json_safe()`,
  and catches exceptions, returning an error dict instead of raising.
- Keep the two layers decoupled so `analysis.py` is independently testable.

### 2. JSON Serialisability
- All tool return values must be JSON-serialisable.
- Use `_json_safe()` in `server.py` to replace `NaN`/`Inf` float values with `None`.
- Return plain Python `dict`, `list`, `int`, `float`, `str`, or `None` — never numpy scalars or awkward arrays.

### 3. File Access
- Local paths and XRootD URLs (`root://server//path/to/file.root`) are both supported.
- Always call `uproot.open(file_path)` via the `_open_file()` helper in `analysis.py`.
- Use `with _open_file(file_path) as f:` to ensure the file handle is released.

### 4. Array Handling
- uproot can return both fixed-size numpy arrays and jagged/variable-length awkward arrays.
- Always flatten through `_flatten_to_float()` before computing statistics or histograms.
- `_flatten_to_float()` handles both `ak.Array` and regular numpy objects.

### 5. MCP Protocol Standards
- **Tool definitions**: Use the `@mcp.tool()` decorator; docstring becomes the tool description.
- **Parameters**: Type-annotated function signatures are sufficient — FastMCP generates the JSON schema.
- **Return type**: All tools return `dict[str, Any]` — FastMCP serialises this to JSON.
- **Error handling**: Catch all exceptions in `server.py` and return `{"error": str(exc), ...}`.

### 6. Python Best Practices
- Use `from __future__ import annotations` at the top of every source file.
- Full type annotations on all public functions.
- Docstrings following the NumPy docstring convention.
- Target Python 3.10+ (use `X | Y` union syntax, `match` statements where appropriate).

## Common Tasks

### Adding a New Tool

1. **Implement the function** in `src/uproot_mcp_server/analysis.py`:

```python
def my_new_analysis(
    file_path: str,
    tree_name: str,
    *,
    optional_param: int = 10,
) -> dict[str, Any]:
    """Return something useful about the tree.

    Parameters
    ----------
    file_path:
        Local path or XRootD URL to the ROOT file.
    tree_name:
        Name of the TTree to inspect.
    optional_param:
        Description of optional parameter.
    """
    with _open_file(file_path) as f:
        tree = f[tree_name]
        # ... computation ...
        return {
            "file_path": file_path,
            "tree_name": tree_name,
            # ... results ...
        }
```

2. **Add the MCP tool** in `src/uproot_mcp_server/server.py`:

```python
@mcp.tool()
def my_new_analysis(
    file_path: str,
    tree_name: str,
    optional_param: int = 10,
) -> dict[str, Any]:
    """Return something useful about the tree.

    Parameters
    ----------
    file_path:
        Local path or XRootD URL to the ROOT file.
    tree_name:
        Name of the TTree.
    optional_param:
        Description of optional parameter (default 10).
    """
    try:
        result = analysis.my_new_analysis(
            file_path,
            tree_name,
            optional_param=optional_param,
        )
        return _json_safe(result)
    except Exception as exc:
        return {"error": str(exc), "file_path": file_path, "tree_name": tree_name}
```

3. **Add tests** in `tests/test_analysis.py` and `tests/test_server.py`
4. **Update documentation**: Add to README.md tool list

### Modifying Analysis Functions

When reading arrays from uproot:

```python
# ✅ GOOD: use _flatten_to_float for any numeric branch
data = _flatten_to_float(tree[branch_name].array(library="ak", **read_kwargs))
finite = data[np.isfinite(data)]

# ❌ BAD: assume fixed-size array, will fail on jagged branches
data = np.array(tree[branch_name].array())
```

When returning statistics:

```python
# ✅ GOOD: guard against non-finite values in results
def _safe(v: Any) -> Any:
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return float(v)

return {"mean": _safe(float(np.mean(finite))), ...}

# ❌ BAD: NaN/Inf in the dict breaks JSON serialisation
return {"mean": float(np.mean(data)), ...}
```

### Testing Changes

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all local tests
python -m pytest tests/ -v

# Run a specific test class
python -m pytest tests/test_analysis.py::TestGetFileStructure -v

# Run remote integration tests (requires network access to dtn-eic.jlab.org)
export UPROOT_TEST_REMOTE_FILE="root://dtn-eic.jlab.org//work/eic2/EPIC/RECO/..."
python -m pytest tests/ -v
```

### Regenerating the Test Fixture

```bash
# Recreate tests/fixtures/test_eic.root
python tests/fixtures/create_fixture.py
```

## Code Style Guidelines

### File Organisation
- Module-level docstring describing the file's purpose
- `from __future__ import annotations` immediately after the docstring
- Standard library imports → third-party imports → local imports (blank line between groups)
- Type aliases and constants at module level before functions
- Internal helpers prefixed with `_`; public API after internal helpers

### Naming Conventions
- **Modules**: `snake_case`
- **Functions / methods**: `snake_case`
- **Type aliases**: `PascalCase` (e.g. `FileStructure`, `BranchStatistics`)
- **Constants**: `UPPER_SNAKE_CASE`
- **Private helpers**: `_leading_underscore`

### Docstrings
- NumPy-style docstrings for all public functions.
- Include `Parameters`, `Returns` sections.
- Document what the dict keys are in the Returns section.

### Error Messages

```python
# ✅ GOOD: descriptive, includes context
raise ValueError(f"bins must be >= 1, got {bins}")

# ❌ BAD: generic, no context
raise ValueError("invalid input")
```

## EIC-Specific Knowledge

### Typical ROOT File Structure

EIC reconstructed data files follow this pattern:

```
events (TTree)
├── MCParticles.momentum.x    # float64 branch (jagged for variable multiplicity)
├── MCParticles.momentum.y
├── MCParticles.momentum.z
├── MCParticles.charge
├── ReconstructedParticles.*  # more branches
└── ...
```

### File Locations (via xrootd-mcp-server)

```
root://dtn-eic.jlab.org//work/eic2/EPIC/RECO/{campaign}/
  epic_craterlake/{process_type}/{process}/{generator}/{beams}/{q2_bin}/{particle}/
    *.eicrecon.tree.edm4eic.root
```

### Common Campaigns
- `25.10.x` — current production series
- `24.07.x` — previous year productions

### Branch Naming Patterns
- Fully-qualified dot-notation: `"MCParticles.momentum.x"`
- Use `get_tree_info` first to discover available branches before calling statistics tools

## Debugging Tips

### Common Issues

1. **`TypeError: cannot convert non-finite float`**
   - The result dict contains `NaN` or `Inf`
   - Apply `_json_safe()` (in `server.py`) or `_safe()` guards (in `analysis.py`)

2. **`ValueError: jagged array cannot be converted to numpy`**
   - A variable-length (jagged) branch was passed directly to numpy
   - Use `_flatten_to_float()` which handles awkward arrays

3. **`KeyError` when accessing tree by name (e.g. `f[tree_name]`)**
   - uproot raises `KeyError` with the missing object key (e.g. `"events"` or `"events;1"`) when indexing the file/tree
   - Use `get_file_structure` first to discover the correct tree/key names before accessing them

4. **Remote file hangs / times out**
   - XRootD must be installed: `pip install -e ".[xrootd]"`
   - Verify URL format: `root://server//absolute/path/to/file.root` (double slash)
   - Check VPN / firewall access to `dtn-eic.jlab.org`

5. **`ModuleNotFoundError: No module named 'uproot_mcp_server'`**
   - Install in editable mode: `pip install -e .`

### Verifying a Tool Manually

```python
# Quick smoke-test without running the full server
from uproot_mcp_server import analysis

result = analysis.get_file_structure("tests/fixtures/test_eic.root")
print(result)

result = analysis.get_branch_statistics(
    "tests/fixtures/test_eic.root", "events", "px"
)
print(result)
```

## Configuration

### Entry Points
```bash
# stdio transport (default, for MCP clients)
uproot-mcp-server
# or equivalently:
python -m uproot_mcp_server.server
```

### MCP Client Configuration

Add to your MCP client configuration file (e.g. Claude Desktop `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "uproot": {
      "command": "uproot-mcp-server"
    }
  }
}
```

For a virtual-environment installation:

```json
{
  "mcpServers": {
    "uproot": {
      "command": "/path/to/venv/bin/uproot-mcp-server"
    }
  }
}
```

### Optional Dependencies
```bash
# XRootD remote file access (root:// URLs)
pip install -e ".[xrootd]"

# Development / testing
pip install -e ".[dev]"
```

## Future Development

### Planned Features
1. **2-D histograms** — correlate two branches
2. **Batch statistics** — compute stats for many branches in one call
3. **ROOT histogram objects** — read `TH1`/`TH2` objects directly from files
4. **Streaming large files** — chunked reading for very large TTrees
5. **Metadata caching** — cache file structure to avoid repeated opens

### Extension Points
- Add new analysis functions to `analysis.py`; wrap with `@mcp.tool()` in `server.py`
- New array-flattening strategies go in `_flatten_to_float()` or a new helper
- Extend `_branch_info()` to return additional per-branch metadata

## Resources

- **MCP Specification**: https://modelcontextprotocol.io/
- **uproot Documentation**: https://uproot.readthedocs.io/
- **awkward-array Documentation**: https://awkward-array.org/
- **EIC Software**: https://eic.github.io/
- **Partner project**: https://github.com/eic/xrootd-mcp-server

## Questions?

For questions about:
- **MCP protocol**: Check MCP SDK docs; see `server.py` for existing patterns
- **uproot usage**: Refer to uproot docs; test with `analysis.py` functions directly
- **awkward arrays**: See awkward-array docs; `_flatten_to_float()` handles most cases
- **EIC data structure**: Use `get_file_structure` on a real file; consult EIC Software docs

## Version History

- **v0.1.0** (2025): Initial release with 4 tools
  - `get_file_structure` — top-level file/tree summary
  - `get_tree_info` — detailed branch metadata
  - `get_branch_statistics` — mean, std, min, max, percentiles
  - `histogram_branch` — 1-D histogram with optional selection

---

*This document should be kept up-to-date as the project evolves. Update it when adding new tools, changing architecture, or discovering new best practices.*
