# GitHub Copilot Instructions

For detailed instructions on working with this repository, please refer to:

**[AGENTS.md](../AGENTS.md)** - Single source of truth for AI agent guidance

## Quick Reference

This is the **uproot MCP Server** project:
- Python MCP server for ROOT file analysis using uproot
- Target: EIC (Electron-Ion Collider) scientific data analysis
- Read AGENTS.md for architecture, conventions, and development guidelines

## Key Points

1. **Layer separation**: `analysis.py` for pure computation, `server.py` for MCP wrappers
2. **JSON safety**: apply `_json_safe()` to all tool return values; replace NaN/Inf with `None`
3. **Array handling**: always use `_flatten_to_float()` before computing statistics
4. **File access**: support both local paths and XRootD URLs (`root://server//path`)
5. **Testing**: run tests with `python -m pytest tests/ -v`

## Common Tasks

See AGENTS.md for:
- Adding new tools (step-by-step guide)
- Modifying analysis functions
- Testing changes (local and remote)
- Code style guidelines
- EIC-specific ROOT file structure

## Architecture

```
src/uproot_mcp_server/analysis.py  - Core analysis logic (uproot, numpy, awkward)
src/uproot_mcp_server/server.py    - FastMCP server, tool definitions, entry point
tests/                             - pytest test suite
```

## Documentation

- **AGENTS.md** - Complete AI agent instructions (read this first!)
- **README.md** - User-facing documentation

---

**Always consult AGENTS.md before making significant changes to ensure consistency with project standards and conventions.**
