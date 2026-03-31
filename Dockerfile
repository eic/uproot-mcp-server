# syntax=docker/dockerfile:1

FROM python:3.12-slim

# Install system dependencies for uproot/XRootD
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project metadata and install dependencies
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install the package with all dependencies
RUN pip install --no-cache-dir ".[xrootd]" && \
    pip cache purge

# Use non-root user for security
RUN useradd -r -s /bin/false appuser && \
    chown -R appuser:appuser /app

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import uproot_mcp_server" || exit 1

# Labels
LABEL org.opencontainers.image.title="uproot MCP Server" \
      org.opencontainers.image.description="Model Context Protocol server for uproot-based ROOT file analysis" \
      org.opencontainers.image.vendor="EIC" \
      org.opencontainers.image.source="https://github.com/eic/uproot-mcp-server" \
      org.opencontainers.image.licenses="MIT"

# Run the MCP server (stdio transport)
CMD ["uproot-mcp-server"]
