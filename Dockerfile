# syntax=docker/dockerfile:1

# ---- Build stage ----
FROM python:3.14-slim AS builder

WORKDIR /build

COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install base package, then xrootd only if a binary wheel is available
# (xrootd does not publish aarch64 wheels, so building from source is avoided).
# fsspec-xrootd is always installed unconditionally because it is a pure-Python
# package and is required to resolve root:// URLs via fsspec even when the
# native xrootd wheel is present.
RUN pip install --no-cache-dir "." && \
    (pip install --no-cache-dir --only-binary=xrootd "xrootd>=5.4.0" || true) && \
    pip install --no-cache-dir "fsspec-xrootd>=0.5.2" && \
    pip cache purge

# ---- Runtime stage ----
FROM python:3.14-slim

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    libuuid1 && \
    rm -rf /var/lib/apt/lists/*

# Copy installed Python packages and entrypoint from builder
COPY --from=builder /usr/local/lib/python3.14/site-packages /usr/local/lib/python3.14/site-packages
COPY --from=builder /usr/local/bin/uproot-mcp-server /usr/local/bin/uproot-mcp-server

# Use non-root user for security
RUN useradd -r -s /bin/false appuser

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
