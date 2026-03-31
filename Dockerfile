# syntax=docker/dockerfile:1

# ---- Build stage ----
FROM python:3.12-slim AS builder

# Install build dependencies needed to compile xrootd from source on non-x86_64
# (pre-built manylinux wheels are only available for x86_64)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    cmake \
    g++ \
    libssl-dev \
    uuid-dev \
    zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir ".[xrootd]" && \
    pip cache purge

# ---- Runtime stage ----
FROM python:3.12-slim

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    libuuid1 && \
    rm -rf /var/lib/apt/lists/*

# Copy installed Python packages and entrypoint from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
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
