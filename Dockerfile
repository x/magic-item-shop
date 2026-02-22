# Build stage: install deps and index items
FROM ghcr.io/astral-sh/uv:bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
ENV UV_NO_DEV=1
ENV UV_PYTHON_INSTALL_DIR=/python
ENV UV_PYTHON_PREFERENCE=only-managed

RUN uv python install 3.12

WORKDIR /app

# Install dependencies (separate layer so code changes don't re-install deps)
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project

# Copy full project and finish install
COPY . /app
RUN uv sync --locked

# Index magic items into ChromaDB at build time
RUN uv run python index_items.py

# Runtime stage
FROM debian:bookworm-slim

RUN groupadd --system --gid 999 nonroot \
 && useradd --system --gid 999 --uid 999 --create-home nonroot

COPY --from=builder --chown=nonroot:nonroot /python /python
COPY --from=builder --chown=nonroot:nonroot /app /app

ENV PATH="/app/.venv/bin:$PATH"
USER nonroot
WORKDIR /app

EXPOSE 8000
CMD ["python", "app.py"]
