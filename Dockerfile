FROM python:3.13-slim

WORKDIR /app

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies (serve extras, no dev)
RUN uv sync --extra serve --no-dev --no-install-project

# Copy package source
COPY convgru_ensemble/ ./convgru_ensemble/

# Install the project itself
RUN uv sync --extra serve --no-dev

# Model checkpoint is mounted at runtime or downloaded from HF Hub
ENV MODEL_CHECKPOINT=/app/model.ckpt
ENV DEVICE=cpu

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "convgru_ensemble.serve:app", "--host", "0.0.0.0", "--port", "8000"]
