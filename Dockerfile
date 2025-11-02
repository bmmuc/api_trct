FROM python:3.11-slim as builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock* ./

RUN uv sync --frozen --no-dev --no-install-project

FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

# Copy application source
COPY src ./src

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_STORAGE_PATH=/app/model_storage

RUN mkdir -p /app/model_storage

# Expose port
EXPOSE 80

# Run the application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "80"]
