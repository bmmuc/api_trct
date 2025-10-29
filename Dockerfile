# syntax=docker/dockerfile:1
FROM python:3.9-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /code

# Copy project files
COPY pyproject.toml .
COPY src ./src

# Install dependencies with uv
RUN uv sync --frozen

# Set environment variable to use the virtual environment
ENV PATH="/code/.venv/bin:$PATH"

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "80"]
