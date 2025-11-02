.PHONY: help install test test-cov lint dev clean performance
.PHONY: docker-build docker-run-local docker-stop docker-logs docker-clean docker-shell

help:
	@echo "=== Anomaly Detection API - Makefile ==="
	@echo ""
	@echo "Installation:"
	@echo "  make install           - Install dependencies with uv"
	@echo ""
	@echo "Testing & Linting:"
	@echo "  make test              - Run unit tests with pytest"
	@echo "  make test-cov          - Run tests with coverage report"
	@echo "  make lint              - Run code linting"
	@echo "  make performance       - Run performance tests"
	@echo ""
	@echo "Development:"
	@echo "  make dev               - Start development server (local)"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build      - Build Docker image"
	@echo "  make docker-run-local  - Run Docker container locally"
	@echo "  make docker-stop       - Stop and remove container"
	@echo "  make docker-logs       - View container logs"
	@echo "  make docker-shell      - Open shell in running container"
	@echo "  make docker-clean      - Remove images and containers"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean             - Clean cache and temp files"
	@echo ""

install:
	@echo "Installing dependencies with uv..."
	uv sync --all-extras
	@echo "✓ Dependencies installed"

test:
	@echo "Running tests..."
	cd src && uv run pytest -p no:launch_testing tests/ -v --tb=short
	@echo "Tests completed"

test-cov:
	@echo "Running tests with coverage..."
	cd src && uv run pytest tests/ -v --cov=. --cov-report=html --cov-report=term --cov-report=term-missing --cov-config=../.coveragerc
	@echo "✓ Tests completed - Coverage report in htmlcov/"

lint:
	@echo "Running linting..."
	@echo "Checking with flake8..."
	uv run flake8 src/ --max-line-length=120 --exclude=__pycache__,*.pyc
	@echo "✓ Flake8 passed"
	@echo "Checking with pylint..."
	uv run pylint src/ --disable=C0111,C0103,R0913 || true
	@echo "✓ Linting completed"

performance:
	@echo "Running performance tests..."
	@echo "Make sure the API is running (make dev or make docker-run-local)"
	uv run python performance_test.py
	@echo "Check performance_report.md and performance_report.json for results"

docker-build:
	@echo "Building Docker image..."
	docker build -t anomaly-detection-api:latest -f Dockerfile .
	@echo "Docker image built: anomaly-detection-api:latest"

docker-run-local:
	@echo "Running Docker container locally..."
	@docker stop anomaly-detection-api 2>/dev/null || true
	@docker rm anomaly-detection-api 2>/dev/null || true
	docker run -d \
		--name anomaly-detection-api \
		-p 8000:80 \
		-v $(PWD)/model_storage:/code/model_storage \
		-e API_HOST=0.0.0.0 \
		-e API_PORT=80 \
		-e LOG_LEVEL=INFO \
		-e MODEL_STORAGE_PATH=/code/model_storage \
		anomaly-detection-api:latest
	@echo "Container started"
	@echo "API: http://localhost:8000"
	@echo "Docs: http://localhost:8000/docs"
	@echo "Health: http://localhost:8000/health"

docker-stop:
	@echo "Stopping Docker container..."
	docker stop anomaly-detection-api 2>/dev/null || true
	docker rm anomaly-detection-api 2>/dev/null || true
	@echo "Container stopped and removed"

docker-logs:
	@echo "Container logs (Ctrl+C to exit):"
	@docker logs -f anomaly-detection-api

docker-shell:
	@echo "Opening shell in container..."
	@docker exec -it anomaly-detection-api /bin/bash

docker-clean:
	@echo "Cleaning Docker resources..."
	@docker stop anomaly-detection-api 2>/dev/null || true
	@docker rm anomaly-detection-api 2>/dev/null || true
	@docker rmi anomaly-detection-api:latest 2>/dev/null || true
	@echo "Cleanup completed"

dev:
	@echo "Starting development server..."
	uv run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

clean:
	@echo "Cleaning up..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage src/.coverage 2>/dev/null || true
	@echo "Cleanup completed"