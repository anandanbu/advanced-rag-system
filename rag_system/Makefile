# Makefile — common developer commands
# Usage: make <target>

.PHONY: help install run run-ui test test-unit lint clean ingest reset docker docker-up docker-down

# Default target
help:
	@echo ""
	@echo "  Advanced RAG System — Developer Commands"
	@echo "  ─────────────────────────────────────────"
	@echo "  make install       Install all dependencies"
	@echo "  make run           Start the FastAPI backend"
	@echo "  make run-ui        Start the Streamlit frontend"
	@echo "  make test          Run all tests"
	@echo "  make test-unit     Run only unit tests (fast, no API needed)"
	@echo "  make lint          Run code style checks"
	@echo "  make clean         Remove cache and temp files"
	@echo "  make ingest DIR=./docs  Ingest a folder of documents"
	@echo "  make reset         Reset the vector store (asks for confirmation)"
	@echo "  make eval          Run evaluation script"
	@echo "  make cli           Start interactive CLI"
	@echo "  make docker-up     Build and start with Docker Compose"
	@echo "  make docker-down   Stop Docker services"
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────────────

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install pytest pytest-asyncio httpx black flake8

# ── Running ───────────────────────────────────────────────────────────────────

run:
	uvicorn main:app --reload --host 0.0.0.0 --port 8000

run-ui:
	streamlit run streamlit_app.py

cli:
	python scripts/query_cli.py --show-sources

# ── Testing ───────────────────────────────────────────────────────────────────

test:
	pytest tests/ -v

test-unit:
	pytest tests/test_ingestion.py tests/test_memory.py -v

test-embeddings:
	pytest tests/test_embeddings.py -v -s

test-pipeline:
	pytest tests/test_rag_pipeline.py -v

test-api:
	pytest tests/test_api.py -v

test-coverage:
	pytest tests/ --cov=. --cov-report=html --cov-omit="tests/*,scripts/*"
	@echo "Coverage report: htmlcov/index.html"

# ── Code Quality ──────────────────────────────────────────────────────────────

lint:
	flake8 . --max-line-length=100 --exclude=.venv,venv,__pycache__,data

format:
	black . --line-length=100 --exclude=".venv|venv|__pycache__|data"

# ── Data Operations ───────────────────────────────────────────────────────────

ingest:
	@if [ -z "$(DIR)" ]; then echo "Usage: make ingest DIR=./your_docs_folder"; exit 1; fi
	python scripts/ingest_folder.py --folder $(DIR)

ingest-recursive:
	@if [ -z "$(DIR)" ]; then echo "Usage: make ingest-recursive DIR=./your_docs_folder"; exit 1; fi
	python scripts/ingest_folder.py --folder $(DIR) --recursive

reset:
	python scripts/reset_db.py

eval:
	python scripts/evaluate_rag.py

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.pyo" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage 2>/dev/null || true
	@echo "Cleaned!"

# ── Docker ────────────────────────────────────────────────────────────────────

docker-build:
	docker build -f docker/Dockerfile -t rag-system:latest .

docker-up:
	docker-compose -f docker/docker-compose.yml up --build -d
	@echo "Services starting... API: http://localhost:8000 | UI: http://localhost:8501"

docker-down:
	docker-compose -f docker/docker-compose.yml down

docker-logs:
	docker-compose -f docker/docker-compose.yml logs -f rag-api

docker-shell:
	docker exec -it rag-api bash
