.PHONY: install infra up down init ingest serve benchmark test

install:
	pip install -r requirements.txt

infra:            ## start PostgreSQL (pgvector) and Neo4j only
	docker compose up -d postgres neo4j

up:               ## full stack incl. API on :8000
	docker compose up -d --build

down:
	docker compose down

init:
	python scripts/init_db.py

ingest:
	python scripts/ingest.py data/sample_corpus

serve:
	uvicorn app.main:app --reload --port 8000

benchmark:
	python scripts/run_benchmark.py

test:
	python -m pytest -q
