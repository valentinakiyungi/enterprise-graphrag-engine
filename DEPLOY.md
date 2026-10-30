# Deployment

The service needs three things: the API container, PostgreSQL with the `pgvector` extension, and Neo4j. Two routes are below.

## Option A: one VPS with Docker Compose (simplest, everything in one place)

Any Linux VPS with 2 GB+ RAM works (Neo4j is the hungry part).

```bash
# on the server (Ubuntu example)
curl -fsSL https://get.docker.com | sh
git clone https://github.com/valentinakiyungi/enterprise-graphrag-engine.git
cd enterprise-graphrag-engine
cp .env.example .env
nano .env     # OPENAI_API_KEY, POSTGRES_PASSWORD, NEO4J_PASSWORD, and a long random API_KEY
docker compose up -d --build
curl localhost:8000/ready
```

Use URL-safe characters in `POSTGRES_PASSWORD` (letters and digits), because it is embedded in a connection URL.

Database ports are bound to `127.0.0.1` only. The API listens on port 8000: put it behind HTTPS before exposing it, for example with Caddy:

```bash
# Caddyfile (point your domain's DNS A record at the server first)
api.example.com {
    reverse_proxy localhost:8000
}
```

Set `API_KEY` so strangers cannot run queries on your OpenAI bill. Updates: `git pull && docker compose up -d --build`. Data persists in the `pgdata` and `neo4jdata` volumes; back them up.

## Option B: managed services

1. **Neo4j:** create a free instance on Neo4j Aura. Save the connection URI (`neo4j+s://...`), username and password.
2. **API + Postgres on Render:** in the Render dashboard choose New, then Blueprint, and select this repo (it reads `render.yaml`). Fill in `OPENAI_API_KEY`, `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`. Render creates Postgres and generates `API_KEY`. `render.yaml` has not been tested, so check its plan names against Render's current options.
3. Verify: `curl https://<your-service>.onrender.com/ready`.

Railway and Fly.io also work: deploy the Dockerfile, attach a Postgres with pgvector, point `NEO4J_*` at Aura, and set the same environment variables. The container respects the `PORT` variable.

## After deploying

1. `POST /ingest` with your documents.
2. `POST /query` with `"evaluate": true` to confirm Ragas scoring works with your OpenAI key.
3. Run the benchmark against your own data before quoting any accuracy or hallucination numbers.

## Notes

- Costs: ingestion makes one embedding call and one extraction call per chunk; `evaluate: true` adds several LLM calls per query. Leave `EVALUATE_BY_DEFAULT=false` for latency-sensitive traffic and sample instead.
- pgvector's HNSW index supports up to 2000 dimensions. `text-embedding-3-small` (1536) fits; for `text-embedding-3-large` set `EMBEDDING_DIM=1536` (the code requests shortened vectors). Changing `EMBEDDING_DIM` on an existing database requires recreating the `chunks` table.
- Ragas is pinned to `>=0.2,<0.4`; its API changes between releases, so re-test before upgrading.
