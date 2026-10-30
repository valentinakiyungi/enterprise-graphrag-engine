# Deployment

The service needs three things: the API container, PostgreSQL with the `pgvector` extension, and Neo4j, plus an LLM and embedding provider.

## Option A: run it free on your own computer

Everything runs locally at no cost: Docker runs the databases and the API, and Ollama runs the models.

1. Install Ollama (https://ollama.com), then pull the models:
   ```
   ollama pull qwen2.5:7b
   ollama pull nomic-embed-text
   ```
   `qwen2.5:7b` needs roughly 5 GB of disk and 8 GB+ of RAM. On a small machine try `llama3.2:3b`; extraction quality will be lower.
2. `cp env.free.example .env`, then replace the three `change-me` values with random strings (`openssl rand -hex 12`).
3. `docker compose up --build`

Local models are slower and less accurate than hosted ones. If `POST /ingest` reports `extraction_failures`, set `STRUCTURED_OUTPUT_METHOD=function_calling` in `.env` and restart. Ragas scoring with a small model is slow and can fail on some answers; failures are returned in the `evaluation.error` field instead of breaking the query.

## Option B: one VPS with Docker Compose

Any Linux VPS with 2 GB+ RAM works (Neo4j is the hungry part). Use an OpenAI key (`.env.example`) or any OpenAI-compatible provider via `OPENAI_BASE_URL`; a VPS can run Ollama too if it has enough RAM.

```bash
curl -fsSL https://get.docker.com | sh
git clone https://github.com/valentinakiyungi/enterprise-graphrag-engine.git
cd enterprise-graphrag-engine
cp .env.example .env
nano .env
docker compose up -d --build
curl localhost:8000/ready
```

Use URL-safe characters (letters and digits) in `POSTGRES_PASSWORD`. Database ports are bound to `127.0.0.1` only. Put the API behind HTTPS before exposing it, for example with Caddy:

```
api.example.com {
    reverse_proxy localhost:8000
}
```

Set `API_KEY` so strangers cannot use your LLM quota. Updates: `git pull && docker compose up -d --build`. Data persists in the `pgdata` and `neo4jdata` volumes; back them up.

## Option C: free hosting (small demos only)

Checked against the providers' own pages on 30 Oct 2026; free tiers change, so re-check before relying on them.

| Piece | Free option | Limits to know |
|---|---|---|
| Postgres + pgvector | [Neon](https://neon.com) Free plan | pgvector supported; 1 GB storage per project; compute suspends after 5 minutes idle |
| Neo4j | [Neo4j AuraDB Free](https://neo4j.com/cloud/platform/aura-graph-database/) | 200,000 nodes, 400,000 relationships, 1 GB; no uptime guarantee |
| API hosting | [Render](https://render.com) free web service | spins down after 15 min idle (about 1 minute to wake); 750 free hours per month; Render says free instances are not for production |

Notes:
- Skip Render's own free Postgres: Render deletes it 30 days after creation. Use Neon instead.
- **Ollama cannot run on these free hosts.** The deployed API needs a hosted LLM provider. Use any OpenAI-compatible provider that has a free tier (for example Google's Gemini API or Groq) by setting `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `CHAT_MODEL`, `EMBEDDING_MODEL` and `EMBEDDING_DIM`. I have not verified any provider's current free limits or endpoint details; check them yourself. Ingestion and evaluation make many LLM calls per document, so expect free rate limits to slow or stop large jobs.
- Render's page does not state the RAM of free instances. The Ragas dependency stack is heavy, so the container may run out of memory; if it does, use Option B.

Steps:
1. Create a Neon project, copy its connection string (it includes `sslmode=require`) and use it as `POSTGRES_DSN`.
2. Create an AuraDB Free instance and save the URI (`neo4j+s://...`), username and password. Use the database name shown in the Aura console for `NEO4J_DATABASE` if it is not `neo4j`.
3. In Render choose New, then Blueprint, select this repo (it reads `render.yaml`), and fill in the environment variables. Render generates `API_KEY`.
4. Check `https://<your-service>.onrender.com/ready`.

## After deploying

1. `POST /ingest` with your documents.
2. `POST /query` with `"evaluate": true` to confirm Ragas scoring works with your model.
3. Run the benchmark on your own data before quoting any accuracy or hallucination numbers.

## Notes

- Costs with a paid provider: ingestion makes one embedding call and one extraction call per chunk; `evaluate: true` adds several LLM calls per query. Leave `EVALUATE_BY_DEFAULT=false` for latency-sensitive traffic and sample instead.
- pgvector's HNSW index supports up to 2000 dimensions. `EMBEDDING_DIM` must match your embedding model (1536 for `text-embedding-3-small`, 768 for `nomic-embed-text`). Changing it on an existing database requires recreating the `chunks` table.
- Ragas is pinned to `>=0.2,<0.4`; its API changes between releases, so re-test before upgrading.
