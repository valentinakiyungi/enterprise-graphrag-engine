"""Create PostgreSQL (pgvector) tables and Neo4j constraints, then check health."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.main import configure_logging  # noqa: E402
from app.stores.neo4j_store import Neo4jGraphStore  # noqa: E402
from app.stores.postgres import PgVectorStore  # noqa: E402


def main() -> int:
    s = get_settings()
    configure_logging(s.log_level)
    pg = PgVectorStore(s.database_url, s.embedding_dim, 1, 2)
    pg.init_schema()
    print("✓ PostgreSQL schema ready:", pg.stats())
    pg.close()
    neo = Neo4jGraphStore(s.neo4j_uri, s.neo4j_http_url, s.neo4j_user, s.neo4j_password,
                          s.neo4j_database, s.neo4j_transport)
    neo.init_schema()
    print(f"✓ Neo4j schema ready via {neo.transport}:", neo.stats())
    neo.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
