"""Bulk-ingest a folder of .md / .txt / .json documents.

Usage:
    python scripts/ingest.py data/sample_corpus
    python scripts/ingest.py ./docs --glob "**/*.md"

JSON files may hold one object or a list of {"title", "content", "source", "metadata"}.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.main import configure_logging  # noqa: E402
from app.services import create_services  # noqa: E402


def iter_documents(root: Path, pattern: str):
    paths = [root] if root.is_file() else sorted(root.glob(pattern))
    for path in paths:
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            for doc in data if isinstance(data, list) else [data]:
                yield doc["title"], doc["content"], doc.get("source", str(path)), doc.get("metadata", {})
        elif path.suffix.lower() in {".md", ".txt", ".markdown"}:
            text = path.read_text(encoding="utf-8")
            first = text.lstrip().splitlines()[0] if text.strip() else path.stem
            title = first.lstrip("# ").strip() if first.startswith("#") else path.stem.replace("_", " ")
            yield title, text, str(path), {"filename": path.name}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path)
    parser.add_argument("--glob", default="**/*", help="glob pattern when path is a folder")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)
    svc = create_services(settings)
    ok = failed = 0
    try:
        for title, content, source, meta in iter_documents(args.path, args.glob):
            try:
                r = svc.ingestion.ingest(title, content, source, meta)
                ok += 1
                print(f"✓ {title}: {r.chunks} chunks, {r.entities} entities, "
                      f"{r.relations} relations ({r.seconds}s)")
            except Exception as exc:
                failed += 1
                print(f"✗ {title}: {exc}", file=sys.stderr)
    finally:
        svc.close()
    print(f"\nIngested {ok} document(s), {failed} failure(s).")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
