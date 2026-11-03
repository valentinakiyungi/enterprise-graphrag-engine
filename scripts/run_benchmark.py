"""Run the vector-RAG vs GraphRAG benchmark from the command line.

Usage:
    python scripts/run_benchmark.py                       # default dataset, vector vs hybrid
    python scripts/run_benchmark.py --limit 5 --no-self-correct
    python scripts/run_benchmark.py --dataset my_eval.json --modes vector hybrid graph

Writes a JSON report to reports/ and stores the run in Postgres (eval_runs).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.evaluation.benchmark import BenchmarkRunner, load_dataset  # noqa: E402
from app.main import configure_logging  # noqa: E402
from app.services import create_services  # noqa: E402


def fmt(v) -> str:
    return "—" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)


def print_summary(summary: dict) -> None:
    per_mode = summary["per_mode"]
    modes = list(per_mode)
    metrics = ["multi_hop_accuracy", "answer_accuracy", "faithfulness",
               "context_precision", "hallucination_rate", "avg_latency_ms", "avg_attempts"]
    print("\n" + "metric".ljust(22) + "".join(m.rjust(12) for m in modes))
    for m in metrics:
        print(m.ljust(22) + "".join(fmt(per_mode[mode][m]).rjust(12) for mode in modes))
    cmp = summary.get("comparison") or {}
    if cmp:
        print("\nGraphRAG (hybrid) vs pure vector RAG:")
        for k, v in cmp.items():
            if k.endswith("_pct"):
                print(f"  {k.replace('_pct', '').replace('_', ' '):<38} {fmt(v)}%")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, default=None)
    p.add_argument("--modes", nargs="+", default=["vector", "hybrid"],
                   choices=["vector", "hybrid", "graph"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--no-self-correct", action="store_true",
                   help="disable the Ragas-gated correction loop during answering")
    p.add_argument("--name", default=None)
    args = p.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)
    items = load_dataset(args.dataset)[: args.limit] if args.limit else load_dataset(args.dataset)
    svc = create_services(settings)
    try:
        run_id = svc.pg.create_eval_run(args.name, {**vars(args), "dataset": str(args.dataset)})
        summary, results = BenchmarkRunner(svc).run(
            items, args.modes, not args.no_self_correct, args.concurrency
        )
        svc.pg.finish_eval_run(run_id, summary, results)
    finally:
        svc.close()

    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = out_dir / f"benchmark_{stamp}.json"
    out.write_text(json.dumps({"run_id": run_id, "summary": summary, "results": results},
                              indent=2, default=str))
    print_summary(summary)
    print(f"\nFull report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
