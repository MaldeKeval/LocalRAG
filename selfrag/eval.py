from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from rich.console import Console
from rich.table import Table

from .config import Settings, ensure_dirs
from .index import VectorIndex
from .retrieval import retrieve
from .rerank import maybe_rerank


def _load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def run_eval(questions_path: Path) -> int:
    console = Console()
    settings = Settings()
    ensure_dirs(settings)

    if not questions_path.exists():
        console.print(f"[red]Eval file not found:[/red] {questions_path}")
        return 2

    index = VectorIndex(settings)
    qs = _load_jsonl(questions_path)
    if not qs:
        console.print("[yellow]No questions found.[/yellow]")
        return 0

    table = Table(title="Retrieval Eval (lightweight)")
    table.add_column("id")
    table.add_column("hit@k")
    table.add_column("top_citation")

    hits = 0
    for q in qs:
        qid = str(q.get("id") or "")
        question = str(q.get("question") or "")
        must = [Path(x).name for x in (q.get("must_cite") or [])]

        chunks = retrieve(settings, index, question)
        chunks = maybe_rerank(settings.reranker_model, question, chunks)

        cited = [Path(c.source_path).name for c in chunks]
        hit = any(m in cited for m in must) if must else bool(chunks)
        hits += 1 if hit else 0
        top = cited[0] if cited else "-"
        table.add_row(qid, "1" if hit else "0", top)

    console.print(table)
    acc = hits / len(qs)
    console.print(f"\nHit@k: {hits}/{len(qs)} = {acc:.2f}")
    return 0


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="Lightweight retrieval eval: JSONL lines with id, question, optional must_cite (PDF basenames)."
    )
    p.add_argument(
        "--questions",
        type=Path,
        required=True,
        help="Path to questions JSONL.",
    )
    args = p.parse_args()
    raise SystemExit(run_eval(args.questions))


if __name__ == "__main__":
    main()

