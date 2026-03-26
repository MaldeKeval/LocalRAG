from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from .config import Settings, ensure_dirs, load_settings
from .index import VectorIndex
from .ingest import ingest_pdf, iter_pdfs, load_manifest, save_manifest, update_manifest_for_pdf
from .llm import generate
from .llm_probe import probe_llm
from .prompting import build_grounded_prompt
from .retrieval import retrieve
from .rerank import maybe_rerank
from .util import manifest_key_for_pdf, sha256_file


app = typer.Typer(add_completion=False, help="SelfRag: fully local PDF RAG (3GPP).")
console = Console()

def _safe_for_console(s: str) -> str:
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        return s.encode(enc, errors="replace").decode(enc, errors="replace")
    except Exception:
        return s.encode("utf-8", errors="replace").decode("utf-8", errors="replace")


@app.command()
def ingest(pdf_dir: Optional[Path] = typer.Argument(None, help="Folder containing PDFs.")) -> None:
    """
    Ingest PDFs from a folder into the local index (incremental).
    """
    settings = load_settings()
    ensure_dirs(settings)

    target = pdf_dir or settings.pdf_dir
    if not target.exists():
        raise typer.BadParameter(f"PDF directory not found: {target}")

    index = VectorIndex(settings)
    total_chunks = 0

    target_root = target.resolve()
    manifest = load_manifest(settings.index_dir, target_root)
    current_keys = {manifest_key_for_pdf(p, target_root) for p in iter_pdfs(target)}

    # Drop manifest + index rows for PDFs no longer under the ingest folder.
    pruned = False
    for k in list(manifest.keys()):
        if k not in current_keys:
            pruned = True
            entry = manifest.get(k) or {}
            doc_id = str(entry.get("doc_id") or "").strip()
            if doc_id:
                index.delete_doc_id(doc_id)
            manifest.pop(k, None)

    to_process: list[Path] = []
    for pdf in iter_pdfs(target):
        key = manifest_key_for_pdf(pdf, target_root)
        file_hash = sha256_file(pdf)
        prev = manifest.get(key)
        if not prev or prev.get("file_hash") != file_hash:
            to_process.append(pdf)

    if not to_process and not pruned:
        console.print("[green]No new/changed PDFs found.[/green]")
        return

    for pdf in to_process:
        file_hash = sha256_file(pdf)
        docs = ingest_pdf(settings, pdf, file_hash=file_hash, pdf_root=target_root)
        if not docs:
            console.print(f"[yellow]No text extracted:[/yellow] {pdf}")
            continue

        index.upsert_docs(docs)
        update_manifest_for_pdf(manifest, pdf, target_root, file_hash=file_hash, doc_id=docs[0].doc_id)
        total_chunks += len(docs)
        console.print(f"[cyan]Ingested[/cyan] {pdf.name} ({len(docs)} chunks)")

    save_manifest(settings.index_dir, manifest)
    console.print(f"[green]Done.[/green] Added/updated chunks: {total_chunks}")


@app.command()
def status() -> None:
    """
    Show index status.
    """
    settings = load_settings()
    ensure_dirs(settings)
    manifest = load_manifest(settings.index_dir, settings.pdf_dir.resolve())
    index = VectorIndex(settings)

    table = Table(title="SelfRag Status")
    table.add_column("Item")
    table.add_column("Value")
    table.add_row("PDF folder", str(settings.pdf_dir))
    table.add_row("Index folder", str(settings.index_dir))
    table.add_row("Tracked PDFs", str(len(manifest)))
    table.add_row("Documents (unique doc_id)", str(index.unique_docs()))
    table.add_row("Chunks", str(index.count_chunks()))
    table.add_row("Embedding model", settings.embedding_model)
    table.add_row("LLM provider", settings.llm_provider)
    table.add_row("LLM model", settings.ollama_model if settings.llm_provider == "ollama" else settings.openai_compat_model)
    probe = probe_llm(settings)
    table.add_row("LLM check", probe.status_line)
    console.print(table)


@app.command()
def ask(
    question: str,
    top_k: Optional[int] = typer.Option(None, help="Override top_k for retrieval."),
    hybrid: Optional[bool] = typer.Option(None, help="Enable BM25 hybrid retrieval for this query."),
    debug_context: bool = typer.Option(False, help="Print retrieved context blocks."),
    no_llm: bool = typer.Option(False, help="Skip generation; only show citations/context."),
    max_context_chars: int = typer.Option(1600, help="Max characters to show per retrieved chunk in debug output."),
) -> None:
    """
    Ask a question grounded in your ingested PDFs.
    """
    settings = load_settings()
    ensure_dirs(settings)
    if top_k is not None:
        settings.top_k = int(top_k)
    if hybrid is not None:
        settings.hybrid_bm25 = bool(hybrid)

    index = VectorIndex(settings)
    chunks = retrieve(settings, index, question)
    chunks = maybe_rerank(
        settings.reranker_model,
        question,
        chunks,
        device=getattr(settings, "reranker_device", "auto"),
    )

    if not no_llm:
        probe = probe_llm(settings)
        if not probe.ok:
            console.print(
                Panel(
                    Text(probe.message),
                    title="LLM warning",
                    border_style="yellow",
                    expand=False,
                )
            )
        prompt = build_grounded_prompt(question, chunks)
        result = generate(settings, prompt)
        answer_text = _safe_for_console(result.text or "").strip()
        if not answer_text:
            answer_text = "(empty response)"
        console.print()
        console.print(
            Panel(
                Text(answer_text),
                title="Answer",
                border_style="green",
                expand=False,
            )
        )
    else:
        console.print("[yellow]LLM skipped.[/yellow] Showing retrieved evidence only.")
    if chunks:
        cite = Table(title="Citations", show_header=True, header_style="bold")
        cite.add_column("#", justify="right", width=3)
        cite.add_column("PDF", overflow="fold")
        cite.add_column("Pages", justify="right", width=10)
        for i, c in enumerate(chunks, start=1):
            cite.add_row(str(i), Path(c.source_path).name, f"{c.page_start}-{c.page_end}")
        console.print()
        console.print(cite)

    if debug_context and chunks:
        console.print()
        console.print("[bold]Retrieved context[/bold]")
        for i, c in enumerate(chunks, start=1):
            header = f"[{i}] {Path(c.source_path).name} p.{c.page_start}-{c.page_end}"
            body = _safe_for_console(c.chunk_text.strip())
            if max_context_chars > 0 and len(body) > max_context_chars:
                body = body[: max_context_chars - 1].rstrip() + "…"
            panel = Panel(
                Text(body),
                title=header,
                border_style="cyan",
                expand=False,
            )
            console.print(panel)


class _WatchHandler(FileSystemEventHandler):
    def __init__(self, settings: Settings, pdf_root: Path):
        super().__init__()
        self.settings = settings
        self.pdf_root = pdf_root.resolve()
        self.index = VectorIndex(settings)
        self.manifest = load_manifest(settings.index_dir, self.pdf_root)

    def _maybe_ingest(self, path: Path) -> None:
        if path.suffix.lower() != ".pdf":
            return
        if not path.exists():
            return

        file_hash = sha256_file(path)
        key = manifest_key_for_pdf(path, self.pdf_root)
        prev = self.manifest.get(key)
        if prev and prev.get("file_hash") == file_hash:
            return

        docs = ingest_pdf(self.settings, path, file_hash=file_hash, pdf_root=self.pdf_root)
        if not docs:
            return
        self.index.upsert_docs(docs)
        update_manifest_for_pdf(self.manifest, path, self.pdf_root, file_hash=file_hash, doc_id=docs[0].doc_id)
        save_manifest(self.settings.index_dir, self.manifest)
        console.print(f"[cyan]Auto-ingested[/cyan] {path.name} ({len(docs)} chunks)")

    def on_created(self, event):
        if event.is_directory:
            return
        self._maybe_ingest(Path(event.src_path))

    def on_modified(self, event):
        if event.is_directory:
            return
        self._maybe_ingest(Path(event.src_path))


@app.command()
def watch(pdf_dir: Optional[Path] = typer.Argument(None, help="Folder to watch for PDFs.")) -> None:
    """
    Watch a folder and ingest PDFs on change.
    """
    settings = load_settings()
    ensure_dirs(settings)
    target = pdf_dir or settings.pdf_dir
    target.mkdir(parents=True, exist_ok=True)

    console.print(f"[green]Watching[/green] {target} (Ctrl+C to stop)")
    handler = _WatchHandler(settings, target.resolve())
    observer = Observer()
    observer.schedule(handler, str(target), recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


@app.command()
def open(path: Path, page: Optional[int] = typer.Option(None, help="Page number to show (manual navigation).")) -> None:
    """
    Open a PDF in the default viewer.
    """
    if not path.exists():
        raise typer.BadParameter(f"File not found: {path}")
    os.startfile(str(path))  # type: ignore[attr-defined]
    if page is not None:
        console.print(f"Opened. Please navigate to page {page}.")
