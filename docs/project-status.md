# Project Status and Notes

## Status

The pipeline is implemented with:

- Incremental ingest
- LanceDB index
- Retrieval (optional hybrid BM25 and rerank)
- CLI commands: `ingest`, `status`, `ask`, `watch`, `open`
- Gradio web UI
- Local LLM providers

Optional lightweight retrieval eval:

```bash
python -m selfrag.eval --questions your_questions.jsonl
```

Each JSONL line supports:

- `id`
- `question`
- optional `must_cite` list of source basenames for simple hit@k checks

## Notes

- Ingestion is incremental: new or changed PDF/DOCX files are detected by file hash.
- Answers include citations (source filename + location range).

## Disclaimer

This project is provided **as is**, without warranty of any kind. The author(s) do not guarantee that the software will work correctly, remain available, or be free of defects. Use it at your own risk. The author(s) are not responsible for any damage, data loss, system failures, or other issues that may result from using this software.
