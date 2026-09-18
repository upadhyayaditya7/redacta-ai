"""Batch mode: redact every document in a folder, with optional watch loop.

``python app.py batch inbox/ --vault-pass …`` processes every supported
file (txt / pdf / png / jpg / …) once; ``--watch`` keeps polling every
N seconds so a drop-folder workflow works — new files are redacted as
they arrive.  Results land in ``<folder>_redacted/`` next to it, and an
``audit.json`` per run records what was done.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from .pipeline import PipelineReport, RedactionPipeline
from .redact import RedactionResult, redact_image, redact_pdf, redact_text
from .vault import Vault

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
TEXT_EXTS = {".txt", ".md", ".csv", ".json", ".log", ".xml"}
PDF_EXTS = {".pdf"}
SUPPORTED = IMAGE_EXTS | TEXT_EXTS | PDF_EXTS

POLL_SECONDS = 5.0


@dataclass
class BatchItem:
    src: Path
    out: Path
    report: PipelineReport | None   # None for PDF/image (report lives inside redact_*)
    result: RedactionResult | None
    error: str | None = None


def _iter_targets(folder: Path) -> list[Path]:
    """All files in *folder* (unsupported ones are surfaced with an error
    rather than silently skipped, so nothing quietly escapes redaction)."""
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and not p.name.startswith(".")
    )


def process_folder(
    folder: Path,
    pipeline: RedactionPipeline,
    vault: Vault | None = None,
    types=None,
) -> list[BatchItem]:
    """Redact every supported file in *folder* once. Returns per-file items."""
    out_dir = folder.with_name(folder.name + "_redacted")
    out_dir.mkdir(parents=True, exist_ok=True)
    items: list[BatchItem] = []
    for path in _iter_targets(folder):
        ext = path.suffix.lower()
        try:
            if ext not in SUPPORTED:
                raise ValueError(f"unsupported file type '{ext or '(none)'}' — skipped")
            if ext in PDF_EXTS:
                result, out_path = redact_pdf(path, pipeline, vault, types, out_dir=out_dir)
                items.append(BatchItem(path, Path(out_path), None, result))
            elif ext in IMAGE_EXTS:
                result, out_path = redact_image(path, pipeline, vault, types, out_dir=out_dir)
                items.append(BatchItem(path, Path(out_path), None, result))
            else:
                text = path.read_text(encoding="utf-8", errors="replace")
                report = pipeline.analyze(text, types)
                result = redact_text(text, report, vault, types)
                out_path = out_dir / (path.stem + ".redacted.txt")
                out_path.write_text(result.redacted_text, encoding="utf-8")
                items.append(BatchItem(path, out_path, report, result))
        except Exception as exc:  # noqa: BLE001 — batch must survive one bad file
            items.append(BatchItem(path, out_dir / path.name, None, None, error=str(exc)))
    return items


def watch_folder(folder: Path, pipeline: RedactionPipeline, vault: Vault | None = None,
                 types=None, poll: float = POLL_SECONDS) -> None:
    """Keep watching *folder* and process new arrivals as they appear."""
    print(f"Watching {folder} (poll every {poll:.0f}s) — Ctrl+C to stop.")
    try:
        while True:
            targets = _iter_targets(folder)
            fresh = [p for p in targets if not _already_done(p)]
            for path in fresh:
                print(f"  → {path.name}")
            if fresh:
                process_folder(folder, pipeline, vault, types)
            time.sleep(poll)
    except KeyboardInterrupt:
        print("\nWatch stopped.")


def _already_done(src: Path) -> bool:
    """A file counts as done when any redacted output for it exists."""
    out_dir = src.with_name(src.parent.name + "_redacted")
    return any(out_dir.glob(src.stem + "_redacted.*")) or any(out_dir.glob(src.stem + ".redacted.*"))
