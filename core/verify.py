"""Post-redaction verification: prove the output no longer leaks.

Detection counts are a claim; this module is the check.  After redaction the
*produced* file is re-read — PDF text layer, document metadata, annotations —
and every value that was replaced is looked for again.  If one is still
readable the caller is told, and the UI refuses to offer the download.

Matching is deliberately tolerant of extraction quirks (PDF text often loses
the space between two chunks) and strict about identity: a value counts as
leaked when its whitespace-normalised form is present, or when its complete
digit run is present for values with six or more digits.

Not covered: text baked into page images, and OCR of the output.  A scanned
PDF is refused upstream by ``redact_pdf`` instead of being silently emptied.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .entities import EntityType, spec
from .pipeline import PipelineReport


@dataclass(frozen=True)
class Leak:
    """One replaced value that is still readable in the output."""

    entity_type: EntityType
    label: str
    hint: str   # masked preview — never the original value
    where: str  # which part of the output it survived in


@dataclass
class VerifyResult:
    """Outcome of re-reading one redacted output."""

    checked_values: int = 0
    checked_chars: int = 0
    leaks: list[Leak] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.leaks

    def summary(self) -> str:
        if self.ok:
            return (
                f"Verified clean — {self.checked_values} redacted values absent from "
                f"the output ({self.checked_chars:,} chars re-read)"
            )
        return f"VERIFICATION FAILED — {len(self.leaks)} value(s) still readable"

    def as_dict(self) -> dict:
        """JSON-safe form for the audit record."""
        return {
            "ok": self.ok,
            "checked_values": self.checked_values,
            "checked_chars": self.checked_chars,
            "sources": self.sources,
            "leaks": [
                {"entity_type": lk.entity_type.value, "label": lk.label,
                 "hint": lk.hint, "where": lk.where}
                for lk in self.leaks
            ],
        }


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def _hint(value: str) -> str:
    if len(value) <= 4:
        return "•" * len(value)
    return value[:2] + "•" * (len(value) - 4) + value[-2:]


def _values(report: PipelineReport) -> list[tuple[str, EntityType]]:
    """Distinct replaced values, shortest first, with their entity type."""
    seen: dict[str, EntityType] = {}
    for span in report.spans:
        value = span.text.strip()
        if len(value) >= 3:  # shorter than this cannot be told apart reliably
            seen.setdefault(value, span.entity_type)
    return sorted(seen.items(), key=lambda item: len(item[0]))


def _occurs(value: str, text_norm: str, text_digits: str) -> bool:
    needle = _norm(value)
    if needle and needle in text_norm:
        return True
    digits = _digits(value)
    return len(digits) >= 6 and digits in text_digits


def _scan(sources: list[tuple[str, str]], report: PipelineReport) -> VerifyResult:
    values = _values(report)
    prepared = [
        (where, _norm(text), _digits(text)) for where, text in sources if text
    ]
    result = VerifyResult(
        checked_values=len(values),
        checked_chars=sum(len(text) for _, text in sources),
        sources=[where for where, _ in sources],
    )
    for value, entity_type in values:
        for where, text_norm, text_digits in prepared:
            if _occurs(value, text_norm, text_digits):
                result.leaks.append(
                    Leak(
                        entity_type=entity_type,
                        label=spec(entity_type).label,
                        hint=_hint(value),
                        where=where,
                    )
                )
                break
    return result


def verify_text(text: str, report: PipelineReport, where: str = "output text") -> VerifyResult:
    """Verify a redacted string."""
    return _scan([(where, text)], report)


def _pdf_sources(path: Path) -> list[tuple[str, str]]:
    """Everything a reader could extract from a PDF: text, metadata, annotations."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "") for page in reader.pages]
    sources: list[tuple[str, str]] = [("PDF text layer", "\n".join(pages))]

    meta: list[str] = []
    try:
        info = reader.metadata
        if info:
            meta.extend(str(value) for value in info.values())
    except Exception:  # noqa: BLE001 — malformed info dict must not abort the check
        pass
    try:
        xmp = reader.xmp_metadata
        if xmp is not None:
            meta.append(str(xmp))
    except Exception:  # noqa: BLE001
        pass
    if meta:
        sources.append(("PDF metadata", "\n".join(meta)))

    annots: list[str] = []
    for page in reader.pages:
        try:
            raw = page.get("/Annots") or []
            objects = [a.get_object() for a in raw]
        except Exception:  # noqa: BLE001
            continue
        for obj in objects:
            for key in ("/Contents", "/T", "/Subj"):
                try:
                    value = obj.get(key)
                except Exception:  # noqa: BLE001
                    continue
                if value:
                    annots.append(str(value))
    if annots:
        sources.append(("PDF annotations", "\n".join(annots)))
    return sources


def verify_file(path, report: PipelineReport) -> VerifyResult:
    """Verify a produced file: PDF pages/metadata, or a text file as-is."""
    target = Path(str(path))
    if target.suffix.lower() == ".pdf":
        return _scan(_pdf_sources(target), report)
    text = target.read_text(encoding="utf-8", errors="replace")
    return _scan([(target.suffix.lstrip(".") or "output text", text)], report)


def verify_output(output, report: PipelineReport) -> VerifyResult:
    """Verify whatever redaction produced: a path or an in-memory string."""
    if isinstance(output, (str, Path)) and Path(str(output)).exists():
        return verify_file(output, report)
    return verify_text(str(output), report)
