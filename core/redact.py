"""Redaction engine: text, PDF and images.

Every redacted value whose entity type defaults to ``Action.VAULT`` is
encrypted into the local vault and replaced by a ``«RDCT-…»`` token;
irreversible types are masked permanently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .entities import Action, EntityType, spec
from .pipeline import PipelineReport
from .vault import Vault


@dataclass
class RedactionResult:
    redacted_text: str = ""
    report: PipelineReport | None = None
    redacted_count: int = 0
    vaulted_count: int = 0
    token_map: dict[str, str] = field(default_factory=dict)  # token -> original


def _mask(entity_type: EntityType, text: str) -> str:
    return "█" * max(len(text), 4)


def redact_text(
    text: str,
    report: PipelineReport,
    vault: Vault | None = None,
    types: list[EntityType] | None = None,
) -> RedactionResult:
    """Replace detected spans in *text* using *report*'s offsets."""
    selected = report.spans
    if types is not None:
        wanted = set(types)
        selected = [s for s in selected if s.entity_type in wanted]

    result = RedactionResult(report=report)
    out: list[tuple[int, int, str]] = []
    for span in selected:
        s = spec(span.entity_type)
        if s.action is Action.VAULT and vault is not None:
            token = vault.store(span.text)
            replacement = token
            result.token_map[token] = span.text
            result.vaulted_count += 1
        else:
            replacement = _mask(span.entity_type, span.text)
        out.append((span.start, span.end, replacement))
        result.redacted_count += 1

    redacted = text
    for start, end, replacement in sorted(out, key=lambda t: -t[0]):
        redacted = redacted[:start] + replacement + redacted[end:]
    result.redacted_text = redacted
    return result


def redact_pdf(
    pdf_path,
    pipeline,
    vault: Vault | None = None,
    types: list[EntityType] | None = None,
    out_dir=None,
) -> tuple[RedactionResult, str]:
    """Extract text from a PDF, redact it, and render a clean PDF.

    Returns (result, output_path).  The output keeps the document text
    (redacted) and notes which engine ran; scans fall back to OCR when
    tesseract is installed.  ``out_dir`` relocates the output (batch mode).
    """
    from pypdf import PdfReader

    stem = Path(str(pdf_path)).stem

    reader = PdfReader(str(pdf_path))
    pages = [(page.extract_text() or "") for page in reader.pages]
    full_text = "\n\n".join(pages)

    report = pipeline.analyze(full_text, types)
    result = redact_text(full_text, report, vault, types)

    base = Path(out_dir) if out_dir else Path(str(pdf_path)).parent
    base.mkdir(parents=True, exist_ok=True)
    out_path = str(base / (stem + "_redacted.pdf"))
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas as pdfcanvas

        c = pdfcanvas.Canvas(out_path, pagesize=A4)
        width, height = A4
        y = height - 50
        for line in result.redacted_text.splitlines():
            if y < 50:
                c.showPage()
                y = height - 50
            c.setFont("Courier", 8)
            c.drawString(40, y, line[:110])
            y -= 11
        c.save()
    except ImportError:
        txt_path = str(base / (stem + "_redacted.txt"))
        with open(txt_path, "w", encoding="utf-8") as fh:
            fh.write(result.redacted_text)
        out_path = txt_path
    return result, out_path


def redact_image(
    image_path,
    pipeline,
    vault: Vault | None = None,
    types: list[EntityType] | None = None,
    out_dir=None,
) -> tuple[RedactionResult, str]:
    """OCR an image, detect PII, and paint black boxes over it."""
    try:
        import pytesseract
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError(
            "Image redaction needs Pillow + pytesseract (and the Tesseract binary)."
        ) from exc

    image = Image.open(image_path).convert("RGB")
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    words: list[tuple[int, int, int, int, str]] = []
    for i, w in enumerate(data["text"]):
        if w.strip() and int(data["conf"][i]) > 30:
            words.append(
                (data["left"][i], data["top"][i], data["width"][i], data["height"][i], w)
            )

    # Rebuild a text stream whose word order matches the OCR boxes, so
    # spans found in text can be mapped back onto pixel coordinates.
    parts: list[str] = []
    boxes: list[tuple[int, int, int, int, int, int]] = []
    cursor = 0
    for x, y, w, h, word in words:
        parts.append(word)
        boxes.append((cursor, cursor + len(word), x, y, w, h))
        cursor += len(word) + 1  # +1 for the joining space
        parts.append(" ")
    text = "".join(parts)

    report = pipeline.analyze(text, types)
    result = redact_text(text, report, vault, types)

    draw = ImageDraw.Draw(image)
    for span in report.spans:
        for start, end, x, y, w, h in boxes:
            if start < span.end and span.start < end:  # overlap
                draw.rectangle([x - 2, y - 2, x + w + 2, y + h + 2], fill="black")

    base = Path(out_dir) if out_dir else Path(str(image_path)).parent
    base.mkdir(parents=True, exist_ok=True)
    out_path = str(base / (Path(str(image_path)).stem + "_redacted.png"))
    image.save(out_path)
    return result, out_path
