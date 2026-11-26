"""Redaction engine: text, PDF and images.

Every redacted value whose entity type defaults to ``Action.VAULT`` is
encrypted into the local vault and replaced by a ``«RDCT-…»`` token;
irreversible types are masked permanently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from .entities import Action, EntityType, spec
from .pipeline import PipelineReport
from .vault import Vault

# Text-layer scrub character.  ASCII on purpose: it is encodable in every
# PDF font and string encoding (the visual mask is the black box).
_SCRUB_CHAR = "*"
# Generous advance-per-character estimate for drawing boxes.  Over-covering
# is safe; under-covering would leave PII visible.
_CHAR_WIDTH = 0.6


@dataclass
class RedactionResult:
    redacted_text: str = ""
    report: PipelineReport | None = None
    redacted_count: int = 0
    vaulted_count: int = 0
    token_map: dict[str, str] = field(default_factory=dict)  # token -> original
    metadata_dropped: int = 0      # document-info entries that named a redacted value
    annotations_removed: int = 0   # page annotations dropped from the output


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


def _page_text_with_positions(page) -> tuple[str, list[tuple[int, int, float, float, float]]]:
    """Extract a page's text plus the position of every extracted chunk.

    Returns ``(text, chunks)`` where each chunk is
    ``(start, end, x, y, font_size)``; ``x``/``y`` are the chunk's baseline
    origin in page user space.  ``text`` is built from the same chunks, so
    offsets into it always line up with the recorded positions.
    """
    pieces: list[str] = []
    chunks: list[tuple[int, int, float, float, float]] = []
    cursor = 0

    def visitor(text, cm, tm, font_dict, font_size):
        nonlocal cursor
        if not text:
            return
        m = cm if cm is not None else (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        x = m[0] * tm[4] + m[2] * tm[5] + m[4]
        y = m[1] * tm[4] + m[3] * tm[5] + m[5]
        scale = math.hypot(m[0], m[1]) or 1.0
        size = abs(float(font_size)) * scale if font_size else 10.0 * scale
        pieces.append(text)
        chunks.append((cursor, cursor + len(text), float(x), float(y), size))
        cursor += len(text)

    page.extract_text(visitor_text=visitor)
    return "".join(pieces), chunks


def _page_marks(spans, chunks, tokens: dict[str, str]) -> list[list]:
    """Turn page-local spans into ``[x, y, w, h, label]`` black boxes."""
    marks: list[list] = []
    for span, local_start, local_end in spans:
        label = tokens.get(span.text)
        for start, end, x, y, size in chunks:
            lo, hi = max(local_start, start), min(local_end, end)
            if lo >= hi:
                continue
            char_w = max(size, 1.0) * _CHAR_WIDTH
            marks.append([
                x + (lo - start) * char_w,
                y - 0.22 * size,
                (hi - lo) * char_w,
                1.3 * size,
                label,
            ])
            label = None  # draw the token once, on the first covered chunk
    return marks


def _overlay_reader(page, marks: list[list]):
    """Render the boxes onto a same-size page that can be merged on top."""
    from io import BytesIO

    from pypdf import PdfReader
    from reportlab.pdfbase.pdfmetrics import stringWidth
    from reportlab.pdfgen import canvas as pdfcanvas

    box = page.mediabox
    width, height = float(box.width), float(box.height)

    buffer = BytesIO()
    canvas = pdfcanvas.Canvas(buffer, pagesize=(width, height))
    for x, y, w, h, label in marks:
        size = min(7.0, h * 0.72)
        label_width = stringWidth(label, "Helvetica", size) if label else 0.0
        # Widen the box rather than shrink the token: a reader has to be
        # able to read the token back out of a vaulted document.
        canvas.setFillColorRGB(0, 0, 0)
        canvas.rect(x - 1.0, y, max(w, label_width) + 2.0, h, stroke=0, fill=1)
        if label:
            canvas.setFillColorRGB(1, 1, 1)
            canvas.setFont("Helvetica", size)
            canvas.drawString(x + 0.5, y + h * 0.26, label)
    canvas.save()
    buffer.seek(0)
    return PdfReader(buffer)


def _scrub_literal(literal, values: tuple[str, ...]):
    """Mask redacted values inside one PDF string literal."""
    from pypdf.generic import ByteStringObject, TextStringObject

    is_text = isinstance(literal, TextStringObject)
    is_bytes = isinstance(literal, ByteStringObject)
    if not (is_text or is_bytes):
        return literal
    try:
        original = str(literal) if is_text else literal.original_bytes.decode("latin-1")
    except Exception:  # noqa: BLE001 — undecodable literal: leave it alone
        return literal
    masked = original
    for value in values:
        if value in masked:
            masked = masked.replace(value, _SCRUB_CHAR * len(value))
    if masked == original:
        return literal
    return TextStringObject(masked) if is_text else ByteStringObject(masked.encode("latin-1", "replace"))


def _scrub_text_layer(page, pdf, values: tuple[str, ...]) -> None:
    """Mask *values* inside the page's text operators.

    The black boxes hide the glyphs, but the original characters would still
    be copy-pasteable out of the PDF.  Rewriting the string operands keeps
    the layout (only the characters change) and closes that hole.
    """
    if not values:
        return
    contents = page.get_contents()
    if contents is None:
        return

    from pypdf.generic import ContentStream, TextStringObject

    stream = ContentStream(contents, pdf)
    changed = False
    new_operations = []
    for operands, operator in stream.operations:
        if operator in (b"Tj", b"'", b'"') and operands:
            operands = list(operands)
            masked = _scrub_literal(operands[-1], values)
            if masked is not operands[-1]:
                operands[-1] = masked
                changed = True
        elif operator == b"TJ" and operands:
            array = operands[0]
            if hasattr(array, "__setitem__"):
                for i, item in enumerate(array):
                    if isinstance(item, TextStringObject):
                        masked = _scrub_literal(item, values)
                        if masked is not item:
                            array[i] = masked
                            changed = True
        new_operations.append((operands, operator))

    if changed:
        stream.operations = new_operations
        page.replace_contents(stream)


# Descriptive /Info fields are dropped wholesale: they are free text written by
# whoever produced the document, so a "Subject: statement for <holder>" line
# leaks a name that no text rule can judge.  Timestamps carry no identity.
_INFO_KEEP = ("/CreationDate", "/ModDate")


def _clean_metadata(writer, reader) -> int:
    """Give the redacted copy a metadata dict that names nobody.

    A fresh ``PdfWriter`` does not inherit ``/Info``, but a copied or XMP
    packet can travel with the file, so the output is given an explicit dict:
    producer, plus the original timestamps.  Everything descriptive is
    removed rather than masked — a half-masked author field looks sanitised
    and still identifies the document.
    """
    dropped = 0
    clean: dict[str, str] = {}
    try:
        info = dict(reader.metadata or {})
    except Exception:  # noqa: BLE001 — a malformed info dict must not block redaction
        info = {}
    for key, value in info.items():
        text = "" if value is None else str(value)
        if not text:
            continue
        if str(key) in _INFO_KEEP:
            clean[str(key)] = text
        else:
            dropped += 1
    clean["/Producer"] = "Redacta AI"
    try:
        writer.add_metadata(clean)
    except Exception:  # noqa: BLE001 — never fail a redaction over cosmetics
        pass
    try:
        writer.xmp_metadata = None
    except Exception:  # noqa: BLE001
        pass
    return dropped


def _strip_page_extras(page) -> int:
    """Remove annotations and page-level XMP from an output page.

    Annotations are dropped whole rather than edited: their text, author and
    appearance stream all survive an edit, and an annotation renders *above*
    the black boxes.  XMP is dropped because a page can carry its own packet.
    Returns the number of annotations removed.
    """
    from pypdf.generic import NameObject

    removed = 0
    raw = page.get("/Annots")
    if raw is not None:
        try:
            removed = len(raw)
        except TypeError:
            removed = 1
        try:
            del page[NameObject("/Annots")]
        except Exception:  # noqa: BLE001 — leave the page intact if this fails
            removed = 0
    if "/Metadata" in page:
        try:
            del page[NameObject("/Metadata")]
        except Exception:  # noqa: BLE001
            pass
    return removed


def _wrap_text(text: str, width: int = 110) -> str:
    """Wrap long lines (used only by the no-reportlab text fallback)."""
    import textwrap

    lines: list[str] = []
    for line in text.splitlines():
        lines.extend(textwrap.wrap(line, width=width, break_long_words=True) or [""])
    return "\n".join(lines)


def redact_pdf(
    pdf_path,
    pipeline,
    vault: Vault | None = None,
    types: list[EntityType] | None = None,
    out_dir=None,
) -> tuple[RedactionResult, str]:
    """Redact a PDF in place of its layout: the original pages are kept and
    every detected span is covered with a black box, so formatting, images
    and untruncated text survive.  Token text is printed white inside the box
    when the value went to the vault, so the document stays restorable.

    Returns (result, output_path).  ``out_dir`` relocates the output (batch
    mode).  A PDF with no text layer (a scan) is refused rather than turned
    into a silently empty file.
    """
    from pypdf import PdfReader, PdfWriter

    src = Path(str(pdf_path))
    reader = PdfReader(str(src))

    page_texts: list[str] = []
    page_chunks: list[list[tuple[int, int, float, float, float]]] = []
    for page in reader.pages:
        text, chunks = _page_text_with_positions(page)
        page_texts.append(text)
        page_chunks.append(chunks)

    starts: list[int] = []
    cursor = 0
    for text in page_texts:
        starts.append(cursor)
        cursor += len(text) + 2  # "\n\n" separator
    full_text = "\n\n".join(page_texts)

    if not full_text.strip():
        raise RuntimeError(
            "No extractable text in this PDF — it is very likely a scan. "
            "OCR it first, or redact the page images instead."
        )

    report = pipeline.analyze(full_text, types)
    result = redact_text(full_text, report, vault, types)

    selected = report.spans
    if types is not None:
        wanted = set(types)
        selected = [s for s in selected if s.entity_type in wanted]
    tokens = {original: token for token, original in result.token_map.items()}

    base = Path(out_dir) if out_dir else src.parent
    base.mkdir(parents=True, exist_ok=True)
    out_path = base / (src.stem + "_redacted.pdf")

    try:
        import reportlab  # noqa: F401
    except ImportError:
        txt_path = base / (src.stem + "_redacted.txt")
        txt_path.write_text(_wrap_text(result.redacted_text), encoding="utf-8")
        return result, str(txt_path)

    writer = PdfWriter()
    try:
        for page, text, chunks, start in zip(reader.pages, page_texts, page_chunks, starts):
            end = start + len(text)
            local = [
                (span, max(0, span.start - start), min(len(text), span.end - start))
                for span in selected
                if span.start < end and span.end > start
            ]
            # Mask only the page's own values: a common short value must not
            # shred every other page's text.  Very short or multi-line values
            # are skipped here — the box still covers them visually.
            values = tuple(sorted(
                {s.text for s, _, _ in local if len(s.text) >= 3 and "\n" not in s.text},
                key=len,
                reverse=True,
            ))
            # pypdf 7 only allows rewriting a page's content once the page
            # belongs to the writer, so attach it first, then scrub and merge.
            writer.add_page(page)
            target = writer.pages[-1]
            result.annotations_removed += _strip_page_extras(target)
            _scrub_text_layer(target, writer, values)
            marks = _page_marks(local, chunks, tokens)
            if marks:
                overlay = _overlay_reader(target, marks)
                target.merge_page(overlay.pages[0])
        result.metadata_dropped = _clean_metadata(writer, reader)
        with open(out_path, "wb") as fh:
            writer.write(fh)
    finally:
        writer.close()
    return result, str(out_path)


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
