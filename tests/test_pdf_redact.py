"""Tests for PDF redaction: layout kept, no truncation, no silent empty output."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pypdf import PdfReader
from reportlab.pdfgen import canvas as pdfcanvas

from core.pipeline import RedactionPipeline
from core.redact import redact_pdf
from core.vault import Vault


def _write_pdf(path: Path, pages: list[list[str]]) -> None:
    canvas = pdfcanvas.Canvas(str(path), pagesize=(595, 842))
    for lines in pages:
        y = 800
        canvas.setFont("Helvetica", 10)
        for line in lines:
            canvas.drawString(40, y, line)
            y -= 14
        canvas.showPage()
    canvas.save()


def _text(path: Path) -> str:
    return "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)


def _vault(tmp_path: Path) -> Vault:
    vault = Vault(path=tmp_path / "vault.json")
    vault.create("test-pass-123")
    vault.load("test-pass-123")
    return vault


def test_redaction_keeps_the_page_layout_and_never_truncates(tmp_path: Path) -> None:
    """Regression: the output used to be a re-flowed text dump cut at 110 chars."""
    tail = "SENTINEL-END-OF-THE-LINE"
    long_line = ("Employee handbook " + "lorem ipsum " * 30).strip() + " " + tail
    src = tmp_path / "doc.pdf"
    _write_pdf(src, [["Statement of account", long_line, "Contact: jane.doe@example.com"]])

    result, out = redact_pdf(src, RedactionPipeline(use_ner=False), None)
    out = Path(out)

    assert out.suffix == ".pdf" and out.exists()
    assert len(PdfReader(str(out)).pages) == 1
    assert result.redacted_count >= 1

    text = _text(out)
    assert "Statement of account" in text, "original page text must survive"
    assert tail in text, "long lines must not be truncated"
    assert "jane.doe@example.com" not in text, "PII must not stay copy-pasteable"
    assert "*" in text, "the text layer should be masked in place"


def test_vaulted_value_leaves_its_token_in_the_pdf(tmp_path: Path) -> None:
    """A vaulted document has to stay restorable, so the token is drawn in the box."""
    src = tmp_path / "card.pdf"
    _write_pdf(src, [["Card: 4111 1111 1111 1111"]])

    result, out = redact_pdf(src, RedactionPipeline(use_ner=False), _vault(tmp_path))

    assert result.vaulted_count == 1
    (token,) = result.token_map
    text = _text(Path(out))
    assert token in text
    assert "4111 1111 1111 1111" not in text


def test_every_page_is_redacted(tmp_path: Path) -> None:
    src = tmp_path / "two-pages.pdf"
    _write_pdf(
        src,
        [
            ["Page one", "owner: alice@example.com"],
            ["Page two", "owner: bob@example.com"],
        ],
    )

    result, out = redact_pdf(src, RedactionPipeline(use_ner=False), None)

    assert result.redacted_count == 2
    reader = PdfReader(str(out))
    assert len(reader.pages) == 2
    text = _text(Path(out))
    assert "alice@example.com" not in text
    assert "bob@example.com" not in text
    assert "Page one" in text and "Page two" in text


def test_scanned_pdf_is_refused_instead_of_emptied(tmp_path: Path) -> None:
    """Regression: an image-only PDF used to produce a silent empty document."""
    src = tmp_path / "scan.pdf"
    canvas = pdfcanvas.Canvas(str(src), pagesize=(595, 842))
    canvas.rect(100, 100, 200, 200, stroke=0, fill=1)  # graphics, no text
    canvas.save()

    with pytest.raises(RuntimeError, match="No extractable text"):
        redact_pdf(src, RedactionPipeline(use_ner=False), None)
    assert not (tmp_path / "scan_redacted.pdf").exists()
