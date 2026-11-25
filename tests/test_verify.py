"""Post-redaction verification must catch a real leak, not just say "ok"."""

from __future__ import annotations

from pathlib import Path

from core.entities import EntityType
from core.pipeline import PipelineReport, RedactionPipeline
from core.redact import redact_pdf, redact_text
from core.regex_rules import Span
from core.verify import verify_file, verify_text

PII = (
    "Account holder PAN: YYSGB5432H\n"
    "Card: 4111 1111 1111 1111\n"
    "Registered e-mail: aarav235@proton.me\n"
)


def _pipeline() -> RedactionPipeline:
    return RedactionPipeline(use_ner=False)


def _write_pdf(path: Path, lines: list[str], author: str | None = None) -> Path:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=A4)
    if author:
        c.setAuthor(author)
    y = 800
    for line in lines:
        c.drawString(40, y, line)
        y -= 16
    c.showPage()
    c.save()
    return path


def test_redacted_text_verifies_clean():
    report = _pipeline().analyze(PII)
    result = redact_text(PII, report, None)
    check = verify_text(result.redacted_text, report)
    assert check.ok, check.leaks
    assert check.checked_values >= 3


def test_unredacted_copy_is_reported_as_a_leak():
    report = _pipeline().analyze(PII)
    check = verify_text(PII, report)
    assert not check.ok
    assert len(check.leaks) >= 3
    # The report never carries the original value, only a masked preview.
    for leak in check.leaks:
        assert "•" in leak.hint
        assert leak.where == "output text"


def test_pdf_output_verifies_clean_and_the_original_does_not(tmp_path):
    src = _write_pdf(tmp_path / "statement.pdf", [
        "Account holder PAN: YYSGB5432H",
        "Registered e-mail: aarav235@proton.me",
    ])
    result, out = redact_pdf(src, _pipeline(), None)
    clean = verify_file(out, result.report)
    assert clean.ok, [(lk.label, lk.where, lk.hint) for lk in clean.leaks]
    assert clean.checked_chars > 0
    # Control: the same check must fail on the unredacted original, otherwise
    # an "ok" above would prove nothing.
    assert not verify_file(src, result.report).ok


def test_value_left_in_pdf_metadata_is_reported(tmp_path):
    src = _write_pdf(tmp_path / "meta.pdf", ["Statement of account"], author="YYSGB5432H")
    report = _pipeline().analyze("Account holder PAN: YYSGB5432H")
    check = verify_file(src, report)
    assert not check.ok
    assert any(leak.where == "PDF metadata" for leak in check.leaks)


def test_values_too_short_to_identify_are_not_checked():
    span = Span(EntityType.PERSON, 0, 2, "Al", 0.9, "ner", "test span")
    check = verify_text("Al", PipelineReport(spans=[span]))
    assert check.checked_values == 0
    assert check.ok
