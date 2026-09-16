"""Tests for regex detection and the pipeline merge."""

import random

from core.entities import EntityType
from core.pipeline import RedactionPipeline
from core.regex_rules import RegexDetector, _resolve_overlaps
from utils.samples import aadhaar, card, document, ifsc, pan


def test_detects_aadhaar_with_checksum():
    num = aadhaar(random.Random(1))
    spans = RegexDetector().detect(f"Aadhaar: {num}").spans
    assert spans and spans[0].entity_type is EntityType.AADHAAR
    assert "Verhoeff" in spans[0].reason


def test_rejects_invalid_aadhaar():
    spans = RegexDetector().detect("Aadhaar: 234567891234").spans
    assert not any(s.entity_type is EntityType.AADHAAR for s in spans)


def test_detects_pan_ifsc_email():
    text = "PAN ABCPA1234F IFSC HDFC0000123 mail foo.bar@gmail.com"
    found = {s.entity_type for s in RegexDetector().detect(text).spans}
    assert EntityType.PAN in found
    assert EntityType.IFSC in found
    assert EntityType.EMAIL in found


def test_generated_pan_and_ifsc_validate():
    rng = random.Random(7)
    p, i = pan(rng), ifsc(rng)
    text = f"PAN {p} IFSC {i}"
    kinds = {s.entity_type for s in RegexDetector().detect(text).spans}
    assert EntityType.PAN in kinds and EntityType.IFSC in kinds


def test_credit_card_detection():
    c = card(random.Random(3))
    spans = RegexDetector().detect(f"Card: {c}").spans
    assert spans and spans[0].entity_type is EntityType.CREDIT_CARD


def test_full_document_pipeline():
    report = RedactionPipeline(use_ner=False).analyze(document(random.Random(5)))
    kinds = report.summary()
    assert kinds[EntityType.AADHAAR] >= 1
    assert kinds[EntityType.PAN] >= 1
    assert kinds[EntityType.EMAIL] >= 1
    assert report.regex_ms > 0


def test_overlap_resolution_prefers_higher_score():
    from core.regex_rules import Span

    a = Span(EntityType.BANK_ACCOUNT, 0, 12, "411111111111", 0.6, "regex")
    b = Span(EntityType.CREDIT_CARD, 0, 12, "411111111111", 1.0, "regex")
    kept = _resolve_overlaps([a, b])
    assert len(kept) == 1 and kept[0].score == 1.0


def test_types_filter():
    text = "email a@b.com PAN ABCDE1234F"
    spans = RegexDetector().detect(text, [EntityType.PAN]).spans
    assert all(s.entity_type is EntityType.PAN for s in spans)
