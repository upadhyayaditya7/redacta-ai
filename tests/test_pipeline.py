"""Tests for regex detection and the pipeline merge."""

import random

from core.entities import EntityType
from core.pipeline import RedactionPipeline
from core.redact import redact_text
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


def test_value_span_excludes_the_field_label():
    """A span must cover the value, not the "DOB :" prefix in front of it.

    Regression: the span used to start at the label, so redaction ate the
    document's own labels and the span stopped matching the value's offsets.
    """
    text = "DOB            : 14/03/1996"
    spans = RegexDetector().detect(text).spans
    dob = [s for s in spans if s.entity_type is EntityType.DATE_OF_BIRTH]
    assert dob, "DOB not detected"
    assert dob[0].text == "14/03/1996"
    assert text[dob[0].start:dob[0].end] == "14/03/1996"


def test_password_span_is_the_password_only():
    spans = RegexDetector().detect("Note: password: Hunter2026!").spans
    pw = [s for s in spans if s.entity_type is EntityType.PASSWORD]
    assert pw and pw[0].text == "Hunter2026!"


def test_secret_span_is_the_key_only():
    spans = RegexDetector().detect("api_key = sk_test_51Kx9qqAbCdEfGh123456").spans
    secret = [s for s in spans if s.entity_type is EntityType.SECRET_KEY]
    assert secret and secret[0].text == "sk_test_51Kx9qqAbCdEfGh123456"


def test_password_detected_when_label_and_value_are_in_separate_cells():
    """Table layout: label on one line, password on the next, no colon.

    Regression: a real statement leaked ``Meridian@2026`` because the rule
    demanded ``password:`` on one line.
    """
    text = "Net banking user ID aarav.sharma1987\nPassword\nMeridian@2026\n"
    spans = RegexDetector().detect(text).spans
    pw = [s for s in spans if s.entity_type is EntityType.PASSWORD]
    assert pw and pw[0].text == "Meridian@2026"


def test_password_detected_when_label_and_value_share_a_line_without_colon():
    text = "Net banking user ID aarav.sharma1987 Password Meridian@2026"
    spans = RegexDetector().detect(text).spans
    pw = [s for s in spans if s.entity_type is EntityType.PASSWORD]
    assert pw and pw[0].text == "Meridian@2026"


def test_prefixed_api_keys_are_detected_without_any_label():
    # The fixtures are assembled at runtime so the file holds no key-shaped
    # literal: GitHub push protection rejects commits containing one.
    for key in (
        "sk_" + "live_51MrdKx9qqAbCdEfGh123456",
        "whsec" + "_92bXq1LmN7pR4tYu8VwZ",
        "AKIA" + "IOSFODNN7EXAMPLE",
        "ghp" + "_16C7e42F292c6912E7710c838347Ae178B4a",
    ):
        spans = RegexDetector().detect(f"Webhook signing secret {key}").spans
        secret = [s for s in spans if s.entity_type is EntityType.SECRET_KEY]
        assert secret, f"not detected: {key}"
        assert secret[0].text == key


def test_prose_about_a_password_field_is_not_flagged():
    for text in (
        "The password policy requires rotation every 90 days.",
        "Reset your password at least once a year.",
    ):
        spans = RegexDetector().detect(text).spans
        assert not [s for s in spans if s.entity_type is EntityType.PASSWORD]


def test_customer_id_is_detected_from_its_label():
    for text in ("Customer ID MNRB00184732", "Customer ID\nMNRB00184732"):
        spans = RegexDetector().detect(text).spans
        cid = [s for s in spans if s.entity_type is EntityType.CUSTOMER_ID]
        assert cid, f"not detected in {text!r}"
        assert cid[0].text == "MNRB00184732"


def test_net_banking_user_id_is_detected_across_a_line_break():
    text = "Net banking user ID\naarav.sharma1987\nPassword\nMeridian@2026\n"
    spans = RegexDetector().detect(text).spans
    uid = [s for s in spans if s.entity_type is EntityType.USER_ID]
    assert uid and uid[0].text == "aarav.sharma1987"
    assert not [s for s in spans if s.text == "Password"]


def test_card_tail_in_a_narration_line_is_detected():
    text = "22/10/2026 AMAZON PAY INDIA PVT LTD CARD 0626 7858 POS9974100995 425.00"
    spans = RegexDetector().detect(text).spans
    card = [s for s in spans if s.entity_type is EntityType.CREDIT_CARD]
    assert card and card[0].text == "0626 7858"
    assert card[0].reason == "card fragment after a CARD label"


def test_transaction_reference_numbers_are_not_read_as_mobiles():
    """Narration rows glue the mode code to the reference (``POS9974100995``).

    Regression: the digit run was only guarded on its left by ``(?<!\\d)``, so
    every narration row counted as a phone — 52 transaction references on one
    statement — and redaction blanked them for no privacy reason.
    """
    text = (
        "Registered mobile +91 6339683276\n"
        "22/10/2026 AMAZON PAY INDIA PVT LTD POS9974100995 425.00\n"
        "05/10/2026 SALARY OCT 2026 - EXECUTIVE RESOURCES NEF8036801068 128,450.00\n"
        "11/10/2026 ZOMATO DISTRICT UPI8311720302 8,710.00\n"
        "10/10/2026 Transfer to Diya Nair IMP7683891884 6,781.00\n"
        "09/10/2026 HOME LOAN EMI ECH6686035741 3,266.00\n"
    )
    mobiles = [
        s.text for s in RegexDetector().detect(text).spans
        if s.entity_type is EntityType.INDIAN_MOBILE
    ]
    assert mobiles == ["+91 6339683276"]


def test_prose_about_accounts_and_cards_is_not_flagged():
    for text in (
        "Reset your net banking user ID regularly.",
        "Please contact customer care if your card is lost.",
    ):
        spans = RegexDetector().detect(text).spans
        assert not [
            s
            for s in spans
            if s.entity_type in {EntityType.USER_ID, EntityType.CUSTOMER_ID, EntityType.CREDIT_CARD}
        ], f"false positive in {text!r}"


def test_redaction_keeps_document_labels():
    """Only the value is replaced; the field label stays readable."""
    text = "DOB            : 14/03/1996\n"
    report = RedactionPipeline(use_ner=False).analyze(text)
    result = redact_text(text, report, None)
    assert "DOB" in result.redacted_text
    assert "14/03/1996" not in result.redacted_text


def test_verified_regex_outranks_a_confident_ner_span():
    """A checksum-verified hit is a fact; a NER guess must not displace it."""
    from core.regex_rules import Span

    ner = Span(EntityType.PERSON, 0, 16, "4111111111111111", 0.99, "ner-gliner:person")
    rx = Span(EntityType.CREDIT_CARD, 0, 16, "4111111111111111", 1.0, "regex:Card")
    kept = _resolve_overlaps([ner, rx])
    assert len(kept) == 1
    assert kept[0].entity_type is EntityType.CREDIT_CARD


def test_names_after_holder_and_payee_labels_are_detected():
    text = "Account holder Aarav Sharma Customer ID MNRB00184732 Payment to Rohan Verma"
    names = [s.text for s in RegexDetector().detect(text).spans if s.entity_type is EntityType.PERSON]
    assert names == ["Aarav Sharma", "Rohan Verma"]


def test_nominee_table_name_is_detected():
    text = "Nominee Relationship Mobile E-mail Share Meera Sharma Spouse +91 7954174143"
    names = [s.text for s in RegexDetector().detect(text).spans if s.entity_type is EntityType.PERSON]
    assert names == ["Meera Sharma"]


def test_name_label_on_one_line_does_not_swallow_the_next_field():
    text = "Account Holder : Aarav Sharma\nAadhaar        : 658325595435"
    spans = RegexDetector().detect(text).spans
    assert [s.text for s in spans if s.entity_type is EntityType.PERSON] == ["Aarav Sharma"]
    assert [s.text for s in spans if s.entity_type is EntityType.AADHAAR] == ["658325595435"]


def test_organisations_are_not_redacted_as_people():
    for text in (
        "Payment to BIGBASKET RETAIL LTD UPI2863057667",
        "TRANSFER TO AMAZON PAY INDIA PVT LTD",
        "Payment to Amazon Pay India Pvt Ltd",
    ):
        spans = RegexDetector().detect(text).spans
        assert not [s for s in spans if s.entity_type is EntityType.PERSON], text


def test_postal_address_ending_in_a_pin_is_detected():
    text = (
        "Correspondence address Flat 704, Sea Breeze Residency, 14 Linking Road, "
        "Bandra West, Mumbai 400050 Home branch Bandra West"
    )
    spans = [s for s in RegexDetector().detect(text).spans if s.entity_type is EntityType.LOCATION]
    assert spans and spans[0].text.startswith("Flat 704")
    assert spans[0].text.endswith("400050")


def test_bank_letterhead_address_is_not_flagged():
    text = "Meridian House, 3rd Floor, Plot 12, Bandra Kurla Complex, Mumbai 400051"
    spans = RegexDetector().detect(text).spans
    assert not [s for s in spans if s.entity_type is EntityType.LOCATION]


def test_statement_redaction_removes_names_and_address_but_keeps_labels():
    text = (
        "Account holder Aarav Sharma\n"
        "Correspondence address Flat 704, Sea Breeze Residency, 14 Linking Road, "
        "Bandra West, Mumbai 400050\n"
        "Payment to Rohan Verma"
    )
    report = RedactionPipeline(use_ner=False).analyze(text)
    out = redact_text(text, report, None).redacted_text
    for value in ("Aarav Sharma", "Rohan Verma", "Flat 704", "400050"):
        assert value not in out
    for label in ("Account holder", "Correspondence address", "Payment to"):
        assert label in out
