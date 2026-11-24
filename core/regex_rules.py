"""Regex-based PII detector.

Deterministic, fast, and explains every hit ("why").  Indian identifiers
are checksum-validated, so a 12-digit order id is not flagged as Aadhaar.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .entities import EntityType
from .validators import (
    is_plausible_date,
    is_valid_aadhaar,
    is_valid_credit_card,
    is_valid_ifsc,
    is_valid_pan,
)


@dataclass
class Span:
    """A detected entity: character offsets + metadata."""

    entity_type: EntityType
    start: int
    end: int
    text: str
    score: float
    detector: str
    reason: str = ""

    def overlaps(self, other: "Span") -> bool:
        return self.start < other.end and other.start < self.end

    def __post_init__(self) -> None:
        self.start = int(self.start)
        self.end = int(self.end)


@dataclass
class DetectionResult:
    """All detections for one document, with overlap resolution applied."""

    spans: list[Span] = field(default_factory=list)

    def by_type(self) -> dict[EntityType, list[Span]]:
        out: dict[EntityType, list[Span]] = {}
        for s in self.spans:
            out.setdefault(s.entity_type, []).append(s)
        return out


# --- Patterns -----------------------------------------------------------

_AADHAAR = r"(?<!\d)([2-9]\d{3}[ -]?\d{4}[ -]?\d{4})(?!\d)"
_PAN = r"(?<![A-Z0-9])([A-Z]{3}[PCHFATBLJG][A-Z][0-9]{4}[A-Z])(?![A-Z0-9])"
_IFSC = r"(?<![A-Z0-9])([A-Z]{4}0[A-Z0-9]{6})(?![A-Z0-9])"
_UPI = (
    r"(?<![A-Za-z0-9._-])"
    r"([A-Za-z][A-Za-z0-9._-]{1,48}@(?:ok(?:hdfcbank|icici|axis|sbi|biz|paytm|ybl|axisbank)|upi|apl|ybl|paytm|ibl|axl))"
    r"(?![A-Za-z0-9._-])"
)
_BANK_ACCT = r"(?<!\d)(\d{11,18})(?!\d)"
_IN_MOBILE = r"(?<!\d)(\+?91[ -]?)?([6-9]\d{4}[ -]?\d{5})(?!\d)"
_EMAIL = r"(?<![A-Za-z0-9._%+-])([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})(?![A-Za-z0-9._%+-])"
_CC = r"(?<!\d)((?:\d{4}[ -]?){3}\d{1,4})(?!\d)"
_SECRET = (
    r"(?i)\b(?:api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token|private[_-]?key)"
    r"\b\s*[:=]\s*(\S+)"
)
#: Credential values that carry a recognisable issuer prefix.  These are
#: matched on their own, with no ``label:`` in front, because statements and
#: config dumps routinely put the value in a neighbouring cell or line.
_KEY_PREFIX = (
    r"(?<![A-Za-z0-9_-])("
    r"(?:sk|pk)_(?:test|live)_[0-9a-zA-Z]{16,}"
    r"|whsec_[0-9A-Za-z]{16,}"
    r"|rk_live_[0-9A-Za-z]{16,}"
    r"|AKIA[0-9A-Z]{16}"
    r"|gh[pousr]_[0-9A-Za-z]{20,}"
    r"|xox[abprs]-[0-9A-Za-z-]{10,}"
    r"|AIza[0-9A-Za-z_-]{30,}"
    r")(?![A-Za-z0-9_-])"
)
_SK_KEY = r"\b(sk|pk)_(?:test|live)_[0-9a-zA-Z]{16,}\b"
_OTP = r"(?i)\b(?:otp|one[- ]time (?:password|code))\b\s*(?:is|:|-)?\s*([0-9]{4,8})\b"
_PW = r"(?i)\b(pass(word)?|pwd)\b\s*[:=]\s*(\S{4,64})"
#: Same field, but the label and the value are only separated by whitespace —
#: a table cell or a PDF text run where the colon is invisible to extraction
#: (``Password`` / ``Meridian@2026`` on consecutive lines).  The value must
#: still look like a credential (digits plus a symbol), so prose such as
#: "the password policy requires rotation" is left alone.
_PW_SPACED = (
    r"(?i)\b(?:pass(?:word)?|pwd)\b\s+"
    r"((?=[^\s]*\d)(?=[^\s]*[^A-Za-z0-9])[^\s]{4,64})(?=\s|$)"
)
#: Credential field with a whitespace-only separator, where the value has no
#: issuer prefix (a bare 32-character token and similar).
_SECRET_SPACED = (
    r"(?i)\b(?:api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token"
    r"|private[_-]?key|signing secret|webhook secret)\b\s+"
    r"((?=[^\s]*\d)(?=[^\s]*[A-Za-z])[A-Za-z0-9_-]{16,})(?=\s|$)"
)
_DOB_PREFIX = r"(?i)(?:d\.?o\.?b\.?|dob|date of birth|birth date)\s*[:=-]?\s*"
_DMY = r"(\d{1,2})[ \-/.]([A-Za-z]{3,9}|\d{1,2})[ \-/.](\d{4})"

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _validate(entity_type: EntityType, text: str, groups: tuple[str, ...]) -> tuple[float, str] | None:
    """Return (score, reason) if the candidate is genuine, else None."""
    if entity_type is EntityType.AADHAAR:
        if is_valid_aadhaar(groups[0]):
            return 1.0, "Verhoeff checksum valid (UIDAI)"
        return None
    if entity_type is EntityType.PAN:
        if is_valid_pan(groups[0]):
            return 1.0, "PAN structure + holder type valid"
        return None
    if entity_type is EntityType.IFSC:
        if is_valid_ifsc(groups[0]):
            return 1.0, "IFSC format valid (bank0branch)"
        return None
    if entity_type is EntityType.CREDIT_CARD:
        if is_valid_credit_card(groups[0]):
            return 1.0, "Luhn checksum valid"
        return None
    if entity_type is EntityType.DATE_OF_BIRTH:
        d, m, y = groups
        month = _MONTHS.get(m[:3].lower()) if m.isalpha() else int(m)
        if month and is_plausible_date(int(d), month, int(y)):
            return 0.9, "Prefixed DOB with plausible date"
        return None
    if entity_type is EntityType.BANK_ACCOUNT:
        return 0.6, "11-18 digit number (checksum unavailable)"
    if entity_type is EntityType.SECRET_KEY:
        if groups[0] and re.fullmatch(_KEY_PREFIX, groups[0]):
            return 1.0, "Known secret-key prefix (sk_/whsec_/AKIA/…)"
        return 0.95, "Credential key=value pattern"
    if entity_type is EntityType.OTP:
        return 0.95, "'OTP is <code>' pattern"
    if entity_type is EntityType.PASSWORD:
        return 0.9, "password field (label next to value)"
    return 0.8, "regex pattern match"


#: ``(type, label, pattern, first_group, last_group)``.  The group range says
#: which capture groups hold the *value*: several patterns match a field label
#: (``DOB :``, ``password:``) so that the value can be validated in context, and
#: the span must cover the value alone — otherwise redaction eats the document's
#: own labels and downstream offsets stop lining up with the real value.
#: ``(0, 0)`` means "the whole match".
_RULES: tuple[tuple[EntityType, str, re.Pattern[str], int, int], ...] = (
    (EntityType.AADHAAR, "Aadhaar", re.compile(_AADHAAR), 1, 1),
    (EntityType.PAN, "PAN", re.compile(_PAN), 1, 1),
    (EntityType.IFSC, "IFSC", re.compile(_IFSC), 1, 1),
    (EntityType.UPI_ID, "UPI", re.compile(_UPI), 1, 1),
    (EntityType.EMAIL, "Email", re.compile(_EMAIL), 1, 1),
    (EntityType.SECRET_KEY, "Secret", re.compile(_SECRET), 1, 1),
    (EntityType.SECRET_KEY, "Key-prefix", re.compile(_KEY_PREFIX), 1, 1),
    (EntityType.SECRET_KEY, "Secret", re.compile(_SECRET_SPACED), 1, 1),
    (EntityType.OTP, "OTP", re.compile(_OTP), 1, 1),
    (EntityType.PASSWORD, "Password", re.compile(_PW), 3, 3),
    (EntityType.PASSWORD, "Password", re.compile(_PW_SPACED), 1, 1),
    (EntityType.DATE_OF_BIRTH, "DOB", re.compile(_DOB_PREFIX + _DMY), 1, 3),
    (EntityType.CREDIT_CARD, "Card", re.compile(_CC), 1, 1),
    (EntityType.INDIAN_MOBILE, "IN-mobile", re.compile(_IN_MOBILE), 0, 0),
    (EntityType.BANK_ACCOUNT, "BankAcct", re.compile(_BANK_ACCT), 1, 1),
)


class RegexDetector:
    """Deterministic detector. Returns every validated hit with a reason."""

    name = "regex"

    def detect(self, text: str, types: list[EntityType] | None = None) -> DetectionResult:
        wanted = set(types) if types else None
        spans: list[Span] = []
        for entity_type, label, rx, first_group, last_group in _RULES:
            if wanted is not None and entity_type not in wanted:
                continue
            for m in rx.finditer(text):
                groups = m.groups() if rx.groups else ()
                if first_group:
                    start, end = m.start(first_group), m.end(last_group)
                    if start < 0 or end < 0:  # optional group did not match
                        continue
                else:
                    start, end = m.start(), m.end()
                checked = groups[0] if groups else m.group(0)
                verdict = _validate(entity_type, checked, groups)
                if verdict is None:
                    continue
                score, reason = verdict
                spans.append(
                    Span(
                        entity_type=entity_type,
                        start=start,
                        end=end,
                        text=text[start:end],
                        score=score,
                        detector=f"regex:{label}",
                        reason=reason,
                    )
                )
        return DetectionResult(_resolve_overlaps(spans))


def _verified(span: Span) -> int:
    """1 when a checksum/format validator signed off on this hit, else 0.

    A verified regex hit (Verhoeff, Luhn, PAN structure) is a fact; a NER hit is
    a probability.  Ranking by score alone let a 0.997 "phone number" guess
    displace the exact regex match and, worse, let a broad NER person span
    swallow a card or Aadhaar number inside it.  Verified hits therefore always
    win an overlap, and only the remaining spans are ranked by score.
    """
    return 1 if span.detector.startswith("regex") and span.score >= 0.99 else 0


def _resolve_overlaps(spans: list[Span]) -> list[Span]:
    """Keep the best span when detections overlap (verified first, then score)."""
    ordered = sorted(
        spans, key=lambda s: (-_verified(s), -s.score, -(s.end - s.start), s.start)
    )
    kept: list[Span] = []
    for s in ordered:
        if not any(s.overlaps(k) for k in kept):
            kept.append(s)
    return sorted(kept, key=lambda s: s.start)
