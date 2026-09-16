"""Checksum validators for identity numbers.

Regex alone produces false positives ("12 digits" could be an order id).
These validators let Redacta AI confirm candidates before flagging them,
which is the difference between a demo and a credible product.
"""

from __future__ import annotations

import re
from datetime import date

# --- Aadhaar (Verhoeff checksum, UIDAI spec) ----------------------------

_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]
_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def _verhoeff_checksum(digits: str) -> int:
    c = 0
    for i, ch in enumerate(reversed(digits)):
        c = _D[c][_P[(i + 1) % 8][int(ch)]]
    return c


def _verhoeff_full(digits: str) -> int:
    c = 0
    for i, ch in enumerate(reversed(digits)):
        c = _D[c][_P[i % 8][int(ch)]]
    return c


def is_valid_aadhaar(value: str) -> bool:
    """Validate a 12-digit Aadhaar number (Verhoeff checksum, no leading 0/1)."""
    digits = re.sub(r"\D", "", value)
    if len(digits) != 12 or digits[0] in "01":
        return False
    return _verhoeff_full(digits) == 0


def make_aadhaar(base11: str) -> str:
    """Compute the check digit for an 11-digit Aadhaar base (test-data helper)."""
    if len(base11) != 11 or not base11.isdigit() or base11[0] in "01":
        raise ValueError("Aadhaar base must be 11 digits starting with 2-9")
    return base11 + str(_INV[_verhoeff_checksum(base11)])


# --- PAN (Permanent Account Number) ------------------------------------

_PAN_RE = re.compile(r"^[A-Z]{3}[PCHFATBLJG][A-Z]\d{4}[A-Z]$")
_HOLDERS = set("PCHFATBLJG")


def is_valid_pan(value: str) -> bool:
    """Validate a PAN: AAA-structure with a valid holder-type letter."""
    v = value.strip().upper()
    if not _PAN_RE.match(v):
        return False
    return v[3] in _HOLDERS


# --- IFSC --------------------------------------------------------------

_IFSC_RE = re.compile(r"^[A-Z]{4}0[A-Z0-9]{6}$")


def is_valid_ifsc(value: str) -> bool:
    """Validate IFSC format: 4 bank-code letters, '0', 6 branch chars."""
    return bool(_IFSC_RE.match(value.strip().upper()))


# --- Credit card (Luhn) -------------------------------------------------


def is_valid_credit_card(value: str) -> bool:
    digits = re.sub(r"\D", "", value)
    if not 13 <= len(digits) <= 19:
        return False
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


# --- Dates / misc -------------------------------------------------------

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def is_plausible_date(day: int, month: int, year: int) -> bool:
    """A DOB plausible for a living Indian resident."""
    if not (1900 <= year <= date.today().year):
        return False
    try:
        parsed = date(year, month, day)
    except ValueError:
        return False
    return date(1900, 1, 1) <= parsed <= date.today()
