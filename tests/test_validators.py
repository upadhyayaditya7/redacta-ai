"""Tests for checksum validators."""

import random

from core.validators import (
    is_plausible_date,
    is_valid_aadhaar,
    is_valid_credit_card,
    is_valid_ifsc,
    is_valid_pan,
    make_aadhaar,
)


def test_aadhaar_valid():
    assert is_valid_aadhaar("234567891234") is False  # arbitrary 12 digits
    # Build a genuinely valid one from the generator:
    rng = random.Random(42)
    num = make_aadhaar("23456789123")
    assert is_valid_aadhaar(num)
    assert is_valid_aadhaar(num[:4] + "-" + num[4:8] + "-" + num[8:])  # grouped


def test_aadhaar_invalid_checksum_rejected():
    base = "23456789123"
    good = make_aadhaar(base)
    bad_last = str((int(good[-1]) + 1) % 10)
    assert not is_valid_aadhaar(good[:-1] + bad_last)


def test_aadhaar_rejects_leading_zero_one():
    assert not is_valid_aadhaar("123456789012")
    assert not is_valid_aadhaar("023456789012")


def test_pan_valid_and_invalid():
    assert is_valid_pan("ABCPA1234F")   # P = individual holder
    assert is_valid_pan("MNOPQ1234R")   # Q = partnership firm
    assert not is_valid_pan("ABCDD1234F")     # D is not a holder-type letter
    assert not is_valid_pan("ABCPA12345")     # last char must be a letter
    assert is_valid_pan("abcpa1234f")         # case-insensitive


def test_ifsc():
    assert is_valid_ifsc("HDFC0000123")
    assert not is_valid_ifsc("HDFC00001234")   # too long
    assert not is_valid_ifsc("HDFC1000123")    # 5th char must be 0


def test_credit_card_luhn():
    assert is_valid_credit_card("4532015112830366")
    assert not is_valid_credit_card("4532015112830367")


def test_plausible_dates():
    assert is_plausible_date(14, 3, 1996)
    assert not is_plausible_date(31, 2, 1996)   # Feb 31
    assert not is_plausible_date(14, 3, 1850)
