"""Synthetic sample-data generator.

Produces *fake but checksum-valid* Indian identifiers so demos and tests
exercise the validators without touching anyone's real data.  All data is
randomly generated on the fly — nothing here is real.
"""

from __future__ import annotations

import random
import string

from core.validators import is_valid_credit_card, make_aadhaar

_FIRST = ["Aarav", "Vivaan", "Ananya", "Diya", "Ishaan", "Meera", "Rohan", "Pihu"]
_LAST = ["Sharma", "Verma", "Iyer", "Nair", "Patel", "Gupta", "Reddy", "Khan"]
_BANKS = ["HDFC", "ICIC", "SBIN", "UTIB", "KKBK"]  # IFSC bank-code prefixes
_UPI_DOMAINS = ["okhdfcbank", "okicici", "ybl", "paytm", "apl", "upi"]


def aadhaar(rng: random.Random) -> str:
    return make_aadhaar("".join(rng.choice("23456789") for _ in range(11)))


def pan(rng: random.Random) -> str:
    holder = rng.choice("PCHFATBLJG")
    letters = "".join(rng.choice(string.ascii_uppercase) for _ in range(4))
    return letters[:3] + holder + letters[3] + f"{rng.randint(1000, 9999)}" + rng.choice(string.ascii_uppercase)


def ifsc(rng: random.Random) -> str:
    return rng.choice(_BANKS) + "0" + "".join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(6))


def mobile(rng: random.Random) -> str:
    return "+91 " + str(rng.randint(6000000000, 9999999999))


def email(rng: random.Random, name: str) -> str:
    handle = name.lower().replace(" ", ".") + str(rng.randint(1, 999))
    return handle + "@" + rng.choice(["gmail.com", "outlook.com", "proton.me"])


def card(rng: random.Random) -> str:
    # Visa-style prefix 4, remainder computed so Luhn passes.
    body = "4" + "".join(str(rng.randint(0, 9)) for _ in range(14))
    for check in range(10):
        candidate = body + str(check)
        if is_valid_credit_card(candidate):
            return " ".join(candidate[i:i+4] for i in (0, 4, 8, 12))
    return body + "0"


def upi(rng: random.Random, name: str) -> str:
    handle = name.lower().replace(" ", "") + rng.choice(["", "9", "42"])
    return handle + "@" + rng.choice(_UPI_DOMAINS)


def document(rng: random.Random | None = None) -> str:
    """A fake bank-statement-style document touching every detector."""
    rng = rng or random.Random()
    name = rng.choice(_FIRST) + " " + rng.choice(_LAST)
    first = name.split()[0]
    return (
        f"Account Holder : {name}\n"
        f"Aadhaar        : {aadhaar(rng)}\n"
        f"PAN            : {pan(rng)}\n"
        f"IFSC           : {ifsc(rng)}\n"
        f"Mobile         : {mobile(rng)}\n"
        f"Email          : {email(rng, first)}\n"
        f"UPI            : {upi(rng, first)}\n"
        f"Card           : {card(rng)}\n"
        f"DOB            : 14/03/1996\n"
        f"Date           : 12 Sep 2026\n"
        f"Balance        : INR 1,42,030.55\n"
        f"Note           : password: Hunter2026! api_key = sk_test_51Kx9qqAbCdEfGh123456\n"
    )
