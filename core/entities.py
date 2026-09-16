"""Entity type registry for Redacta AI.

This module is the single source of truth for the entity types Redacta AI
detects and redacts.  Detectors emit ``Span`` objects tagged with an
``EntityType``; redactors and the vault decide how each type is handled
based on the specs below.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Action(str, Enum):
    """How a detected entity is treated during redaction."""

    VAULT = "vault"                # reversible: original stored in local vault
    IRREVERSIBLE = "irreversible"  # permanently scrubbed, no recovery


@dataclass(frozen=True)
class EntitySpec:
    """Presentation + policy metadata for an entity type."""

    label: str          # human readable name
    icon: str           # emoji used by CLI/UI
    action: Action      # default redaction policy
    severity: str = "normal"  # "critical" | "normal" | "low"


class EntityType(str, Enum):
    # --- India-specific identifiers (the wedge) -------------------------
    AADHAAR = "AADHAAR"
    PAN = "PAN"
    IFSC = "IFSC"
    UPI_ID = "UPI_ID"
    BANK_ACCOUNT = "BANK_ACCOUNT"
    INDIAN_MOBILE = "INDIAN_MOBILE"

    # --- Universal PII --------------------------------------------------
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    DATE_OF_BIRTH = "DATE_OF_BIRTH"
    CREDIT_CARD = "CREDIT_CARD"
    PERSON = "PERSON"          # NER-only
    LOCATION = "LOCATION"      # NER-only
    ORG = "ORG"                # NER-only

    # --- Credentials ----------------------------------------------------
    PASSWORD = "PASSWORD"
    SECRET_KEY = "SECRET_KEY"
    OTP = "OTP"


ENTITY_SPECS: dict[EntityType, EntitySpec] = {
    EntityType.AADHAAR: EntitySpec("Aadhaar number", "🆔", Action.VAULT, "critical"),
    EntityType.PAN: EntitySpec("PAN card number", "💳", Action.VAULT, "critical"),
    EntityType.IFSC: EntitySpec("IFSC code", "🏦", Action.VAULT, "critical"),
    EntityType.UPI_ID: EntitySpec("UPI ID", "📲", Action.VAULT, "critical"),
    EntityType.BANK_ACCOUNT: EntitySpec("Bank account number", "🏛️", Action.VAULT, "critical"),
    EntityType.INDIAN_MOBILE: EntitySpec("Indian mobile number", "📱", Action.VAULT, "normal"),
    EntityType.EMAIL: EntitySpec("Email address", "📧", Action.IRREVERSIBLE, "normal"),
    EntityType.PHONE: EntitySpec("Phone number", "☎️", Action.VAULT, "normal"),
    EntityType.DATE_OF_BIRTH: EntitySpec("Date of birth", "🎂", Action.VAULT, "normal"),
    EntityType.CREDIT_CARD: EntitySpec("Credit/debit card number", "💳", Action.VAULT, "critical"),
    EntityType.PERSON: EntitySpec("Person name", "👤", Action.VAULT, "normal"),
    EntityType.LOCATION: EntitySpec("Location / address", "📍", Action.VAULT, "low"),
    EntityType.ORG: EntitySpec("Organisation", "🏢", Action.IRREVERSIBLE, "low"),
    EntityType.PASSWORD: EntitySpec("Password field", "🔑", Action.IRREVERSIBLE, "critical"),
    EntityType.SECRET_KEY: EntitySpec("API key / secret", "🔐", Action.IRREVERSIBLE, "critical"),
    EntityType.OTP: EntitySpec("One-time password", "🔢", Action.IRREVERSIBLE, "critical"),
}

ALL_ENTITY_TYPES: list[EntityType] = list(EntityType)


def spec(entity_type: EntityType) -> EntitySpec:
    """Return the metadata spec for an entity type."""
    return ENTITY_SPECS[entity_type]
