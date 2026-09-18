"""Audit trail: a machine-readable record of every redaction run.

The sidebar promises a "full audit trail" — this module makes it true.
An audit record answers *what* was found and *what was done*, without
storing any original values (hints only), so the report itself is safe
to archive or hand to a compliance officer.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .entities import EntityType, spec
from .pipeline import PipelineReport
from .redact import RedactionResult

APP_NAME = "redacta-ai"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class AuditEntry:
    """One detected span as recorded in the audit report (no raw values)."""

    entity_type: str
    label: str
    detector: str
    score: float
    reason: str
    action: str            # "vaulted" | "irreversible"
    hint: str              # masked preview, e.g. "83••••••••28"
    token: str | None = None   # present only when vaulted


@dataclass
class AuditRecord:
    """Full record for one processed file."""

    source: str
    sha256: str | None
    chars: int
    engine: str                     # "regex" | "regex+ner"
    regex_ms: float
    ner_ms: float
    ner_note: str | None
    redacted_count: int
    vaulted_count: int
    entities: list[AuditEntry] = field(default_factory=list)
    tool: str = f"{APP_NAME} {__version__}"
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent, ensure_ascii=False)


def build_record(
    source: str,
    report: PipelineReport,
    result: RedactionResult,
    vault_hints: dict[str, str] | None = None,
    sha256_hex: str | None = None,
) -> AuditRecord:
    """Assemble an audit record from a pipeline report + redaction result.

    ``vault_hints`` maps token -> hint for vaulted entries (from Vault.entry_hint).
    """
    vault_hints = vault_hints or {}
    # token_map is token -> original *text*; match spans back via that text.
    text_to_token: dict[str, str] = {}
    for tok, orig in result.token_map.items():
        text_to_token.setdefault(orig, tok)

    entries: list[AuditEntry] = []
    for s in report.spans:
        sp = spec(s.entity_type)
        token = text_to_token.get(s.text)
        if token is not None:
            action, hint = "vaulted", vault_hints.get(token, "?")
        else:
            # No token => the value was masked permanently (either the type is
            # irreversible, or no vault was unlocked for this run).
            action = "irreversible"
            hint = (s.text[:2] + "•" * max(len(s.text) - 4, 0) + s.text[-2:]
                    if len(s.text) > 4 else "•" * len(s.text))
        entries.append(
            AuditEntry(
                entity_type=s.entity_type.value,
                label=sp.label,
                detector=s.detector,
                score=round(s.score, 2),
                reason=s.reason,
                action=action,
                hint=hint,
                token=token,
            )
        )

    engine = "regex+ner" if report.ner_available else "regex"
    return AuditRecord(
        source=source,
        sha256=sha256_hex,
        chars=report.text_length,
        engine=engine,
        regex_ms=round(report.regex_ms, 2),
        ner_ms=round(report.ner_ms, 2),
        ner_note=report.ner_error,
        redacted_count=result.redacted_count,
        vaulted_count=result.vaulted_count,
        entities=entries,
    )


def record_for_file(source_path: Path, report: PipelineReport, result: RedactionResult,
                    vault_hints: dict[str, str] | None = None) -> AuditRecord:
    """Convenience wrapper that also hashes the source file."""
    return build_record(
        source=str(source_path),
        report=report,
        result=result,
        vault_hints=vault_hints,
        sha256_hex=_sha256(source_path),
    )
