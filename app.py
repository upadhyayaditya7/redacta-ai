"""Redacta AI — on-device PII detection & redaction CLI.

    python app.py scan data/samples/demo.txt
    python app.py redact data/samples/demo.txt --vault-pass demo1234
    python app.py reveal «RDCT-AB12CD34» --vault-pass demo1234
    python app.py keygen
    python app.py bench
    python app.py models
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Windows consoles default to cp1252; our tokens/arrows/emoji need UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from core.entities import EntityType
from core.pipeline import RedactionPipeline
from core.redact import redact_pdf, redact_text
from core.vault import Vault
from utils.samples import document


def _build_pipeline(args) -> RedactionPipeline:
    return RedactionPipeline(use_ner=not args.no_ner)


def _parse_types(names: str | None) -> list[EntityType] | None:
    if not names:
        return None
    return [EntityType(n.strip().upper()) for n in names.split(",")]


def cmd_scan(args) -> int:
    text = Path(args.file).read_text(encoding="utf-8")
    pipeline = _build_pipeline(args)
    report = pipeline.analyze(text, _parse_types(args.types))
    print(f"Scanned {len(text):,} chars in {report.total_ms:.1f} ms")
    print(f"regex: {report.regex_ms:.1f} ms | ner: {report.ner_ms:.1f} ms "
          f"({'available' if report.ner_available else 'not installed — regex-only mode'})")
    if report.ner_error:
        print(f"ner note: {report.ner_error}")
    print(f"\nDetected {len(report.spans)} entities:\n")
    for s in report.spans:
        print(f"  [{s.entity_type.value:<14}] {s.text!r:<40} ({s.detector}, score {s.score:.2f}) — {s.reason}")
    return 0


def cmd_redact(args) -> int:
    text = Path(args.file).read_text(encoding="utf-8")
    pipeline = _build_pipeline(args)
    report = pipeline.analyze(text, _parse_types(args.types))
    vault = Vault()
    if args.vault_pass:
        if not vault.exists() or args.yes:
            vault.create(args.vault_pass)
        vault.load(args.vault_pass)
    result = redact_text(text, report, vault if args.vault_pass else None, _parse_types(args.types))
    out = Path(args.out) if args.out else Path(args.file).with_suffix(".redacted.txt")
    out.write_text(result.redacted_text, encoding="utf-8")
    print(f"Redacted {result.redacted_count} entities "
          f"({result.vaulted_count} vaulted, rest irreversible) → {out}")
    if result.token_map and args.show_tokens:
        print("\nTokens created:")
        for token in result.token_map:
            print(f"  {token}  hint: {vault.entry_hint(token)}")
    return 0


def cmd_reveal(args) -> int:
    vault = Vault()
    vault.load(args.vault_pass)
    print(vault.reveal(args.token))
    return 0


def cmd_keygen(args) -> int:
    path = Path("data/samples/demo.txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document(), encoding="utf-8")
    print(f"Generated fake-but-checksum-valid sample document → {path}\n")
    print(path.read_text(encoding="utf-8"))
    return 0


def cmd_bench(args) -> int:
    from utils.benchmark import run, write_report

    results = run()
    path = write_report(results)
    print(f"Benchmark complete → {path}")
    for name in ("regex_only_short", "regex_only_long"):
        r = results[name]
        print(f"  {name}: mean {r['mean_ms']:.2f} ms | p95 {r['p95_ms']:.2f} ms")
    return 0


def cmd_models(args) -> int:
    from models.registry import summary_lines

    print("\n".join(summary_lines()))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="redacta", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--no-ner", action="store_true", help="regex-only (no GLiNER)")
    common.add_argument("--types", help="comma-separated entity types to target")

    p = sub.add_parser("scan", parents=[common], help="detect PII, explain every hit")
    p.add_argument("file")
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("redact", parents=[common], help="redact a text/PDF file")
    p.add_argument("file")
    p.add_argument("--out", help="output path")
    p.add_argument("--vault-pass", help="passphrase to enable the reversible vault")
    p.add_argument("--show-tokens", action="store_true")
    p.add_argument("--yes", action="store_true", help="overwrite an existing vault")
    p.set_defaults(func=cmd_redact)

    p = sub.add_parser("reveal", help="reveal a «RDCT-…» token (needs vault passphrase)")
    p.add_argument("token")
    p.add_argument("--vault-pass", required=True)
    p.set_defaults(func=cmd_reveal)

    sub.add_parser("keygen", help="generate a fake sample document",
                   parents=[common]).set_defaults(func=cmd_keygen)
    sub.add_parser("bench", help="run latency benchmark",
                   parents=[common]).set_defaults(func=cmd_bench)
    sub.add_parser("models", help="show the model stack").set_defaults(func=cmd_models)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
