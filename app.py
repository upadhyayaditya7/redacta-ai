"""Redacta AI — on-device PII detection & redaction CLI.

    python app.py scan data/samples/demo.txt
    python app.py redact data/samples/demo.txt --vault-pass demo1234
    python app.py reveal «RDCT-AB12CD34» --vault-pass demo1234
    python app.py batch inbox/ --vault-pass demo1234 [--watch]
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


def _vault_from_args(args) -> Vault | None:
    """Build an (optionally unlocked) vault from --vault-pass; None if absent."""
    if not args.vault_pass:
        return None
    vault = Vault()
    if not vault.exists() or getattr(args, "yes", False):
        vault.create(args.vault_pass)
    vault.load(args.vault_pass)
    return vault


def _report_path(args, out: Path) -> Path:
    """--audit-report <path> wins; otherwise auto-name next to the output."""
    if args.audit_report and args.audit_report != "none":
        return Path(args.audit_report)
    return out.with_suffix(".audit.json")


def cmd_scan(args) -> int:
    text = Path(args.file).read_text(encoding="utf-8")
    pipeline = _build_pipeline(args)
    report = pipeline.analyze(text, _parse_types(args.types))
    if report.ner_available:
        ner_status = f"available, {report.ner_backend or 'unknown'} backend"
    else:
        ner_status = "not installed — regex-only mode"
    print(f"Scanned {len(text):,} chars in {report.total_ms:.1f} ms")
    print(f"regex: {report.regex_ms:.1f} ms | ner: {report.ner_ms:.1f} ms ({ner_status})")
    if report.ner_error:
        print(f"ner note: {report.ner_error}")
    print(f"\nDetected {len(report.spans)} entities:\n")
    for s in report.spans:
        print(f"  [{s.entity_type.value:<14}] {s.text!r:<40} ({s.detector}, score {s.score:.2f}) — {s.reason}")
    return 0


def cmd_redact(args) -> int:
    src = Path(args.file)
    pipeline = _build_pipeline(args)
    vault = _vault_from_args(args)
    ext = src.suffix.lower()
    if ext == ".pdf":
        from core.redact import redact_pdf

        result, out = redact_pdf(src, pipeline, vault, _parse_types(args.types))
        out = Path(args.out) if args.out else Path(out)
    elif ext in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
        from core.redact import redact_image

        result, out = redact_image(src, pipeline, vault, _parse_types(args.types))
        out = Path(args.out) if args.out else Path(out)
    else:
        text = src.read_text(encoding="utf-8")
        report = pipeline.analyze(text, _parse_types(args.types))
        result = redact_text(text, report, vault, _parse_types(args.types))
        out = Path(args.out) if args.out else src.with_suffix(".redacted.txt")
        out.write_text(result.redacted_text, encoding="utf-8")
    print(f"Redacted {result.redacted_count} entities "
          f"({result.vaulted_count} vaulted, rest irreversible) → {out}")
    if result.token_map and args.show_tokens:
        print("\nTokens created:")
        for token in result.token_map:
            print(f"  {token}  hint: {vault.entry_hint(token) if vault else '••••'}")
    if args.audit_report != "none":
        from core.audit import build_record
        rec = build_record(
            source=str(args.file), report=report, result=result,
            vault_hints={t: vault.entry_hint(t) for t in result.token_map} if vault else None,
        )
        rp = _report_path(args, out)
        rp.write_text(rec.to_json(), encoding="utf-8")
        print(f"Audit report → {rp}")
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


def cmd_batch(args) -> int:
    from core.batch import process_folder, watch_folder

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"Not a folder: {folder}")
        return 1
    pipeline = _build_pipeline(args)
    vault = _vault_from_args(args)
    types = _parse_types(args.types)

    if args.watch:
        watch_folder(folder, pipeline, vault, types)
        return 0

    items = process_folder(folder, pipeline, vault, types)
    out_dir = folder.with_name(folder.name + "_redacted")
    print(f"Processed {sum(1 for i in items if not i.error)}/{len(items)} file(s) → {out_dir}")
    for it in items:
        if it.error:
            print(f"  ! {it.src.name}: {it.error}")
        else:
            print(f"  {it.src.name} → {it.out.name}")
    if args.audit_report != "none":
        import json

        from core.audit import _sha256, build_record
        records = []
        for it in items:
            if it.error:
                records.append({"source": str(it.src), "error": it.error})
                continue
            rec = build_record(
                source=str(it.src), report=it.report or it.result.report, result=it.result,
                vault_hints={t: vault.entry_hint(t) for t in it.result.token_map} if vault else None,
                sha256_hex=_sha256(it.src),
            )
            records.append(json.loads(rec.to_json()))
        rp = out_dir / "audit.json"
        rp.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Audit report → {rp}")
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
    p.add_argument("--audit-report", default=None, metavar="PATH",
                   help="write a JSON audit report (default: <output>.audit.json; 'none' disables)")
    p.set_defaults(func=cmd_redact)

    p = sub.add_parser("reveal", help="reveal a «RDCT-…» token (needs vault passphrase)")
    p.add_argument("token")
    p.add_argument("--vault-pass", required=True)
    p.set_defaults(func=cmd_reveal)

    p = sub.add_parser("batch", parents=[common], help="redact every document in a folder")
    p.add_argument("folder")
    p.add_argument("--vault-pass", help="passphrase to enable the reversible vault")
    p.add_argument("--watch", action="store_true", help="keep watching for new files")
    p.add_argument("--audit-report", default=None, metavar="PATH",
                   help="write a JSON audit report (default: audit.json in the output folder; 'none' disables)")
    p.add_argument("--yes", action="store_true", help="overwrite an existing vault")
    p.set_defaults(func=cmd_batch)

    sub.add_parser("keygen", help="generate a fake sample document",
                   parents=[common]).set_defaults(func=cmd_keygen)
    sub.add_parser("bench", help="run latency benchmark",
                   parents=[common]).set_defaults(func=cmd_bench)
    sub.add_parser("models", help="show the model stack").set_defaults(func=cmd_models)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
