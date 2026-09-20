# Redacta AI 🛡️

**On-device PII detection & redaction for Indian documents — zero cloud, fully explainable.**

Drop a bank statement, ID scan or screenshot. Redacta finds the sensitive data
(Aadhaar, PAN, IFSC, UPI, cards, names, secrets…), shows **why** each detection
was made, and redacts it — either permanently, or *reversibly* into a local
encrypted vault only your passphrase can open.

Built for the **Snapdragon® AI Lab Build & Present Challenge** — designed to run
entirely on-device, with the AI model compiled for the **Hexagon NPU** of
Snapdragon X-powered HP PCs via **Qualcomm AI Hub**.

## Why on-device

Documents full of PII are exactly the files people should never upload.
Redacta's entire pipeline is local: detection, redaction, vault storage and
decryption. The proof is in the demo — run a packet capture while using it and
watch zero bytes leave the machine.

## Features

- **Checksum-validated Indian identifiers** — Aadhaar (Verhoeff), PAN
  (holder-type letter), IFSC, UPI handles, Luhn-validated cards. A 12-digit
  order id is *not* flagged as Aadhaar.
- **Explainable detections** — every hit carries a score and a human-readable
  reason (`Verhoeff checksum valid (UIDAI)`).
- **Reversible redaction vault** — `«RDCT-…»` tokens in the shared document;
  originals encrypted locally (PBKDF2-HMAC-SHA256, 200k iterations +
  HMAC integrity), recovered only with your passphrase.
- **Two-tier detection** — deterministic regex floor (0.2 ms/doc) + optional
  GLiNER zero-shot NER (`urchade/gliner_multi_pii-v1`, Apache-2.0) for names,
  locations and organisations. Degrades gracefully: no torch? Regex still runs.
- **Text, PDF and screenshot redaction** (PDF via pypdf; images via OCR +
  painted redaction boxes).
- **Batch mode** — `python app.py batch inbox/ [--watch]` redacts every
  document in a folder (text/PDF/image); `--watch` keeps polling for new
  arrivals, drop-folder style.
- **Audit trail** — every run emits a JSON report of *what* was detected and
  *what action* was taken (vaulted vs irreversible), with hints only — no raw
  values — so the report is safe to archive for compliance.
- **CLI + Streamlit UI** — script it or click it; the UI also takes file
  uploads (PDF/PNG/JPG/txt) with download buttons for redacted outputs.

## Quickstart

```bash
pip install -r requirements.txt

python app.py keygen                          # fake-but-checksum-valid sample doc
python app.py scan data/samples/demo.txt      # detect + explain
python app.py redact data/samples/demo.txt --vault-pass demo1234 --show-tokens
python app.py batch data/inbox --vault-pass demo1234 [--watch]  # folder mode
python -c "from core.vault import Vault; print(Vault().reveal('«RDCT-…»'))"  # with --vault-pass
python app.py bench                           # latency → benchmarks/baseline.md
python app.py models                          # model stack + AI Hub targets

streamlit run frontend/app.py                 # the UI
```

Tests: `python -m pytest tests -q` (31 passing).

## Architecture

```text
core/
  entities.py     entity registry + redaction policy (vault vs irreversible)
  validators.py   Verhoeff / PAN / IFSC / Luhn / date checks
  regex_rules.py  deterministic detector: 12 rules, overlap resolution, reasons
  ner_detector.py optional GLiNER wrapper — ONNX first, then PyTorch, then
                  regex-only (always reports which backend ran)
  pipeline.py     regex + NER merge with per-stage timings
  redact.py       text / PDF / image redaction
  vault.py        stdlib-crypto reversible vault
  audit.py        JSON audit trail (hints only, no raw values)
  batch.py        folder batch + watch mode
models/
  registry.py       model stack + AI Hub export targets
  export_ai_hub.py  real ONNX export + AI Hub compile/profile for Hexagon NPU
frontend/app.py   Streamlit studio
utils/            synthetic data generator (checksum-valid fakes), benchmark harness
```

## Snapdragon NPU path

The AI tier is exported to ONNX with a pinned PII label prompt and served by
ONNX Runtime on-device — no torch needed at inference. `models/export_ai_hub.py`
then submits that same graph to Qualcomm AI Hub for a QNN compile + profile job
targeting the Hexagon NPU (Snapdragon X / X2 Plus). AI Hub compiles cloud-side,
so real NPU numbers need no Snapdragon device.

```bash
pip install -r requirements-ai.txt
python -m models.export_ai_hub --dry-run    # plan only, no dependencies
python -m models.export_ai_hub --all        # export + verify + benchmark + report
python -m models.export_ai_hub --submit     # needs a free AI Hub token
```

Measured on CPU (see [`benchmarks/npu.md`](benchmarks/npu.md)): regex tier
**0.27 ms/doc**, AI tier **~285 ms/doc** warm — which is exactly the case for
moving the graph onto the HTP. Two findings are recorded there rather than
hidden: naive int8 quantisation collapses the model (8.9% argmax agreement,
0 of 9 entities found), and the token embedding table is 66% of the artifact.

## Status

Working pipeline (detect → explain → redact → vault → reveal → batch → audit),
31 tests passing. AI tier exported to ONNX, verified, and benchmarked; NPU
compile/profile is one token away. Roadmap: Hexagon profile numbers, embedding
vocabulary pruning, OCR screenshot mode polish.

## License / ownership

Original work by the author; submissions remain the participant's IP per the
challenge rules. Sample data is synthetic — no real identifiers anywhere.
