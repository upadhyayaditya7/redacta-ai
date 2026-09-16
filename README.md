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
- **CLI + Streamlit UI** — script it or click it.

## Quickstart

```bash
pip install -r requirements.txt

python app.py keygen                          # fake-but-checksum-valid sample doc
python app.py scan data/samples/demo.txt      # detect + explain
python app.py redact data/samples/demo.txt --vault-pass demo1234 --show-tokens
python -c "from core.vault import Vault; print(Vault().reveal('«RDCT-…»'))"  # with --vault-pass
python app.py bench                           # latency → benchmarks/baseline.md
python app.py models                          # model stack + AI Hub targets

streamlit run frontend/app.py                 # the UI
```

Tests: `python -m pytest tests -q` (21 passing).

## Architecture

```text
core/
  entities.py     entity registry + redaction policy (vault vs irreversible)
  validators.py   Verhoeff / PAN / IFSC / Luhn / date checks
  regex_rules.py  deterministic detector: 12 rules, overlap resolution, reasons
  ner_detector.py optional GLiNER wrapper (graceful fallback)
  pipeline.py     regex + NER merge with per-stage timings
  redact.py       text / PDF / image redaction
  vault.py        stdlib-crypto reversible vault
models/
  registry.py     model stack + AI Hub export targets
  export_ai_hub.py  GLiNER → ONNX → QNN (Hexagon NPU) export scaffold
frontend/app.py   Streamlit studio
utils/            synthetic data generator (checksum-valid fakes), benchmark harness
```

## Snapdragon NPU path

Dev machines run the regex tier on CPU and GLiNER via PyTorch/ONNX Runtime.
The NPU leg (`models/export_ai_hub.py`) compiles GLiNER via Qualcomm AI Hub
into a QNN context binary for Hexagon (Snapdragon X / X2 Plus), and the
benchmark harness (`utils/benchmark.py`) is built to produce the CPU-vs-NPU
comparison table. AI Hub profiling works cloud-side — no Snapdragon device
required to generate the report.

## Status

Working pipeline (detect → explain → redact → vault → reveal → bench), 21
tests passing. Roadmap: AI Hub NPU profiling (M2), OCR screenshot mode polish,
watch-folder batch mode.

## License / ownership

Original work by the author; submissions remain the participant's IP per the
challenge rules. Sample data is synthetic — no real identifiers anywhere.
