# Redacta AI — Challenge Proposal (draft for Sep 30 submission)

> Submission: Snapdragon® AI Lab Build & Present Challenge · individual entry
> Status: working prototype + 21 tests; NPU profiling scheduled as M2.

## 1. Problem

India runs on identity documents: Aadhaar, PAN, bank statements, UPI
screenshots. Millions are shared daily over email, WhatsApp and portals —
usually with *every* identifier exposed, because manual redaction is tedious
and cloud redaction tools require uploading the very data being protected.

There is no trustworthy, automated, on-device redaction story for Indian
identifiers. That gap is exactly what NPUs are for: private inference that
never leaves the machine.

## 2. Solution

Redacta AI detects and redacts sensitive information in documents and
screenshots **locally**, with two redaction modes:

- **Irreversible** — permanently masked for outward sharing.
- **Reversible** — replaced with `«RDCT-…»` tokens; originals encrypted in a
  local vault (PBKDF2-HMAC-SHA256 + HMAC integrity) that only the owner's
  passphrase opens. Authorised workflows (compliance, support escalation) can
  re-identify; everyone else sees tokens.

Every detection is **explained** ("Verhoeff checksum valid (UIDAI)") — no
black boxes.

## 3. Why Snapdragon (alignment with the brief)

- **Private by architecture:** the Hexagon NPU runs the detection model with
  data resident in local memory — no cloud round-trip exists to leak.
- **Latency for interactive redaction:** sub-10 ms regex tier today; GLiNER
  compiled to QNN targets real-time redaction as you type/watch.
- **Qualcomm AI Hub is the export path:** GLiNER-PII → ONNX → QNN context
  binary, profiled on Snapdragon X Elite / X2 Plus via AI Hub's cloud fleet.

## 4. Technical implementation

| Tier | Model / engine | Role | Target runtime |
|---|---|---|---|
| 1 (floor) | 12 deterministic rules + checksums (Verhoeff, Luhn, PAN/IFSC) | Aadhaar, PAN, IFSC, UPI, cards, secrets | CPU, 0.2 ms/doc |
| 2 (AI) | `urchade/gliner_multi_pii-v1` (Apache-2.0) | names, locations, orgs, DOB | ONNX Runtime → QNN on Hexagon NPU |
| 3 (roadmap) | docTR OCR ONNX | screenshots/scans | NPU |
| Vault | stdlib crypto (PBKDF2 + HMAC) | reversible redaction | local |

Measured baseline (this machine): 0.18 ms mean per short doc, 6.6 ms per
~20-page doc — regex tier. NPU comparison lands with M2 profiling.

## 5. Evaluation criteria mapping

- **Technical Implementation** — two-tier detection, checksum validation to
  kill false positives, working vault crypto, benchmark harness in-repo,
  AI Hub export scaffold ready.
- **Application Use Case & Innovation** — India-first identifiers (Aadhaar/
  PAN/IFSC/UPI); *reversible* redaction with auditable tokens is the wedge
  no mainstream tool offers; explainable detections.
- **Deployment & Accessibility** — CLI + one-command Streamlit UI, zero-cloud
  install, graceful degradation (regex-only mode runs anywhere), synthetic
  sample generator so anyone can demo safely.
- **Presentation & Documentation** — this proposal, README, benchmark tables,
  3-minute demo video script (docs/DEMO_SCRIPT.md), packet-capture proof of
  zero network calls.

## 6. Milestones

- **M0 (done):** detection + explanation, redaction, vault, CLI, UI, tests.
- **M1:** OCR screenshot mode end-to-end; batch/watch-folder mode.
- **M2:** GLiNER → ONNX → QNN via AI Hub; NPU-vs-CPU benchmark table.
- **M3:** proposal PDF + demo video + final polish. Submit by Sep 29.

## 7. Risks & mitigations

- **No Snapdragon hardware** → AI Hub cloud profiling gives legitimate NPU
  numbers; dev-demo runs ONNX Runtime; stated honestly in submission.
- **torch/gliner wheels on Python 3.14** → regex tier is dependency-free and
  always demoable; AI layer verified on demand (Python 3.12 venv if needed).
- **Single submission immutability** → intake form dry-run before Sep 27;
  submit ≥24 h early.
