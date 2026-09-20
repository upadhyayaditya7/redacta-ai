# Redacta AI — Challenge Proposal (draft for Sep 30 submission)

> Submission: Snapdragon® AI Lab Build & Present Challenge · individual entry
> Status: working product + 31 tests; AI model exported to ONNX and verified
> on-device; QNN compile/profile pending an AI Hub API token.

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
- **The AI tier is the bottleneck, and NPUs are the fix.** Measured on CPU on
  this machine: the deterministic regex tier answers a document in **0.27 ms**,
  while the GLiNER-PII AI tier costs **~285 ms per document** warm. The AI tier
  is what finds names, organisations and locations that regex cannot express.
  Moving *that* graph onto the Hexagon HTP is precisely what makes AI-grade
  redaction interactive on a Snapdragon X laptop.
- **Qualcomm AI Hub is the export path, and it is already wired:** GLiNER-PII →
  ONNX → QNN context binary → profiled on Snapdragon X via AI Hub's cloud
  fleet. The export is implemented, reproducible and verified; the compile job
  needs only an API token.

## 4. Technical implementation

| Tier | Model / engine | Role | Runtime today |
|---|---|---|---|
| 1 (floor) | 12 deterministic rules + checksums (Verhoeff, Luhn, PAN/IFSC) | Aadhaar, PAN, IFSC, UPI, cards, secrets | CPU, **0.27 ms/doc** |
| 2 (AI) | `urchade/gliner_multi_pii-v1` (Apache-2.0), 289M params | names, locations, orgs, DOB | **ONNX Runtime on CPU, ~285 ms/doc** → QNN/Hexagon NPU |
| 3 (roadmap) | docTR OCR ONNX | screenshots/scans | NPU |
| Vault | stdlib crypto (PBKDF2 + HMAC) | reversible redaction | local, ~ms |

**Measured, not estimated.** The AI tier is exported to ONNX (1157 MB fp32),
verified to detect real PII end-to-end on a synthetic bank statement (person,
organisation, location, DOB, email, phone at 0.95–1.00 confidence), and
benchmarked on CPU. Two findings that shaped the design:

- **Naive int8 quantisation is unusable:** it shrinks the artifact to 349 MB
  but collapses argmax agreement to **8.9%** and detects **0 of 9** entities.
  It is excluded; quantisation is deferred to QNN's HTP-aware calibration.
- **The embedding table is 66% of the model** (768 MB of 1157 MB, from a
  250k-token multilingual vocabulary). Pruning it to the vocabulary reachable
  from Indian financial documents is the largest available NPU-side saving,
  at no accuracy cost in this domain.

## 5. Evaluation criteria mapping

- **Technical Implementation** — two-tier detection, checksum validation to
  kill false positives, working vault crypto, and a **working AI Hub export
  pipeline**: real ONNX trace with pinned label prompts, verified detection,
  measured CPU + pipeline latency in-repo, a measured-and-rejected int8
  experiment, and a one-command compile/profile job for the Hexagon NPU.
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
- **M2 (mostly done):** GLiNER → ONNX ✅, verified detection ✅, CPU + pipeline
  benchmarks ✅, int8 trade-off measured ✅. Remaining: QNN compile + Hexagon
  profile (AI Hub token).
- **M3:** proposal PDF + demo video + final polish. Submit by Sep 29.

## 7. Risks & mitigations

- **No Snapdragon hardware** → AI Hub cloud profiling gives legitimate NPU
  numbers without a device; the dev demo runs the same ONNX artifact on CPU.
  Stated plainly rather than implied.
- **torch/gliner wheels on Python 3.14** → resolved: torch 2.13 + gliner 0.2.29
  install cleanly, and at runtime the ONNX artifact needs only onnxruntime.
  The regex tier remains dependency-free, so the product always runs.
- **QNN compile may reject data-dependent ops** (`NonZero`/`If`/`ScatterND`
  from GLiNER's word pooling) → shapes are pinned static for the job; fallback
  is freezing pooling indices at export time. The risk is documented in the
  benchmark report rather than discovered in the last hour.
- **Single submission immutability** → intake form dry-run before Sep 27;
  submit ≥24 h early.
