# Qualcomm AI Hub — NPU export & profiling guide (M2)

Goal: compile `urchade/gliner_multi_pii-v1` to a QNN context binary for the
Hexagon NPU and capture profiling numbers for the proposal. Cloud compile —
no Snapdragon device needed for the numbers.

## 1. Setup (any machine)

```bash
pip install qai-hub torch gliner
qai-hub configure --api_token <TOKEN>   # token from https://ai.qai-hub.com
```

## 2. Check the scaffold

```bash
python -m models.export_ai_hub --dry-run
python -m models.export_ai_hub --device "Snapdragon X Elite CRD"
```

## 3. What the export does

1. Load GLiNER-PII (Apache-2.0, urchade/gliner_multi_pii-v1).
2. Trace with a fixed PII-label prompt (labels pinned = static shapes).
3. `qai_hub.submit_compile_job` → QNN context binary for the chosen device.
4. `qai_hub.submit_profile_job` → NPU latency + memory report.
5. Paste the report into `benchmarks/npu.md` next to the CPU baseline from
   `python app.py bench`.

## 4. Fallbacks (document honestly in the submission)

- **AI Hub unreachable:** export GLiNER to ONNX locally, run with
  `onnxruntime` (CPU/DirectML) — still "on-device".
- **torch wheels unavailable on Python 3.14:** create a Python 3.12 venv for
  the export step only; the product itself stays stdlib-compatible.

## 5. Numbers to capture for the proposal

| Metric | Regex tier (CPU) | GLiNER CPU | GLiNER Hexagon NPU |
|---|---|---|---|
| Latency / short doc | from `benchmarks/baseline.md` | … | from AI Hub profile |
| p95 latency | … | … | … |
| Memory (peak) | — | … | … |
| Throughput (docs/s) | … | … | … |
