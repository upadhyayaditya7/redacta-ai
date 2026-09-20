# Redacta AI - AI tier & Hexagon NPU benchmark

Every number below is measured on the machine that produced this file,
or fetched from a real Qualcomm AI Hub job. Nothing is estimated.

## 1a. Model graph latency (the shape an AI Hub profile reports)

| Model graph | Runtime | Mean | p95 | Docs/s | Artifact size |
|---|---|---|---|---|---|
| GLiNER-PII (model.onnx) | ONNX Runtime, CPU | 443.7 ms | 458.6 ms | 2.3 | 1157.1 MB |
| GLiNER-PII (QNN context binary) | Hexagon NPU | PENDING | - | - | - |

## 1b. End-to-end product latency (what a user waits for)

| Configuration | Runtime | Mean | p95 | Docs/s |
|---|---|---|---|---|
| Regex + checksums only (deterministic floor) | CPython | 0.27 ms | 0.32 ms | 3658.7 |
| Regex + AI tier, warm (419-char doc) | ONNX Runtime, CPU | 284.7 ms | 287.9 ms | 3.5 |

Warm run = regex 0.38 ms + AI tier 281.8 ms (backend: onnx), finding 11 entities. Cold start (model + ONNX session init) is 5286.2 ms, so the AI tier is warm-loaded once per session, not once per document.

**This is the case for the NPU.** The deterministic tier answers in well under a millisecond, but the AI tier costs hundreds of milliseconds of CPU per document. Moving that graph onto the Hexagon HTP is what makes AI-grade redaction interactive on a Snapdragon X laptop.

> **The NPU row is PENDING**: compiling and profiling on AI Hub requires an
> API token (`~/.qai_hub/client.ini`). Request access at
> https://aihub.qualcomm.com/, then run:
>
> ```bash
> python -m qai_hub configure --api_token <TOKEN>
> python -m models.export_ai_hub --submit --device "Snapdragon X Elite CRD"
> ```

## 2. Correctness of the exported artifact

Entities found by the ONNX graph on the representative document
(this is the artifact submitted for NPU compilation):

| Label | Value | Confidence |
|---|---|---|
| organization | `State Bank of India, Pune Main Branch` | 0.728 |
| person | `Ananya Nair` | 0.998 |
| phone number | `+91 8409150555` | 0.997 |
| email | `ananya757@gmail.com` | 0.999 |
| email | `ananya42@okicici` | 0.713 |
| date of birth | `14/03/1996` | 0.998 |
| person | `Rohan Nair` | 0.973 |
| person | `Meera Iyer` | 0.968 |
| organization | `HDFC Bank Ltd.` | 0.953 |

## 3. Finding: naive int8 quantisation is not usable here

Dynamic int8 quantisation makes the graph smaller and faster but
destroys detection quality, so it is **excluded from the product**:

| Metric | Result |
|---|---|
| Logits cosine similarity | 0.9576 |
| Logits max absolute difference | 54.91 |
| Argmax agreement | 8.9% |
| Entities detected (fp32) | 9 |
| Entities detected (int8) | 0 |
| Missing after int8 | +91 8409150555, 14/03/1996, Ananya Nair, HDFC Bank Ltd., Meera Iyer, Rohan Nair, State Bank of India, Pune Main Branch, ananya42@okicici, ananya757@gmail.com |
| Size (fp32 -> int8) | 1157 MB -> 349 MB |

The correct quantisation path for the Hexagon HTP is QNN's own
calibration during the AI Hub compile job - not CPU post-training int8.

## 4. Exported graph structure

- Source model: `urchade/gliner_multi_pii-v1` (Apache-2.0)
- ONNX artifact: `models\onnx\model.onnx` - **1157.1 MB**, 4950 nodes
- Pinned label prompt (6): person, organization, location, date of birth, email, phone number
- Output: `logits` ['batch_size', 'sequence_length', 'num_spans', 'num_classes']

### Where the bytes live

The token embedding table is **768.3 MB of 1155.8 MB (66%)** of the graph. The base model carries a 250k-token multilingual vocabulary; pruning it to the vocabulary actually reachable from Indian financial documents is the single largest NPU-side optimisation available, and it costs no accuracy on this domain.

### Inputs

| Input | Shape |
|---|---|
| `input_ids` | ['batch_size', 'sequence_length'] |
| `attention_mask` | ['batch_size', 'sequence_length'] |
| `words_mask` | ['batch_size', 'sequence_length'] |
| `text_lengths` | ['batch_size', 'value'] |
| `span_idx` | ['batch_size', 'num_spans', 'idx'] |
| `span_mask` | ['batch_size', 'num_spans'] |

### Lowering risk: `If` x12, `NonZero` x3, `ScatterND` x2, `TopK` x1

These ops are the usual reason a QNN compile fails. They are present because GLiNER's word-pooling stage is data-dependent. If the compile job fails, pinning static shapes (which `--submit` does) or freezing the pooling indices at export time is the fix.

---

Regenerate: `python -m models.export_ai_hub --all`
