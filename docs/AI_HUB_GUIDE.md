# Qualcomm AI Hub — NPU export & profiling

How Redacta's PII detection model gets onto the Hexagon NPU, and how to
reproduce every number in [`benchmarks/npu.md`](../benchmarks/npu.md).

The pipeline lives in `models/export_ai_hub.py`. It replaced an earlier scaffold
that raised `NotImplementedError` — the export, verification, and CPU
benchmarks are now real and run today; only the cloud compile needs an account.

## What the pipeline does

| Step | Command | Needs |
|---|---|---|
| Print the plan, no dependencies | `--dry-run` | nothing |
| Trace model → ONNX with pinned labels | `--export` | `torch`, `gliner`, `onnx` |
| Inspect graph size / ops / shapes | `--inspect` | `onnx` |
| Prove the graph detects real PII | `--verify` | `onnxruntime`, `gliner` |
| Measure CPU latency | `--benchmark` | `onnxruntime` |
| Measure int8 degradation | `--int8-check` | `onnxruntime` |
| Export + verify + bench + write report | `--all` | the above |
| Compile + profile on Hexagon NPU | `--submit` | **AI Hub token** |
| List Snapdragon targets | `--devices` | **AI Hub token** |

```bash
pip install -r requirements-ai.txt

python -m models.export_ai_hub --dry-run     # safe everywhere
python -m models.export_ai_hub --all         # real export + real numbers
python -m models.export_ai_hub --devices     # find your target device name
python -m models.export_ai_hub --submit --device "Snapdragon X Elite CRD"
```

## Getting the API token (the one manual step)

The compile and profile jobs run on Qualcomm's cloud fleet, so they need an
account — no Snapdragon hardware is required to obtain legitimate NPU numbers.

1. Request access at <https://aihub.qualcomm.com/>.
2. Configure the client (writes `~/.qai_hub/client.ini`):
   ```bash
   python -m qai_hub configure --api_token <TOKEN>
   ```
3. Confirm: `python -m models.export_ai_hub --devices`

Until the token exists, `--submit` exits with that instruction and the NPU row
in the report stays `PENDING`. Nothing is ever estimated.

## What the submit job actually does

`--submit` pins the traced graph's dynamic axes to the representative
document's concrete shapes, then:

1. `hub.submit_compile_job(model=model.onnx, device=…, input_specs=…)`
   → QNN context binary for the Hexagon HTP.
2. `hub.submit_profile_job(model=<target model>, device=…)`
   → per-layer latency/utilisation, saved to `benchmarks/ai_hub_profile.json`.
3. Writes the whole result set back into `benchmarks/npu.md`.

On failure it downloads the job log instead of guessing — the most likely cause
with this graph is a non-lowerable op (see the lowering-risk section of the
report).

## Findings worth knowing (all measured)

**The exported graph works.** The ONNX artifact finds real PII end-to-end —
`Ananya Nair`, `State Bank of India, Pune Main Branch`, `Rohan Nair`,
`Meera Iyer`, `HDFC Bank Ltd.`, plus DOB, email and phone — at 0.95–1.00
confidence. It is a working detector, not just a graph that loads.

**Naive int8 quantisation is unusable.** Dynamic int8 shrinks the artifact
1157 MB → 349 MB but destroys the model: argmax agreement drops to **8.9%** and
it detects **0 of 9** entities. It is therefore excluded from the product. The
correct path is QNN's own HTP-aware calibration inside the compile job.

**The token embedding dominates the model.** A 250,105-token multilingual
vocabulary costs **768 MB of 1157 MB (66%)**. Pruning it to the vocabulary
reachable from Indian financial documents is the single biggest NPU-side
optimisation available and costs no accuracy on this domain.

**Some ops may not lower.** The exported graph contains `If` x12, `NonZero` x3,
`ScatterND` x2, `TopK` x1 — from GLiNER's data-dependent word-pooling stage.
`--submit` mitigates by pinning static shapes; if a compile still fails, the fix
is freezing the pooling indices at export time.

## Why the NPU matters here

Measured on CPU: the deterministic regex tier answers a document in **0.27 ms**,
while the AI tier costs **~285 ms** per document (warm). The AI tier is what
finds names and organisations that regex cannot express — so it is exactly the
part that needs the Hexagon HTP to become interactive.

## Honest status

| Item | State |
|---|---|
| ONNX export | done, verified, reproducible |
| CPU benchmarks (graph + pipeline) | done, measured |
| int8 trade-off | measured and rejected |
| QNN compile + Hexagon profile | **pending AI Hub token** |
| Model-size optimisation (vocab prune) | identified, not yet implemented |
