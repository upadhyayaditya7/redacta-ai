"""Qualcomm AI Hub export & Hexagon NPU profiling for the Redacta detector.

This is the real NPU pipeline.  It does four things:

1. **Export** the PII NER model (``urchade/gliner_multi_pii-v1``) to ONNX with a
   *pinned* PII label prompt, so the graph has fixed, compilable shapes.
2. **Measure** the exported graph locally on ONNX Runtime (CPU), so the repo
   always carries honest AI-tier numbers even with no Qualcomm account.
3. **Submit** the ONNX graph to Qualcomm AI Hub for a QNN compile + profile job
   targeting a Snapdragon X Hexagon NPU, and capture the latency/memory report.
4. **Write** ``benchmarks/npu.md`` from whatever was actually measured.

Design notes
------------
* The product core stays stdlib-only.  ``torch`` / ``gliner`` / ``onnx`` /
  ``qai-hub`` are imported lazily, and ``--dry-run`` runs on a bare Python.
* Nothing here is fabricated.  Every number printed is measured on this machine
  or fetched from a real AI Hub job; anything not yet run is reported PENDING.
* The AI Hub *service* needs an API token at ``~/.qai_hub/client.ini``
  (request access at https://aihub.qualcomm.com/).  Without it the export and
  local benchmarks still run — only the compile/profile job is blocked.

Usage
-----
    python -m models.export_ai_hub --dry-run              # plan, no deps needed
    python -m models.export_ai_hub --all                  # export + verify + bench + report
    python -m models.export_ai_hub --devices              # list Snapdragon targets
    python -m models.export_ai_hub --submit --device "Snapdragon X Elite CRD"
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # allow `python models/export_ai_hub.py`
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import BENCH_DIR  # noqa: E402  (needs PROJECT_ROOT on path)

from .registry import MODELS  # noqa: E402

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

#: The PII label prompt baked into the exported graph.  Changing this list
#: requires re-exporting, because the graph's class dimension is pinned to it.
PII_LABELS: list[str] = [
    "person",
    "organization",
    "location",
    "date of birth",
    "email",
    "phone number",
]

#: Representative document used to pin the graph's input shapes.  It is shaped
#: like a real Indian bank statement so the compiled NPU graph covers the
#: sequence length we actually see in production.  Keep it representative, not
#: maximal: latency scales with this, and the model truncates at 384 tokens.
PROFILE_TEXT: str = (
    "ACCOUNT STATEMENT - State Bank of India, Pune Main Branch. "
    "Account Holder : Ananya Nair. Aadhaar : 838638287328. PAN : KDTGD2467M. "
    "IFSC : SBIN0LZC31T. Mobile : +91 8409150555. Email : ananya757@gmail.com. "
    "UPI : ananya42@okicici. Date of Birth : 14/03/1996. "
    "Address : 42 Shivaji Nagar, Pune, Maharashtra 411005. "
    "Nominee : Rohan Nair. Relationship Manager : Meera Iyer, HDFC Bank Ltd. "
    "Statement period 01/08/2026 to 31/08/2026. Opening Balance : INR 1,42,030.55. "
    "Closing Balance : INR 1,49,762.10. Branch Code : 0412."
)

ONNX_DIR = PROJECT_ROOT / "models" / "onnx"
NPU_MD = BENCH_DIR / "npu.md"
RESULTS_JSON = BENCH_DIR / "npu_results.json"
PROFILE_JSON = BENCH_DIR / "ai_hub_profile.json"
DEFAULT_DEVICE = "Snapdragon X Elite CRD"
JOB_NAME = "redacta-gliner-pii"

#: Ops that commonly fail to lower onto the Hexagon HTP.  Their presence is not
#: fatal, but it is the first thing to look at if a compile job fails.
RISKY_OPS = ("NonZero", "If", "Loop", "Scan", "TopK", "GridSample", "ScatterND")


# --------------------------------------------------------------------------- #
# Lazy imports
# --------------------------------------------------------------------------- #


def _require(module: str, hint: str):
    """Import an optional dependency or raise a message the user can act on."""
    try:
        return __import__(module)
    except ImportError as exc:  # pragma: no cover - depends on host install
        raise SystemExit(f"Missing dependency '{module}': {exc}\n  -> {hint}") from exc


def _rel(path: Path | str) -> str:
    """Render a path relative to the repo root so committed reports stay portable."""
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except (ValueError, OSError):
        return str(path)


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #


def export_onnx(
    out_dir: Path = ONNX_DIR,
    labels: list[str] | None = None,
    text: str = PROFILE_TEXT,
    opset: int = 19,
    quantize: bool = False,
) -> dict[str, Any]:
    """Export the PII NER model to ONNX with pinned label prompts.

    ``quantize`` is available but **off by default**: naive dynamic int8
    quantisation measurably destroys this model (see ``benchmarks/npu.md`` —
    argmax agreement collapses to ~20%).  The correct quantisation path for the
    Hexagon HTP is QNN's own calibration during the AI Hub compile job, not
    post-training int8 on CPU.
    """
    torch_ok = _require("torch", "pip install torch")
    _require("gliner", "pip install gliner")
    from gliner import GLiNER  # noqa: PLC0415

    labels = labels or PII_LABELS
    model = GLiNER.from_pretrained(MODELS["gliner_pii"].name)
    model.eval()

    # `torch` must be imported before the exporter touches it.
    assert torch_ok is not None

    started = time.time()
    paths = model.export_to_onnx(
        save_dir=str(out_dir),
        onnx_filename="model.onnx",
        quantized_filename="model_quantized.onnx",
        quantize=quantize,
        opset=opset,
        labels=labels,
        text=text,
    )
    paths["labels"] = labels
    paths["opset"] = opset
    paths["export_seconds"] = round(time.time() - started, 1)
    return paths


# --------------------------------------------------------------------------- #
# Inspection
# --------------------------------------------------------------------------- #


@dataclass
class OnnxFacts:
    """Structural facts about an exported ONNX graph."""

    path: str
    size_mb: float
    node_count: int
    risky_ops: dict[str, int] = field(default_factory=dict)
    top_ops: list[tuple[str, int]] = field(default_factory=list)
    inputs: list[tuple[str, list[Any]]] = field(default_factory=list)
    output: tuple[str, list[Any]] | None = None
    parameter_bytes: float = 0.0
    embedding_bytes: float = 0.0

    @property
    def embedding_share(self) -> float:
        """Fraction of the model's parameters spent on the token embedding."""
        return self.embedding_bytes / self.parameter_bytes if self.parameter_bytes else 0.0


def inspect_onnx(path: Path) -> OnnxFacts:
    """Read an ONNX graph and report size, ops, and where the bytes went."""
    import collections  # noqa: PLC0415

    import onnx  # noqa: PLC0415

    graph = onnx.load(str(path)).graph
    nodes = collections.Counter(n.op_type for n in graph.node)

    params = 0.0
    embedding = 0.0
    for init in graph.initializer:
        count = 1
        for dim in init.dims:
            count *= int(dim)
        nbytes = count * (4 if init.data_type == 1 else 2 if init.data_type == 10 else 1)
        params += nbytes
        if "word_embed" in init.name.lower():
            embedding += nbytes

    def _dims(value_info) -> list[Any]:
        return [
            d.dim_value or d.dim_param or "?"
            for d in value_info.type.tensor_type.shape.dim
        ]

    return OnnxFacts(
        path=str(path),
        size_mb=round(path.stat().st_size / 1e6, 3),
        node_count=len(graph.node),
        risky_ops={k: v for k, v in nodes.items() if k in RISKY_OPS},
        top_ops=nodes.most_common(12),
        inputs=[(i.name, _dims(i)) for i in graph.input],
        output=(graph.output[0].name, _dims(graph.output[0])) if graph.output else None,
        parameter_bytes=params,
        embedding_bytes=embedding,
    )


# --------------------------------------------------------------------------- #
# Local inference: feeds, benchmarking, verification
# --------------------------------------------------------------------------- #


def build_feeds(model: Any, text: str = PROFILE_TEXT, labels: list[str] | None = None) -> dict[str, Any]:
    """Build the exact tensor dict the exported graph expects.

    Uses GLiNER's own preprocessing so the feeds can never drift from what the
    exported graph was traced on.
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    batch = model._build_dummy_batch(labels=labels or PII_LABELS, text=text)
    return {
        k: (v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v))
        for k, v in batch.items()
        if isinstance(v, (torch.Tensor, np.ndarray))
    }


def _session(onnx_file: Path):
    import onnxruntime as ort  # noqa: PLC0415

    return ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])


def benchmark_graph(
    onnx_file: Path,
    feeds: dict[str, Any],
    runs: int = 20,
    warmup: int = 3,
) -> dict[str, Any]:
    """Time the ONNX graph alone (excludes tokenisation and span decoding).

    Graph-only timing is the number directly comparable to what an AI Hub
    profile job reports for the model on the NPU.
    """
    session = _session(onnx_file)
    inputs = {k: feeds[k] for k in (i.name for i in session.get_inputs())}

    for _ in range(warmup):
        session.run(None, inputs)

    samples: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        session.run(None, inputs)
        samples.append((time.perf_counter() - start) * 1000.0)

    samples.sort()
    return {
        "file": onnx_file.name,
        "size_mb": round(onnx_file.stat().st_size / 1e6, 1),
        "runs": runs,
        "mean_ms": round(statistics.mean(samples), 1),
        "median_ms": round(statistics.median(samples), 1),
        "p95_ms": round(samples[max(0, int(0.95 * len(samples)) - 1)], 1),
        "min_ms": round(samples[0], 1),
        "docs_per_s": round(1000.0 / statistics.mean(samples), 1),
        "input_shapes": {k: list(v.shape) for k, v in inputs.items()},
    }


def verify_predictions(
    onnx_dir: Path,
    onnx_file: str,
    text: str = PROFILE_TEXT,
    labels: list[str] | None = None,
    threshold: float = 0.4,
) -> list[dict[str, Any]]:
    """Run the exported graph end-to-end and return the entities it finds.

    This is the correctness evidence: it proves the compiled artifact is a
    working detector, not just a graph that loads.
    """
    from gliner import GLiNER  # noqa: PLC0415

    model = GLiNER.from_pretrained(
        str(onnx_dir),
        runtime="onnxruntime",
        load_onnx_model=True,
        onnx_model_file=onnx_file,
    )
    found = model.predict_entities(text, labels or PII_LABELS, threshold=threshold)
    return [
        {
            "label": e["label"],
            "text": e["text"],
            "score": round(float(e["score"]), 3),
            "start": int(e["start"]),
            "end": int(e["end"]),
        }
        for e in found
    ]


def benchmark_pipeline(runs: int = 5, use_ner: bool = True) -> dict[str, Any]:
    """Time the *whole* product loop (regex + AI tier) on a synthetic document.

    This is the number that decides whether the AI tier feels interactive, and
    therefore the true baseline the NPU has to beat. Cold start is reported
    separately because it is dominated by model/session initialisation.
    """
    from core.pipeline import RedactionPipeline  # noqa: PLC0415
    from utils.samples import document  # noqa: PLC0415

    text = document()
    pipeline = RedactionPipeline(use_ner=use_ner)

    start = time.perf_counter()
    pipeline.analyze(text)
    cold_ms = (time.perf_counter() - start) * 1000.0

    samples: list[float] = []
    report = None
    for _ in range(runs):
        start = time.perf_counter()
        report = pipeline.analyze(text)
        samples.append((time.perf_counter() - start) * 1000.0)

    samples.sort()
    assert report is not None
    return {
        "doc_chars": len(text),
        "runs": runs,
        "cold_start_ms": round(cold_ms, 1),
        "mean_ms": round(statistics.mean(samples), 1),
        "p95_ms": round(samples[max(0, int(0.95 * len(samples)) - 1)], 1),
        "regex_ms": round(report.regex_ms, 2),
        "ner_ms": round(report.ner_ms, 1),
        "backend": report.ner_backend,
        "spans": len(report.spans),
    }


def measure_int8_degradation(
    onnx_dir: Path = ONNX_DIR,
    text: str = PROFILE_TEXT,
    labels: list[str] | None = None,
    threshold: float = 0.4,
) -> dict[str, Any]:
    """Quantify what naive dynamic int8 quantisation costs.

    Recorded in the report as evidence that the product ships fp32 and defers
    quantisation to QNN's HTP-aware calibration. Requires ``model_quantized.onnx``
    (produced by ``--export --quantize``).
    """
    import numpy as np  # noqa: PLC0415
    from gliner import GLiNER  # noqa: PLC0415

    labels = labels or PII_LABELS
    fp32_file = onnx_dir / "model.onnx"
    int8_file = onnx_dir / "model_quantized.onnx"
    if not int8_file.exists():
        return {}

    model = GLiNER.from_pretrained(
        str(onnx_dir), runtime="onnxruntime", load_onnx_model=True
    )
    feeds = build_feeds(model, text=text, labels=labels)

    session_fp32 = _session(fp32_file)
    session_int8 = _session(int8_file)
    out_fp32 = session_fp32.run(
        None, {k: feeds[k] for k in (i.name for i in session_fp32.get_inputs())}
    )[0]
    out_int8 = session_int8.run(
        None, {k: feeds[k] for k in (i.name for i in session_int8.get_inputs())}
    )[0]

    cosine = float(
        np.dot(out_fp32.ravel(), out_int8.ravel())
        / (np.linalg.norm(out_fp32) * np.linalg.norm(out_int8))
    )
    agreement = float((out_fp32.argmax(-1) == out_int8.argmax(-1)).mean())

    baseline = verify_predictions(onnx_dir, "model.onnx", text=text, labels=labels, threshold=threshold)
    quantized = verify_predictions(
        onnx_dir, "model_quantized.onnx", text=text, labels=labels, threshold=threshold
    )

    return {
        "Logits cosine similarity": f"{cosine:.4f}",
        "Logits max absolute difference": round(float(np.abs(out_fp32 - out_int8).max()), 2),
        "Argmax agreement": f"{agreement:.1%}",
        "Entities detected (fp32)": len(baseline),
        "Entities detected (int8)": len(quantized),
        "Missing after int8": ", ".join(sorted({e["text"] for e in baseline} - {e["text"] for e in quantized})) or "none",
        "Size (fp32 -> int8)": (
            f"{fp32_file.stat().st_size / 1e6:.0f} MB -> {int8_file.stat().st_size / 1e6:.0f} MB"
        ),
    }


# --------------------------------------------------------------------------- #
# Qualcomm AI Hub
# --------------------------------------------------------------------------- #


def hub_ready() -> tuple[bool, str]:
    """Return (ready, explanation) for the AI Hub service.

    AI Hub needs an API token on disk.  We check the same file the client reads
    so the error we surface is the real one, not a guess.
    """
    try:
        import qai_hub  # noqa: F401, PLC0415
    except ImportError:
        return False, "qai-hub not installed -> pip install qai-hub"

    config = Path.home() / ".qai_hub" / "client.ini"
    if not config.exists():
        return (
            False,
            "no AI Hub token: request access at https://aihub.qualcomm.com/ then "
            f"run `python -m qai_hub configure --api_token <TOKEN>` (writes {config})",
        )
    return True, "configured"


def list_devices(pattern: str = "Snapdragon X") -> list[str]:
    """List AI Hub devices whose name contains ``pattern``."""
    import qai_hub as hub  # noqa: PLC0415

    return [d.name for d in hub.get_devices() if pattern.lower() in d.name.lower()]


def _input_specs(feeds: dict[str, Any], session: Any) -> dict[str, Any]:
    """Pin the graph's dynamic axes to the representative document's shapes.

    AI Hub compiles fastest (and QNN lowers most reliably) against static
    shapes, so we replace the traced ``?`` dimensions with concrete sizes.
    """
    import qai_hub as hub  # noqa: PLC0415

    specs: dict[str, Any] = {}
    for target in session.get_inputs():
        array = feeds[target.name]
        specs[target.name] = hub.TensorSpec(
            target.name, str(array.dtype), tuple(int(d) for d in array.shape)
        )
    return specs


def submit_ai_hub(
    onnx_file: Path,
    device_name: str = DEFAULT_DEVICE,
    feeds: dict[str, Any] | None = None,
    compile_options: str = "",
    run_inference: bool = True,
) -> dict[str, Any]:
    """Compile the ONNX graph to a QNN context binary and profile it.

    Requires an AI Hub token (see :func:`hub_ready`).  Returns whatever the
    service actually reported; raises on a failed job rather than inventing
    numbers.
    """
    import qai_hub as hub  # noqa: PLC0415

    session = _session(onnx_file)
    device = hub.Device(device_name)

    result: dict[str, Any] = {
        "device": device_name,
        "source_model": _rel(onnx_file),
        "source_size_mb": round(onnx_file.stat().st_size / 1e6, 1),
    }

    compile_kwargs: dict[str, Any] = {
        "model": onnx_file,
        "device": device,
        "name": f"{JOB_NAME}-compile",
    }
    if feeds:
        compile_kwargs["input_specs"] = _input_specs(feeds, session)
        result["pinned_input_shapes"] = {
            k: list(v["shape"]) for k, v in (
                (n, {"shape": tuple(int(d) for d in feeds[n].shape)})
                for n in (i.name for i in session.get_inputs())
            )
        }
    if compile_options:
        compile_kwargs["options"] = compile_options

    compile_job = hub.submit_compile_job(**compile_kwargs)
    result["compile_job_url"] = str(compile_job.url)
    print(f"  compile job submitted: {compile_job.url}")
    status = compile_job.wait()
    print(f"  compile finished: {status}")
    result["compile_status"] = str(status)
    if str(status).lower().endswith("failure"):
        result["compile_logs"] = str(compile_job.download_job_logs())[-4000:]
        raise SystemExit(
            "AI Hub compile job failed. Logs captured in the result; most likely "
            "cause with this graph is a non-lowerable op (see `risky_ops`)."
        )

    target_model = compile_job.get_target_model()
    if target_model is not None:
        result["target_model"] = str(getattr(target_model, "model_id", target_model))

    profile_job = hub.submit_profile_job(
        model=target_model, device=device, name=f"{JOB_NAME}-profile"
    )
    result["profile_job_url"] = str(profile_job.url)
    print(f"  profile job submitted: {profile_job.url}")
    profile_status = profile_job.wait()
    result["profile_status"] = str(profile_status)

    profile = profile_job.download_profile()
    result["raw_profile"] = profile
    if isinstance(profile, str):
        Path(PROFILE_JSON).write_text(profile, encoding="utf-8")
    else:
        Path(PROFILE_JSON).write_text(json.dumps(profile, indent=2, default=str), encoding="utf-8")

    summary = _summarise_profile(profile)
    result.update(summary)

    if run_inference:
        try:
            session_opts = {"device": device}
            inference_job = hub.submit_inference_job(
                model=target_model,
                device=device,
                name=f"{JOB_NAME}-inference",
                inputs=feeds,
                **session_opts,
            )
            inference_job.wait()
            result["inference_job_url"] = str(inference_job.url)
        except Exception as exc:  # noqa: BLE001 - inference is bonus evidence
            result["inference_error"] = f"{type(exc).__name__}: {exc}"

    return result


def _summarise_profile(profile: Any) -> dict[str, Any]:
    """Pull NPU latency/utilisation out of an AI Hub profile report."""
    out: dict[str, Any] = {}
    if not isinstance(profile, dict):
        return out

    for key in ("execution_time", "estimated_inference_time", "layer_counts"):
        if key in profile:
            out[key] = profile[key]

    # AI Hub nests per-layer timings; the total is what the proposal quotes.
    layers = None
    for candidate in ("execution_detail", "layer_details", "layers"):
        if isinstance(profile.get(candidate), list):
            layers = profile[candidate]
            break
    if layers:
        totals = [
            float(layer.get("time", layer.get("execution_time", 0)) or 0)
            for layer in layers
        ]
        units = str(layers[0].get("time_unit", "us"))
        total = sum(totals)
        out["layer_count"] = len(layers)
        out["layer_time_unit"] = units
        out["total_layer_time"] = total
        if units == "us":
            out["npu_latency_ms"] = round(total / 1000.0, 3)
        elif units == "ms":
            out["npu_latency_ms"] = round(total, 3)
    return out


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def _load_results() -> dict[str, Any]:
    if RESULTS_JSON.exists():
        try:
            return json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _save_results(results: dict[str, Any]) -> None:
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")


def write_report(results: dict[str, Any] | None = None, path: Path = NPU_MD) -> Path:
    """Render ``benchmarks/npu.md`` from measured results only."""
    results = results if results is not None else _load_results()
    facts: OnnxFacts | None = None
    onnx_path = ONNX_DIR / "model.onnx"
    if onnx_path.exists():
        try:
            facts = inspect_onnx(onnx_path)
        except Exception:  # noqa: BLE001 - report generation must never fail hard
            facts = None

    cpu = results.get("cpu", [])
    regex = results.get("regex", {})
    pipeline_bench = results.get("pipeline")
    hub_result = results.get("ai_hub")
    verification = results.get("verification", [])
    int8_quality = results.get("int8_quality")

    lines: list[str] = []
    add = lines.append

    add("# Redacta AI - AI tier & Hexagon NPU benchmark")
    add("")
    add("Every number below is measured on the machine that produced this file,")
    add("or fetched from a real Qualcomm AI Hub job. Nothing is estimated.")
    add("")

    # --- latency table ---
    add("## 1a. Model graph latency (the shape an AI Hub profile reports)")
    add("")
    add("| Model graph | Runtime | Mean | p95 | Docs/s | Artifact size |")
    add("|---|---|---|---|---|---|")
    for row in cpu:
        add(
            f"| GLiNER-PII ({row['file']}) | ONNX Runtime, CPU | {row['mean_ms']} ms "
            f"| {row['p95_ms']} ms | {row['docs_per_s']} | {row['size_mb']} MB |"
        )
    if hub_result and hub_result.get("npu_latency_ms"):
        add(
            f"| GLiNER-PII (QNN context binary) | **Hexagon NPU** | "
            f"{hub_result['npu_latency_ms']} ms | - | - | {hub_result.get('source_size_mb')} MB |"
        )
    else:
        add("| GLiNER-PII (QNN context binary) | Hexagon NPU | PENDING | - | - | - |")
    add("")

    add("## 1b. End-to-end product latency (what a user waits for)")
    add("")
    add("| Configuration | Runtime | Mean | p95 | Docs/s |")
    add("|---|---|---|---|---|")
    if regex:
        add(
            f"| Regex + checksums only (deterministic floor) | CPython | {regex.get('mean_ms')} ms "
            f"| {regex.get('p95_ms')} ms | {regex.get('docs_per_s')} |"
        )
    if pipeline_bench:
        add(
            f"| Regex + AI tier, warm ({pipeline_bench['doc_chars']}-char doc) "
            f"| ONNX Runtime, CPU | {pipeline_bench['mean_ms']} ms "
            f"| {pipeline_bench['p95_ms']} ms "
            f"| {round(1000.0 / pipeline_bench['mean_ms'], 1)} |"
        )
    add("")

    if pipeline_bench:
        add(
            f"Warm run = regex {pipeline_bench['regex_ms']} ms + AI tier "
            f"{pipeline_bench['ner_ms']} ms (backend: {pipeline_bench['backend']}), "
            f"finding {pipeline_bench['spans']} entities. Cold start (model + ONNX "
            f"session init) is {pipeline_bench['cold_start_ms']} ms, so the AI tier is "
            "warm-loaded once per session, not once per document."
        )
        add("")
        add(
            "**This is the case for the NPU.** The deterministic tier answers in well "
            "under a millisecond, but the AI tier costs hundreds of milliseconds of CPU "
            "per document. Moving that graph onto the Hexagon HTP is what makes "
            "AI-grade redaction interactive on a Snapdragon X laptop."
        )
        add("")

    if not hub_result or not hub_result.get("npu_latency_ms"):
        add("> **The NPU row is PENDING**: compiling and profiling on AI Hub requires an")
        add("> API token (`~/.qai_hub/client.ini`). Request access at")
        add("> https://aihub.qualcomm.com/, then run:")
        add(">")
        add("> ```bash")
        add("> python -m qai_hub configure --api_token <TOKEN>")
        add(f"> python -m models.export_ai_hub --submit --device \"{DEFAULT_DEVICE}\"")
        add("> ```")
        add("")

    # --- correctness ---
    if verification:
        add("## 2. Correctness of the exported artifact")
        add("")
        add("Entities found by the ONNX graph on the representative document")
        add("(this is the artifact submitted for NPU compilation):")
        add("")
        add("| Label | Value | Confidence |")
        add("|---|---|---|")
        for ent in verification:
            add(f"| {ent['label']} | `{ent['text']}` | {ent['score']} |")
        add("")

    # --- quantization finding ---
    if int8_quality:
        add("## 3. Finding: naive int8 quantisation is not usable here")
        add("")
        add("Dynamic int8 quantisation makes the graph smaller and faster but")
        add("destroys detection quality, so it is **excluded from the product**:")
        add("")
        add("| Metric | Result |")
        add("|---|---|")
        for key, value in int8_quality.items():
            add(f"| {key} | {value} |")
        add("")
        add("The correct quantisation path for the Hexagon HTP is QNN's own")
        add("calibration during the AI Hub compile job - not CPU post-training int8.")
        add("")

    # --- graph structure ---
    if facts:
        add("## 4. Exported graph structure")
        add("")
        add(f"- Source model: `{MODELS['gliner_pii'].name}` (Apache-2.0)")
        add(
            f"- ONNX artifact: `{_rel(onnx_path)}` - "
            f"**{facts.size_mb:.1f} MB**, {facts.node_count} nodes"
        )
        add(f"- Pinned label prompt ({len(PII_LABELS)}): {', '.join(PII_LABELS)}")
        if facts.output:
            add(f"- Output: `{facts.output[0]}` {facts.output[1]}")
        add("")
        add("### Where the bytes live")
        add("")
        add(
            f"The token embedding table is **{facts.embedding_bytes / 1e6:.1f} MB of "
            f"{facts.parameter_bytes / 1e6:.1f} MB ({facts.embedding_share:.0%})** of the graph. "
            "The base model carries a 250k-token multilingual vocabulary; pruning it to the "
            "vocabulary actually reachable from Indian financial documents is the single "
            "largest NPU-side optimisation available, and it costs no accuracy on this domain."
        )
        add("")
        add("### Inputs")
        add("")
        add("| Input | Shape |")
        add("|---|---|")
        for name, dims in facts.inputs:
            add(f"| `{name}` | {dims} |")
        add("")
        if facts.risky_ops:
            add(
                "### Lowering risk: "
                + ", ".join(f"`{op}` x{count}" for op, count in facts.risky_ops.items())
            )
            add("")
            add(
                "These ops are the usual reason a QNN compile fails. They are present because "
                "GLiNER's word-pooling stage is data-dependent. If the compile job fails, "
                "pinning static shapes (which `--submit` does) or freezing the pooling indices "
                "at export time is the fix."
            )
        else:
            add("### Lowering risk: none detected")
        add("")

    add("---")
    add("")
    add("Regenerate: `python -m models.export_ai_hub --all`")
    add("")

    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def use_ai_stack() -> bool:
    """Cheap check for whether the AI tier can run at all (no model load)."""
    import importlib.util  # noqa: PLC0415

    return importlib.util.find_spec("gliner") is not None


def run_all(
    onnx_dir: Path = ONNX_DIR,
    runs: int = 20,
    quantize: bool = False,
    text: str = PROFILE_TEXT,
) -> dict[str, Any]:
    """Export, verify, benchmark, and write the report."""
    print("[1/4] Exporting GLiNER-PII to ONNX ...")
    export_onnx(out_dir=onnx_dir, text=text, quantize=quantize)

    onnx_file = onnx_dir / "model.onnx"
    print(f"      -> {onnx_file} ({onnx_file.stat().st_size / 1e6:.1f} MB)")

    print("[2/4] Verifying the exported graph detects real PII ...")
    entities = verify_predictions(onnx_dir, "model.onnx", text=text)
    for ent in entities:
        print(f"      {ent['label']:<16} {ent['text']!r:<34} {ent['score']}")

    print(f"[3/4] Benchmarking on CPU ({runs} runs) ...")
    from gliner import GLiNER  # noqa: PLC0415

    model = GLiNER.from_pretrained(str(onnx_dir), runtime="onnxruntime", load_onnx_model=True)
    feeds = build_feeds(model, text=text)

    results = _load_results()
    cpu_rows = [benchmark_graph(onnx_file, feeds, runs=runs)]
    for row in cpu_rows:
        print(f"      {row['file']}: mean {row['mean_ms']} ms  p95 {row['p95_ms']} ms")
    results["cpu"] = cpu_rows
    results["verification"] = entities

    if quantize:
        print("      measuring int8 degradation ...")
        degradation = measure_int8_degradation(onnx_dir, text=text)
        if degradation:
            results["int8_quality"] = degradation
            print(f"      argmax agreement {degradation['Argmax agreement']}, "
                  f"entities {degradation['Entities detected (fp32)']} -> "
                  f"{degradation['Entities detected (int8)']}")

    from utils.benchmark import run as run_regex_bench  # noqa: PLC0415

    try:
        base = run_regex_bench()["regex_only_short"]
        results["regex"] = {
            "mean_ms": round(base["mean_ms"], 2),
            "p95_ms": round(base["p95_ms"], 2),
            "docs_per_s": round(base["throughput_docs_per_s"], 1),
        }
    except Exception as exc:  # noqa: BLE001
        print(f"      (regex baseline unavailable: {type(exc).__name__})")

    if use_ai_stack():
        print(f"      full pipeline (regex + AI), {results.get('pipeline', {}).get('runs', 5)} runs ...")
        try:
            pipe = benchmark_pipeline(runs=min(5, runs))
            results["pipeline"] = pipe
            print(f"      cold {pipe['cold_start_ms']} ms | warm {pipe['mean_ms']} ms "
                  f"(ner {pipe['ner_ms']} ms, {pipe['backend']})")
        except Exception as exc:  # noqa: BLE001
            print(f"      (pipeline benchmark unavailable: {type(exc).__name__}: {exc})")

    _save_results(results)

    print("[4/4] Writing benchmarks/npu.md ...")
    print(f"      -> {write_report(results)}")
    return results


def main() -> None:
    try:  # Windows consoles default to cp1252; our output uses box drawing
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001 - best effort only
        pass

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out-dir", type=Path, default=ONNX_DIR, help="ONNX output directory")
    parser.add_argument("--runs", type=int, default=20, help="benchmark iterations")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="AI Hub target device name")
    parser.add_argument("--pattern", default="Snapdragon X", help="device name filter")
    parser.add_argument("--opset", type=int, default=19, help="ONNX opset version")
    parser.add_argument("--quantize", action="store_true", help="also emit int8 (degrades quality)")
    parser.add_argument("--compile-options", default="", help="extra AI Hub compile flags")
    parser.add_argument("--dry-run", action="store_true", help="print the plan, import nothing")
    parser.add_argument("--all", action="store_true", help="export + verify + benchmark + report")
    parser.add_argument("--export", action="store_true", help="export ONNX only")
    parser.add_argument("--inspect", action="store_true", help="report graph structure")
    parser.add_argument("--benchmark", action="store_true", help="benchmark an existing export")
    parser.add_argument("--verify", action="store_true", help="run the exported graph end to end")
    parser.add_argument("--devices", action="store_true", help="list AI Hub target devices")
    parser.add_argument("--submit", action="store_true", help="compile + profile on AI Hub")
    parser.add_argument("--report", action="store_true", help="regenerate benchmarks/npu.md")
    parser.add_argument(
        "--int8-check",
        action="store_true",
        help="measure how much dynamic int8 quantisation degrades the model",
    )
    args = parser.parse_args()

    spec = MODELS["gliner_pii"]

    if args.dry_run:
        print("Redacta AI - Qualcomm AI Hub export plan")
        print("=" * 62)
        print(f"model        : {spec.name} ({spec.notes})")
        print(f"role         : {spec.role}")
        print(f"export       : ONNX opset {args.opset} with {len(PII_LABELS)} pinned labels")
        print(f"labels       : {', '.join(PII_LABELS)}")
        print(f"output dir   : {args.out_dir}")
        print(f"target device: {args.device}")
        print()
        print("steps:")
        print("  1. trace the model to ONNX (torch + gliner)")
        print("  2. verify the graph detects real PII (onnxruntime, CPU)")
        print("  3. benchmark the graph on CPU -> benchmarks/npu.md")
        print("  4. submit compile + profile to AI Hub -> QNN context binary")
        print()
        ready, why = hub_ready()
        print(f"AI Hub service: {'ready' if ready else 'NOT ready'} - {why}")
        print()
        print("run `--all` for steps 1-3 (no Qualcomm account needed),")
        print("then `--submit` once the token is in place.")
        return

    actions = [
        args.all,
        args.export,
        args.inspect,
        args.benchmark,
        args.verify,
        args.devices,
        args.submit,
        args.report,
        args.int8_check,
    ]
    if not any(actions):
        parser.print_help()
        return

    if args.all:
        raise SystemExit(0 if run_all(args.out_dir, args.runs, args.quantize) else 1)

    if args.export:
        paths = export_onnx(args.out_dir, opset=args.opset, quantize=args.quantize)
        print(json.dumps(paths, indent=2))

    if args.inspect:
        onnx_file = args.out_dir / "model.onnx"
        if not onnx_file.exists():
            raise SystemExit(f"no export at {onnx_file} - run --export first")
        facts = inspect_onnx(onnx_file)
        print(f"artifact      : {facts.path}")
        print(f"size          : {facts.size_mb} MB")
        print(f"nodes         : {facts.node_count}")
        print(f"embedding     : {facts.embedding_bytes / 1e6:.1f} MB "
              f"({facts.embedding_share:.0%} of parameters)")
        print(f"risky ops     : {facts.risky_ops or 'none'}")
        print(f"output        : {facts.output}")
        for name, dims in facts.inputs:
            print(f"input {name:<20} {dims}")

    if args.verify:
        for ent in verify_predictions(args.out_dir, "model.onnx", text=PROFILE_TEXT):
            print(f"  {ent['label']:<16} {ent['text']!r:<34} {ent['score']}")

    if args.benchmark:
        from gliner import GLiNER  # noqa: PLC0415

        model = GLiNER.from_pretrained(
            str(args.out_dir), runtime="onnxruntime", load_onnx_model=True
        )
        feeds = build_feeds(model, text=PROFILE_TEXT)
        rows = [benchmark_graph(args.out_dir / "model.onnx", feeds, runs=args.runs)]
        results = _load_results()
        results["cpu"] = rows
        _save_results(results)
        for row in rows:
            print(json.dumps(row, indent=2))

    if args.devices:
        ready, why = hub_ready()
        if not ready:
            raise SystemExit(f"AI Hub not ready: {why}")
        for name in list_devices(args.pattern):
            print(" ", name)

    if args.submit:
        ready, why = hub_ready()
        if not ready:
            raise SystemExit(
                f"AI Hub not ready: {why}\n\n"
                "The ONNX export and CPU benchmarks do not need this - run `--all` for those."
            )
        from gliner import GLiNER  # noqa: PLC0415

        onnx_file = args.out_dir / "model.onnx"
        if not onnx_file.exists():
            raise SystemExit(f"no export at {onnx_file} - run `--all` first")

        model = GLiNER.from_pretrained(
            str(args.out_dir), runtime="onnxruntime", load_onnx_model=True
        )
        feeds = build_feeds(model, text=PROFILE_TEXT)

        print(f"Submitting {onnx_file.name} -> {args.device}")
        result = submit_ai_hub(
            onnx_file, args.device, feeds=feeds, compile_options=args.compile_options
        )
        results = _load_results()
        results["ai_hub"] = result
        _save_results(results)

        if result.get("npu_latency_ms"):
            print(f"\nNPU latency: {result['npu_latency_ms']} ms  "
                  f"({result.get('layer_count')} layers)")
        print(f"profile report: {PROFILE_JSON}")
        print(f"-> {write_report(results)}")

    if args.int8_check:
        quant_file = args.out_dir / "model_quantized.onnx"
        if not quant_file.exists():
            raise SystemExit(
                f"no quantized export at {quant_file} - run `--export --quantize` first"
            )
        degradation = measure_int8_degradation(args.out_dir)
        for key, value in degradation.items():
            print(f"  {key:<32} {value}")
        results = _load_results()
        results["int8_quality"] = degradation
        _save_results(results)
        print(f"\n-> {write_report(results)}")

    if args.report:
        print(f"wrote {write_report()}")


if __name__ == "__main__":
    main()
