"""Model registry for Redacta AI.

Single place that documents every model in the stack, its role, and its
Qualcomm AI Hub export target.  The export script reads this registry.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str                 # HF/AI Hub identifier
    role: str                 # what it does in the pipeline
    runtime: str              # where it runs today (CPU dev) / target (NPU)
    ai_hub_target: str        # AI Hub model type used for export
    optional: bool
    notes: str = ""


MODELS: dict[str, ModelSpec] = {
    "gliner_pii": ModelSpec(
        name="urchade/gliner_multi_pii-v1",
        role="Zero-shot PII NER: names, locations, organisations, DOB",
        runtime="PyTorch (dev) → ONNX Runtime (Windows) → QNN/Hexagon NPU (target)",
        ai_hub_target="ONNX export → QNN context binary (via qai-hub export)",
        optional=True,
        notes="Apache-2.0. Multilingual (8 languages incl. Indian naming).",
    ),
    "ocr": ModelSpec(
        name="Tesseract 5 (or docTR)",
        role="OCR for screenshots and scanned documents",
        runtime="CPU (binary) — replaced by docTR ONNX on NPU roadmap",
        ai_hub_target="docTR (parseq/crnn) ONNX → QNN (roadmap)",
        optional=True,
        notes="pytesseract wrapper; deterministic fallback path.",
    ),
}

NPU_DEVICE = "Snapdragon X Elite / X2 Plus Hexagon NPU (Windows on ARM)"


def summary_lines() -> list[str]:
    lines = [f"Redacta AI model stack — target: {NPU_DEVICE}", ""]
    for key, m in MODELS.items():
        opt = "optional" if m.optional else "required"
        lines.append(f"[{key}] {m.name} ({opt})")
        lines.append(f"    role   : {m.role}")
        lines.append(f"    runtime: {m.runtime}")
        lines.append(f"    AI Hub : {m.ai_hub_target}")
        if m.notes:
            lines.append(f"    notes  : {m.notes}")
        lines.append("")
    return lines
