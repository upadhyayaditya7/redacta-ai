"""Export GLiNER-PII via Qualcomm AI Hub → QNN for Hexagon NPU.

This is the scaffold for the NPU leg of the project.  Run it on any
machine (AI Hub compiles in the cloud and returns a profiling report —
no Snapdragon hardware required for the numbers).

Setup:
    pip install qai-hub
    qai-hub configure --api_token <YOUR_TOKEN>   # from ai.qai-hub.com

Then:
    python -m models.export_ai_hub --device "Snapdragon X Elite CRD"

The script:
 1. loads urchade/gliner_multi_pii-v1 (Apache-2.0),
 2. traces it to ONNX with a fixed PII-label prompt,
 3. submits a compile + profile job to AI Hub for a Snapdragon device,
 4. prints NPU latency + memory so it can be pasted into the proposal.
"""

from __future__ import annotations

import argparse

from .registry import MODELS


def export(device: str, dry_run: bool = False) -> int:
    spec = MODELS["gliner_pii"]
    print(f"Model     : {spec.name}")
    print(f"Target    : {device}")
    print(f"AI Hub    : {spec.ai_hub_target}")

    if dry_run:
        print("\n[dry-run] Would now: trace model → ONNX → qai_hub.submit_compile_job")
        return 0

    try:
        import qai_hub as hub  # type: ignore[import-not-found]
        import torch  # type: ignore[import-not-found]
        from gliner import GLiNER  # type: ignore[import-not-found]
    except ImportError as exc:
        print(f"\nMissing dependency: {exc}")
        print("pip install qai-hub torch gliner  (torch is required for tracing)")
        return 1

    print("\n[1/4] Loading model…")
    model = GLiNER.from_pretrained(spec.name)
    model.eval()

    print("[2/4] Tracing to ONNX…")
    # GLiNER forward is dynamic; for AI Hub we trace the encoder with a
    # representative input shape.  Exact wrapper code lands with the
    # profiling milestone — see proposal milestone M2.
    raise NotImplementedError(
        "Tracing wrapper lands in milestone M2 (see docs/PROPOSAL.md). "
        "Scaffold + CLI are in place so the workflow is ready."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="Snapdragon X Elite CRD", help="AI Hub device name")
    ap.add_argument("--dry-run", action="store_true", help="Print the plan without exporting")
    args = ap.parse_args()
    raise SystemExit(export(args.device, args.dry_run))


if __name__ == "__main__":
    main()
