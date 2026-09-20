"""Tests for the Qualcomm AI Hub export pipeline.

These cover the parts that must be correct on every machine: path
portability, graph inspection, and honest AI Hub readiness reporting. The
multi-gigabyte export itself is exercised by `python -m models.export_ai_hub
--all`, which is far too slow for a unit test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import export_ai_hub as ex  # noqa: E402


def test_relative_paths_are_portable():
    """Reports are committed, so paths must not embed a machine's home dir."""
    inside = ex.PROJECT_ROOT / "benchmarks" / "npu.md"
    assert ex._rel(inside) == str(Path("benchmarks") / "npu.md")
    assert str(ex.PROJECT_ROOT) not in ex._rel(inside)


def test_relative_path_leaves_foreign_paths_alone():
    """A path outside the repo is returned as-is rather than crashing."""
    outside = Path("/tmp/somewhere/else/model.onnx") if sys.platform != "win32" else Path("C:/elsewhere/model.onnx")
    assert ex._rel(outside) == str(outside)


def test_pinned_labels_are_non_empty_and_unique():
    assert ex.PII_LABELS, "the exported graph pins labels; an empty prompt is invalid"
    assert len(set(ex.PII_LABELS)) == len(ex.PII_LABELS)


def test_profile_text_contains_indian_pii_vocabulary():
    """The pinned document must actually exercise the label set."""
    text = ex.PROFILE_TEXT.lower()
    for needle in ("aadhaar", "pan", "ifsc", "date of birth", "state bank"):
        assert needle in text, f"profile document no longer exercises {needle!r}"


def test_hub_ready_reports_reason_without_raising():
    """Readiness is a status, not an exception: the local path must survive it."""
    ready, reason = ex.hub_ready()
    assert isinstance(ready, bool)
    assert reason and isinstance(reason, str)
    if not ready:
        # The explanation must be actionable, not a bare "not configured".
        assert "aihub.qualcomm.com" in reason or "qai-hub" in reason


def test_inspect_onnx_reports_structure(tmp_path):
    """Inspection must find ops, shapes and where the parameter bytes went."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx")

    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = torch.nn.Embedding(64, 8)
            self.proj = torch.nn.Linear(8, 4)

        def forward(self, ids: "torch.Tensor") -> "torch.Tensor":
            return self.proj(self.embed(ids))

    onnx_file = tmp_path / "tiny.onnx"
    torch.onnx.export(
        Tiny().eval(),
        (torch.zeros(1, 5, dtype=torch.long),),
        f=str(onnx_file),
        input_names=["input_ids"],
        output_names=["logits"],
        opset_version=19,
        dynamo=False,
    )

    facts = ex.inspect_onnx(onnx_file)
    assert facts.size_mb > 0
    assert facts.node_count > 0
    assert facts.output is not None and facts.output[0] == "logits"
    assert [name for name, _ in facts.inputs] == ["input_ids"]
    assert facts.parameter_bytes > 0
    # No control flow in this trivial graph, so nothing should look risky.
    assert facts.risky_ops == {}


def test_risky_ops_are_the_known_qnn_blockers():
    """Guard the list that drives the report's lowering-risk section."""
    assert "NonZero" in ex.RISKY_OPS and "If" in ex.RISKY_OPS
