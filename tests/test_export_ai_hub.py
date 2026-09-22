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


def _fake_session_feeds():
    """The graph's real input names/dtypes without loading the 1.16 GB model."""
    from types import SimpleNamespace

    class Arr:
        def __init__(self, shape, dtype):
            self.shape, self.dtype = shape, dtype

    names = [
        ("input_ids", (1, 245), "int64"),
        ("attention_mask", (1, 245), "int64"),
        ("span_idx", (1, 1548, 2), "int64"),
        ("span_mask", (1, 1548), "int64"),
    ]
    session = SimpleNamespace(get_inputs=lambda: [SimpleNamespace(name=n) for n, _, _ in names])
    feeds = {n: Arr(s, d) for n, s, d in names}
    return session, feeds


def test_input_specs_use_shape_and_dtype_tuples():
    """Regression: TensorSpec NamedTuples failed AI Hub's InputSpecs contract.

    The service rejected job j5wl8r84p with "dynamic shapes" because
    ``TensorSpec(name, dtype, shape)`` puts a *string* at ``spec[0]``, so
    the serializer saw no shape at all.  Every value must be
    ``((dim, ...), dtype)`` with concrete dims >= 1.
    """
    session, feeds = _fake_session_feeds()
    specs = ex._input_specs(feeds, session)
    assert set(specs) == {"input_ids", "attention_mask", "span_idx", "span_mask"}
    for name, spec in specs.items():
        assert isinstance(spec, tuple) and len(spec) == 2, f"{name}: expected (shape, dtype)"
        shape, dtype = spec
        assert isinstance(shape, tuple) and all(isinstance(d, int) and d >= 1 for d in shape)
        assert isinstance(dtype, str) and dtype
        # dtype must be the numpy/ONNX string, not a repr() or TensorSpec
        assert " " not in dtype and "TensorSpec" not in dtype


def test_input_specs_serialize_with_real_dtypes():
    """Round-trip through qai_hub's own serializer: int64 must stay int64.

    The shape-only shorthand silently defaults dtype to float32 — which
    would mis-type every input of this all-int64 graph.
    """
    api_utils = pytest.importorskip("qai_hub.api_utils")
    session, feeds = _fake_session_feeds()
    specs = ex._input_specs(feeds, session)
    pb = api_utils.input_shapes_to_tensor_type_list_pb(specs)
    got = {t.name: (tuple(t.tensor_type.shape), t.tensor_type.dtype) for t in pb.types}
    assert got["input_ids"][0] == (1, 245)
    assert got["span_idx"][0] == (1, 1548, 2)
    # int64 must serialize to the int64 enum, not the shape-only
    # shorthand's float32 default
    int64 = api_utils.get_type_value_from_str("int64")
    float32 = api_utils.get_type_value_from_str("float32")
    assert int64 != float32
    assert got["input_ids"][1] == int64


def _tiny_int64_reducemax(tmp_path):
    """The smallest graph that reproduces HTP's rejected op (error 3110)."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper  # noqa: PLC0415

    inp = helper.make_tensor_value_info("x", TensorProto.INT64, [3])
    out = helper.make_tensor_value_info("y", TensorProto.INT64, [])
    reduce_node = helper.make_node(
        "ReduceMax", ["x"], ["y"], name="/core/ReduceMax", keepdims=0
    )
    graph = helper.make_graph([reduce_node], "tiny", [inp], [out])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 19)]
    )
    model.ir_version = 10
    path = tmp_path / "tiny.onnx"
    onnx.save(model, str(path))
    return path


def test_htp_pass_wraps_int64_reducemax(tmp_path):
    """Every int64 ReduceMax HTP rejected must be wrapped in fp32 casts."""
    onnx = pytest.importorskip("onnx")
    src = _tiny_int64_reducemax(tmp_path)
    dst = tmp_path / "tiny_htp.onnx"

    stats = ex.make_htp_compatible(src, dst)

    assert stats["rewritten"] == ["/core/ReduceMax"]
    fixed = onnx.load(str(dst))
    onnx.checker.check_model(fixed)
    assert [n.op_type for n in fixed.graph.node] == ["Cast", "ReduceMax", "Cast"]
    # The wrapper restores int64, so consumers still see the old dtype.
    assert any(
        v.name == "y" and v.type.tensor_type.elem_type == 7
        for v in fixed.graph.output
    )


def test_htp_pass_preserves_outputs_bitwise(tmp_path):
    """Cast(fp32)->ReduceMax->Cast(int64) must not change a single value."""
    ort = pytest.importorskip("onnxruntime")
    numpy = pytest.importorskip("numpy")

    src = _tiny_int64_reducemax(tmp_path)
    dst = tmp_path / "tiny_htp.onnx"
    ex.make_htp_compatible(src, dst)

    data = numpy.array([3, 40, 7], dtype=numpy.int64)
    before = ort.InferenceSession(str(src)).run(None, {"x": data})[0]
    after = ort.InferenceSession(str(dst)).run(None, {"x": data})[0]
    assert before == after == 40
    assert before.dtype == after.dtype == numpy.int64
