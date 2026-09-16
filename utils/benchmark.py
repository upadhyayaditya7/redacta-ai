"""Benchmark: measure detection latency and throughput on sample docs.

Outputs a markdown table under ``benchmarks/`` suitable for pasting into
the proposal.  On a Snapdragon X machine with the QNN backend installed,
the same harness is used to compare CPU vs NPU numbers — that comparison
table is the centerpiece of the "Technical Implementation" score.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from core.config import BENCH_DIR, SAMPLES_DIR
from core.pipeline import RedactionPipeline
from utils.samples import document

WARMUP, RUNS = 3, 30


def bench(pipeline: RedactionPipeline, text: str) -> dict:
    for _ in range(WARMUP):
        pipeline.analyze(text)
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        pipeline.analyze(text)
        times.append(time.perf_counter() - t0)
    return {
        "mean_ms": statistics.mean(times) * 1000,
        "p95_ms": statistics.quantiles(times, n=20)[18] * 1000,
        "throughput_docs_per_s": 1.0 / statistics.mean(times),
    }


def run() -> dict:
    rng_text = document()
    long_text = "\n".join(document() for _ in range(20))  # ~20 "pages"
    results = {
        "meta": {
            "runs": RUNS,
            "warmup": WARMUP,
            "short_doc_chars": len(rng_text),
            "long_doc_chars": len(long_text),
        },
        "regex_only_short": bench(RedactionPipeline(use_ner=False), rng_text),
        "regex_only_long": bench(RedactionPipeline(use_ner=False), long_text),
    }
    return results


def write_report(results: dict, out: Path = BENCH_DIR / "baseline.md") -> Path:
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Redacta AI — baseline benchmark",
        "",
        f"- Runs: {results['meta']['runs']} (+{results['meta']['warmup']} warmup)",
        f"- Short doc: {results['meta']['short_doc_chars']} chars | "
        f"Long doc: {results['meta']['long_doc_chars']} chars",
        "",
        "| Scenario | Mean (ms) | p95 (ms) | Docs/s |",
        "|---|---|---|---|",
    ]
    for name in ("regex_only_short", "regex_only_long"):
        r = results[name]
        lines.append(
            f"| {name} | {r['mean_ms']:.2f} | {r['p95_ms']:.2f} | {r['throughput_docs_per_s']:.0f} |"
        )
    lines += [
        "",
        "_Regex-only baseline. NER/NPU comparison table added in milestone M2._",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


if __name__ == "__main__":
    res = run()
    path = write_report(res)
    print(json.dumps(res, indent=2))
    print(f"\nWrote {path}")
