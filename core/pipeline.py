"""Detection pipeline: regex (always) + NER (optional), merged and explained.

The regex detector is the guaranteed floor — deterministic, checksum-backed.
The NER detector adds names/locations/organisations when GLiNER is
installed; otherwise the pipeline degrades gracefully and says so.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .entities import EntityType
from .ner_detector import NerDetector
from .regex_rules import RegexDetector, Span, _resolve_overlaps


@dataclass
class PipelineReport:
    """Everything the UI needs: spans, timings, which detectors ran."""

    spans: list[Span] = field(default_factory=list)
    regex_ms: float = 0.0
    ner_ms: float = 0.0
    ner_available: bool = False
    ner_backend: str | None = None  # "onnx" (on-device export) | "torch"
    ner_error: str | None = None
    text_length: int = 0

    @property
    def total_ms(self) -> float:
        return self.regex_ms + self.ner_ms

    def summary(self) -> dict[EntityType, int]:
        out: dict[EntityType, int] = {}
        for s in self.spans:
            out[s.entity_type] = out.get(s.entity_type, 0) + 1
        return out


class RedactionPipeline:
    """Regex + optional NER, merged with overlap resolution."""

    def __init__(self, use_ner: bool = True) -> None:
        self.regex = RegexDetector()
        self.ner = NerDetector() if use_ner else None

    def analyze(self, text: str, types: list[EntityType] | None = None) -> PipelineReport:
        report = PipelineReport(text_length=len(text))

        t0 = time.perf_counter()
        regex_result = self.regex.detect(text, types)
        report.regex_ms = (time.perf_counter() - t0) * 1000
        spans = list(regex_result.spans)

        if self.ner is not None:
            t0 = time.perf_counter()
            ner_result = self.ner.detect(text, types)
            report.ner_ms = (time.perf_counter() - t0) * 1000
            # "available" now means the model actually loaded and ran —
            # not merely that no error has been recorded yet.
            report.ner_available = self.ner.loaded
            report.ner_backend = self.ner.backend
            report.ner_error = self.ner.error
            spans.extend(ner_result.spans)

        report.spans = _resolve_overlaps(spans)
        return report
