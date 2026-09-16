"""NER detector: wraps GLiNER (or any transformers NER model) for PII.

This detector is *optional*.  When the model or libraries are unavailable
(cold machine, CPU-only demo, pip failure on Python 3.14), the pipeline
still works — regex detection is the guaranteed floor and NER adds
names, locations and organisations on top.

Model choice: ``urchade/gliner_multi_pii-v1`` — an open-source,
zero-shot NER trained specifically for PII across 8 languages
(including Indian naming patterns).  Exportable via Qualcomm AI Hub.
"""

from __future__ import annotations

from .entities import EntityType
from .regex_rules import DetectionResult, Span, _resolve_overlaps

GLINER_MODEL = "urchade/gliner_multi_pii-v1"

# GLiNER PII labels -> Redacta entity types.
_LABEL_MAP: dict[str, EntityType] = {
    "person": EntityType.PERSON,
    "per": EntityType.PERSON,
    "location": EntityType.LOCATION,
    "loc": EntityType.LOCATION,
    "organization": EntityType.ORG,
    "org": EntityType.ORG,
    "email": EntityType.EMAIL,
    "phone_number": EntityType.PHONE,
    "phone": EntityType.PHONE,
    "date_of_birth": EntityType.DATE_OF_BIRTH,
    "dob": EntityType.DATE_OF_BIRTH,
    "credit_card_number": EntityType.CREDIT_CARD,
    "credit card number": EntityType.CREDIT_CARD,
}

# Labels regex already covers better (avoid duplicate/conflicting hits).
_REGEX_COVERED = {
    EntityType.AADHAAR, EntityType.PAN, EntityType.IFSC, EntityType.UPI_ID,
    EntityType.EMAIL, EntityType.CREDIT_CARD, EntityType.INDIAN_MOBILE,
    EntityType.PASSWORD, EntityType.SECRET_KEY, EntityType.OTP,
}


class NerDetector:
    """Optional zero-shot NER detector. Gracefully reports availability."""

    name = "ner-gliner"

    def __init__(self, model_name: str = GLINER_MODEL, threshold: float = 0.5) -> None:
        self.model_name = model_name
        self.threshold = threshold
        self._model = None
        self.loaded = False  # True only after a successful model load
        self.error: str | None = None

    @property
    def available(self) -> bool:
        try:
            self._load()
        except Exception as exc:  # noqa: BLE001 - report any load failure
            self.error = f"{type(exc).__name__}: {exc}"
            return False
        return True

    def _load(self):  # noqa: ANN202
        if self._model is not None:
            return self._model
        try:
            from gliner import GLiNER  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError("gliner package not installed") from exc
        self._model = GLiNER.from_pretrained(self.model_name)
        self.loaded = True
        return self._model

    def detect(self, text: str, types: list[EntityType] | None = None) -> DetectionResult:
        # Default: NER handles every label it knows, minus what regex
        # already covers better.  Explicit type filters are respected.
        wanted = set(types) if types else set(_LABEL_MAP.values())
        wanted -= _REGEX_COVERED
        labels = [
            label
            for label, et in _LABEL_MAP.items()
            if et in wanted
        ]
        if not labels:
            return DetectionResult([])
        try:
            model = self._load()
        except Exception as exc:  # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"
            return DetectionResult([])
        raw = model.predict_entities(text, labels, threshold=self.threshold)
        spans = [
            Span(
                entity_type=_LABEL_MAP.get(
                    ent["label"].lower().replace(" ", "_").replace("-", "_"),
                    EntityType.PERSON,
                ),
                start=ent["start"],
                end=ent["end"],
                text=ent["text"],
                score=float(ent.get("score", 0.5)),
                detector=f"{self.name}:{ent['label']}",
                reason=f"NER confidence {ent.get('score', 0.5):.2f}",
            )
            for ent in raw
        ]
        return DetectionResult(_resolve_overlaps(spans))
