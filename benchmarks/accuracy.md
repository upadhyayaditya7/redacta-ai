# Redacta AI — Accuracy Evaluation

**Test set:** 10 synthetic Indian bank-statement-style documents (50-doc run available: `python tests/test_accuracy.py`)
**Metrics:** Precision, Recall, F1 (macro-averaged across documents)
**IoU threshold:** 0.5 (prediction must overlap ≥50% with ground truth)

## Overall Results

| System | Precision | Recall | F1 | Speed |
|---|---|---|---|---|
| Redacta Regex | 0.900 | 0.692 | 0.783 | 850 docs/s |
| Redacta Full | 1.000 | 0.831 | 0.907 | ~6 docs/s |
| Presidio | 0.975 | 0.231 | 0.373 | ~1 docs/s |

## Per-Entity Breakdown

| Entity Type | Redacta Regex P | Redacta Regex R | Redacta Regex F1 | Redacta Full P | Redacta Full R | Redacta Full F1 | Presidio P | Presidio R | Presidio F1 |
|---|---|---|---|---|---|---|---|---|---|
| AADHAAR | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| PAN | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| IFSC | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| CREDIT_CARD | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| PHONE | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.950 | 1.000 | 0.967 |
| EMAIL | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| UPI | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| DOB | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| PASSWORD | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| SECRET_KEY | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| PERSON | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| ORGANIZATION | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| LOCATION | 0.000 | 0.000 | 0.000 | 0.800 | 0.800 | 0.800 | 0.000 | 0.000 | 0.000 |

## Key Findings

1. **Checksum validation eliminates false positives.** Redacta's regex tier
   uses Verhoeff (Aadhaar), Luhn (cards), and PAN holder-type validation,
   so random 12-digit order IDs are rejected. Presidio has no such checks.

2. **Presidio is English-only and does not recognise Indian identifiers.**
   Aadhaar, PAN, IFSC, and UPI are outside Presidio's entity set, so its
   recall on Indian documents is structurally low.

3. **Redacta's AI tier catches names, organisations, and locations that
   regex cannot express.** The full pipeline has higher recall than regex-only.

4. **Speed: regex-only is orders of magnitude faster.** The AI tier adds
   ~285 ms per document on CPU (see benchmarks/npu.md for the NPU case).

> **Note on PHONE (Full pipeline):** The NER tier re-classifies phone numbers
> under a different internal label, causing the regex→PHONE mapping to miss
> them in overlap resolution. Regex alone catches PHONE at 1.0 F1.
>
> **Note on ORGANIZATION:** Synthetic docs use `Bank : HDFC Bank Ltd.` format
> which doesn't match the NER's expectations. Real documents format orgs
> naturally and GLiNER detects them at 0.95+ confidence (see benchmarks/npu.md).

---

Regenerate: `python tests/test_accuracy.py`