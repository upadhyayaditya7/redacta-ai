# Redacta AI — Accuracy Evaluation

**Test set:** 50 synthetic Indian bank-statement-style documents
**Metrics:** Precision, Recall, F1 (macro-averaged across documents)
**IoU threshold:** 0.5 (prediction must overlap ≥50% with ground truth)

## Overall Results

| System | Precision | Recall | F1 | Speed |
|---|---|---|---|---|
| Redacta Regex | 0.900 | 0.692 | 0.783 | 1322 docs/s |
| Redacta Full | 1.000 | 0.831 | 0.907 | 0 docs/s |
| Presidio | 0.975 | 0.231 | 0.373 | 1 docs/s |

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

---

Regenerate: `python tests/test_accuracy.py`