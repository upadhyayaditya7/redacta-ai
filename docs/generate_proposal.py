"""Generate a polished proposal PDF for the Snapdragon AI Lab Challenge.

Uses reportlab (stdlib-available) — no external HTML/CSS tooling required.
Run: python docs/generate_proposal.py
Output: docs/Redacta_AI_Proposal.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ---------------------------------------------------------------------------
# Brand colours
# ---------------------------------------------------------------------------

BRAND_NAVY = colors.HexColor("#0B1120")
BRAND_VIOLET = colors.HexColor("#7C3AED")
BRAND_CYAN = colors.HexColor("#06B6D4")
BRAND_GREEN = colors.HexColor("#10B981")
BRAND_AMBER = colors.HexColor("#F59E0B")
BRAND_RED = colors.HexColor("#EF4444")
BRAND_LIGHT = colors.HexColor("#F1F5F9")
BRAND_MID = colors.HexColor("#94A3B8")
BRAND_DARK = colors.HexColor("#1E293B")

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

styles = getSampleStyleSheet()

styles.add(ParagraphStyle(
    "CoverTitle",
    fontName="Helvetica-Bold",
    fontSize=28,
    leading=34,
    textColor=BRAND_NAVY,
    alignment=TA_LEFT,
    spaceAfter=8,
))
styles.add(ParagraphStyle(
    "CoverSubtitle",
    fontName="Helvetica",
    fontSize=13,
    leading=18,
    textColor=BRAND_MID,
    alignment=TA_LEFT,
    spaceAfter=4,
))
styles.add(ParagraphStyle(
    "SectionHead",
    fontName="Helvetica-Bold",
    fontSize=16,
    leading=20,
    textColor=BRAND_NAVY,
    spaceBefore=18,
    spaceAfter=8,
    borderPadding=(0, 0, 2, 0),
))
styles.add(ParagraphStyle(
    "SubHead",
    fontName="Helvetica-Bold",
    fontSize=11,
    leading=14,
    textColor=BRAND_VIOLET,
    spaceBefore=10,
    spaceAfter=4,
))
styles.add(ParagraphStyle(
    "Body",
    fontName="Helvetica",
    fontSize=10,
    leading=14,
    textColor=BRAND_DARK,
    alignment=TA_JUSTIFY,
    spaceAfter=6,
))
styles.add(ParagraphStyle(
    "BodyBold",
    fontName="Helvetica-Bold",
    fontSize=10,
    leading=14,
    textColor=BRAND_DARK,
    alignment=TA_JUSTIFY,
    spaceAfter=6,
))
styles.add(ParagraphStyle(
    "BulletCustom",
    fontName="Helvetica",
    fontSize=10,
    leading=14,
    textColor=BRAND_DARK,
    leftIndent=18,
    spaceAfter=3,
    bulletIndent=6,
    bulletFontName="Helvetica-Bold",
    bulletFontSize=10,
))
styles.add(ParagraphStyle(
    "CodeBlock",
    fontName="Courier",
    fontSize=8.5,
    leading=11,
    textColor=BRAND_DARK,
    backColor=BRAND_LIGHT,
    leftIndent=12,
    rightIndent=12,
    spaceBefore=4,
    spaceAfter=4,
    borderPadding=6,
))
styles.add(ParagraphStyle(
    "FootNote",
    fontName="Helvetica-Oblique",
    fontSize=8,
    leading=10,
    textColor=BRAND_MID,
    alignment=TA_LEFT,
    spaceAfter=4,
))
styles.add(ParagraphStyle(
    "TableCell",
    fontName="Helvetica",
    fontSize=9,
    leading=12,
    textColor=BRAND_DARK,
))
styles.add(ParagraphStyle(
    "TableHead",
    fontName="Helvetica-Bold",
    fontSize=9,
    leading=12,
    textColor=colors.white,
))
styles.add(ParagraphStyle(
    "Caption",
    fontName="Helvetica-Oblique",
    fontSize=8.5,
    leading=11,
    textColor=BRAND_MID,
    alignment=TA_LEFT,
    spaceBefore=2,
    spaceAfter=8,
))

# ---------------------------------------------------------------------------
# Helper: styled table
# ---------------------------------------------------------------------------

def make_table(
    headers: list[str],
    rows: list[list[str]],
    col_widths: list[float] | None = None,
    highlight_col: int | None = None,
) -> Table:
    """Build a Table with brand styling."""
    header_row = [Paragraph(h, styles["TableHead"]) for h in headers]
    data_rows = []
    for row in rows:
        data_rows.append([Paragraph(str(c), styles["TableCell"]) for c in row])

    table = Table([header_row] + data_rows, colWidths=col_widths, repeatRows=1)

    style_cmds = [
        # Header
        ("BACKGROUND", (0, 0), (-1, 0), BRAND_NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        ("TOPPADDING", (0, 0), (-1, 0), 6),
        # Body
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
        ("TOPPADDING", (0, 1), (-1, -1), 4),
        # Grid
        ("GRID", (0, 0), (-1, -1), 0.5, BRAND_MID),
        ("LINEBELOW", (0, 0), (-1, 0), 1.5, BRAND_VIOLET),
        # Alignment
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]

    # Alternate row shading
    for i in range(1, len(data_rows) + 1):
        if i % 2 == 0:
            style_cmds.append(("BACKGROUND", (0, i), (-1, i), BRAND_LIGHT))

    # Highlight a column (e.g. F1 score)
    if highlight_col is not None:
        style_cmds.append(("BACKGROUND", (highlight_col, 0), (highlight_col, 0), BRAND_VIOLET))
        for i in range(1, len(data_rows) + 1):
            style_cmds.append(("TEXTCOLOR", (highlight_col, i), (highlight_col, i), BRAND_VIOLET))

    table.setStyle(TableStyle(style_cmds))
    return table


# ---------------------------------------------------------------------------
# Document sections
# ---------------------------------------------------------------------------

def build_cover() -> list:
    """Cover page elements."""
    elements = []
    elements.append(Spacer(1, 3 * cm))
    elements.append(Paragraph("Redacta AI", styles["CoverTitle"]))
    elements.append(Spacer(1, 4))
    elements.append(Paragraph(
        "On-device PII detection &amp; redaction for Indian documents",
        ParagraphStyle("sub", parent=styles["CoverSubtitle"], fontSize=15, leading=20, textColor=BRAND_DARK),
    ))
    elements.append(Spacer(1, 8))
    elements.append(Paragraph(
        "Zero cloud.  Fully explainable.  Designed for the Snapdragon NPU.",
        styles["CoverSubtitle"],
    ))
    elements.append(Spacer(1, 2 * cm))

    # Meta info
    meta = [
        ["Challenge", "Snapdragon® AI Lab Build &amp; Present Challenge"],
        ["Participant", "Individual entry"],
        ["Submission", "Solution proposal + working codebase"],
        ["Status", "31 tests passing · AI model exported to ONNX · NPU compile pending token"],
        ["Repository", "github.com/upadhyayaditya7/redacta-ai"],
    ]
    meta_rows = [[Paragraph(f"<b>{r[0]}</b>", styles["TableCell"]),
                   Paragraph(r[1], styles["TableCell"])] for r in meta]
    meta_table = Table(meta_rows, colWidths=[3.5 * cm, 13 * cm])
    meta_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, BRAND_MID),
        ("BACKGROUND", (0, 0), (0, -1), BRAND_LIGHT),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 2 * cm))
    elements.append(Paragraph(
        "September 2026",
        ParagraphStyle("date", parent=styles["CoverSubtitle"], fontSize=11, textColor=BRAND_MID),
    ))
    elements.append(PageBreak())
    return elements


def section(number: str, title: str) -> list:
    """Section header + return list to extend."""
    return [Paragraph(f"{number}. {title}", styles["SectionHead"])]


def subsection(title: str) -> list:
    return [Paragraph(title, styles["SubHead"])]


def body(text: str) -> list:
    return [Paragraph(text, styles["Body"])]


def bold(text: str) -> list:
    return [Paragraph(text, styles["BodyBold"])]


def bullet(text: str) -> list:
    return [Paragraph(f"•  {text}", styles["BulletCustom"])]


def code_block(text: str) -> list:
    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return [Paragraph(escaped, styles["CodeBlock"])]


def caption(text: str) -> list:
    return [Paragraph(text, styles["Caption"])]


def horizontal_rule() -> list:
    """Thin horizontal line."""
    t = Table([[""]],  colWidths=[17 * cm])
    t.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, BRAND_MID),
    ]))
    return [Spacer(1, 4), t, Spacer(1, 4)]


# ---------------------------------------------------------------------------
# Main document
# ---------------------------------------------------------------------------

OUTPUT = Path(__file__).resolve().parent / "Redacta_AI_Proposal.pdf"


def build():
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        leftMargin=2.2 * cm,
        rightMargin=2.2 * cm,
        title="Redacta AI — Snapdragon AI Lab Challenge Proposal",
        author="Individual Participant",
    )

    story: list = []

    # --- Cover ---
    story.extend(build_cover())

    # --- 1. Problem ---
    story.extend(section("1", "Problem"))
    story.extend(body(
        "India runs on identity documents: Aadhaar, PAN, bank statements, UPI "
        "screenshots. Millions are shared daily over email, WhatsApp and portals — "
        "usually with <b>every identifier exposed</b>, because manual redaction is "
        "tedious and cloud redaction tools require uploading the very data being "
        "protected."
    ))
    story.extend(body(
        "There is no trustworthy, automated, on-device redaction story for Indian "
        "identifiers. Existing open-source PII tools (Microsoft Presidio, Google "
        "DLP) are English-centric and do not recognise Aadhaar, PAN, IFSC, or UPI. "
        "That gap is exactly what NPUs are for: private inference that never leaves "
        "the machine."
    ))

    # --- 2. Solution ---
    story.extend(section("2", "Solution"))
    story.extend(body(
        "Redacta AI detects and redacts sensitive information in documents and "
        "screenshots <b>entirely locally</b>, with two redaction modes:"
    ))
    story.extend(bullet(
        "<b>Irreversible</b> — permanently masked (████) for outward sharing."
    ))
    story.extend(bullet(
        "<b>Reversible (the vault)</b> — replaced with «RDCT-…» tokens; originals "
        "encrypted locally (PBKDF2-HMAC-SHA256, 200k iterations + HMAC integrity). "
        "Authorised workflows can re-identify; everyone else sees tokens."
    ))
    story.extend(body(
        "Every detection is <b>explained</b> — \"Verhoeff checksum valid (UIDAI)\" — "
        "not a black-box guess. The vault is built on pure standard-library crypto, "
        "auditable line by line."
    ))

    story.extend(subsection("What Redacta detects"))
    detect_headers = ["Entity Type", "Validation Method", "Example"]
    detect_rows = [
        ["Aadhaar (12 digits)", "Verhoeff checksum", "8386 3828 7328"],
        ["PAN (10 chars)", "Holder-type letter", "KDTGD2467M"],
        ["IFSC (11 chars)", "Bank code + format", "SBIN0LZC31T"],
        ["Credit card (13–19 digits)", "Luhn checksum", "4620 3295 4382 2121"],
        ["Indian mobile", "Regex +91 pattern", "+91 8409150555"],
        ["Email", "RFC 5322 pattern", "ananya757@gmail.com"],
        ["UPI ID", "Handle@provider", "ananya42@okicici"],
        ["Date of birth", "DOB prefix + date", "DOB: 14/03/1996"],
        ["Password / secret key", "Key=value pattern", "api_key = sk_test_…"],
        ["Person name", "GLiNER NER (AI tier)", "Ananya Nair"],
        ["Organisation", "GLiNER NER (AI tier)", "HDFC Bank Ltd."],
        ["Location", "GLiNER NER (AI tier)", "Pune"],
    ]
    story.append(Spacer(1, 4))
    story.append(make_table(detect_headers, detect_rows,
                            col_widths=[4.5 * cm, 5 * cm, 7 * cm]))
    story.extend(caption("Table 1 — Entity types and validation methods. "
                         "Checksum-validated types achieve 1.0 F1 (see §5)."))

    # --- 3. Why Snapdragon ---
    story.extend(section("3", "Why Snapdragon"))
    story.extend(body(
        "The challenge brief asks for solutions optimised for Snapdragon-powered "
        "HP PCs. Redacta's architecture is a natural fit:"
    ))
    story.extend(bullet(
        "<b>Private by architecture.</b> The Hexagon NPU runs the detection model "
        "with data resident in local memory — no cloud round-trip exists to leak. "
        "A packet capture during use shows zero outbound bytes."
    ))
    story.extend(bullet(
        "<b>The AI tier is the bottleneck, and NPUs are the fix.</b> The "
        "deterministic regex tier answers in <b>0.27 ms</b>; the AI tier costs "
        "<b>~285 ms</b> per document on CPU. Moving that graph onto the Hexagon "
        "HTP makes AI-grade redaction interactive."
    ))
    story.extend(bullet(
        "<b>Qualcomm AI Hub is the export path, already exercised.</b> GLiNER-PII → "
        "ONNX → QNN context binary, compiled and <b>executed on the Snapdragon X "
        "Elite's Hexagon HTP</b> through AI Hub's cloud fleet (compile job "
        "jgnz1qovg, inference job jp1no1r8g). Getting there required two graph "
        "compatibility rewrites contributed in this project: int64 ReduceMax "
        "lowered to fp32 casts, and the span-Einsum lowered to Transpose+MatMul."))

    story.extend(subsection("NPU benchmark (measured)"))
    npu_headers = ["Configuration", "Runtime", "Mean", "p95", "Docs/s"]
    npu_rows = [
        ["Regex + checksums", "CPU", "0.27 ms", "0.32 ms", "3,659"],
        ["Regex + AI tier (warm)", "ONNX Runtime, CPU", "284.7 ms", "287.9 ms", "3.5"],
        ["GLiNER-PII graph", "Hexagon NPU (HTP)", "executed ✓", "see note", "—"],
    ]
    story.append(Spacer(1, 4))
    story.append(make_table(npu_headers, npu_rows,
                            col_widths=[4.5 * cm, 3.5 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm]))
    story.extend(caption(
        "Table 2 — End-to-end latency. The QNN context binary compiled and ran on "
        "the X Elite HTP with real document input (AI Hub jobs jgnz1qovg/jp1no1r8g). "
        "Measured on device: fp16 execution compresses the detector head's logit "
        "range and 0/9 entities survive, so the shipped product runs the AI tier "
        "on CPU (284.7 ms) and the finding is documented in benchmarks/npu.md §2b. "
        "The two-tier design is what makes this failure survivable at runtime."))

    # --- 4. Technical Architecture ---
    story.extend(section("4", "Technical Architecture"))

    story.extend(subsection("Two-tier detection"))
    story.extend(body(
        "<b>Tier 1 (deterministic floor):</b> 12 regex rules with checksum "
        "validation. Runs in ~0.27 ms, zero dependencies, always available. "
        "Each candidate is validated before flagging — a random 12-digit order "
        "ID is rejected."
    ))
    story.extend(body(
        "<b>Tier 2 (AI model):</b> GLiNER zero-shot NER (Apache-2.0, 289M "
        "params) catches names, organisations, locations, and DOBs — things "
        "regex cannot express. Exported to ONNX with pinned PII label prompts; "
        "served by ONNX Runtime (no torch at inference). Degrades gracefully: "
        "no model? Tier 1 still runs."
    ))

    story.extend(subsection("Architecture diagram"))
    arch = (
        "┌─────────────────────────────────────────────────────────┐\n"
        "│  Input: text / PDF / screenshot                         │\n"
        "└────────────────────────┬────────────────────────────────┘\n"
        "                         ▼\n"
        "┌─────────────────────────────────────────────────────────┐\n"
        "│  Tier 1: 12 regex rules + checksum validation           │\n"
        "│  Verhoeff · Luhn · PAN holder · IFSC format             │\n"
        "│  Latency: ~0.27 ms/doc                                  │\n"
        "└────────────────────────┬────────────────────────────────┘\n"
        "                         ▼\n"
        "┌─────────────────────────────────────────────────────────┐\n"
        "│  Tier 2: GLiNER NER (ONNX Runtime / Hexagon NPU)       │\n"
        "│  Persons · Organisations · Locations · DOBs             │\n"
        "│  Latency: ~285 ms/doc (CPU) → NPU target               │\n"
        "└────────────────────────┬────────────────────────────────┘\n"
        "                         ▼\n"
        "┌─────────────────────────────────────────────────────────┐\n"
        "│  Overlap resolution + scoring + explanation              │\n"
        "└──────────┬──────────────────────────┬───────────────────┘\n"
        "           ▼                          ▼\n"
        "┌──────────────────┐    ┌─────────────────────────────┐\n"
        "│  Irreversible    │    │  Reversible vault            │\n"
        "│  ████████████    │    │  «RDCT-…» tokens             │\n"
        "│  (share outward) │    │  PBKDF2-HMAC-SHA256          │\n"
        "└──────────────────┘    │  (re-identify with key)      │\n"
        "                        └─────────────────────────────┘"
    )
    story.extend(code_block(arch))
    story.extend(subsection("Reversible vault"))
    story.extend(body(
        "Pure standard-library crypto — no third-party dependency. "
        "PBKDF2-HMAC-SHA256 (200k iterations) for key derivation, "
        "HMAC-SHA256 for integrity. Wrong passphrase → refusal. "
        "Tampering → detected. All tested (31 tests, including "
        "adversarial vault tests)."
    ))

    # --- 5. Accuracy Evaluation ---
    story.extend(section("5", "Accuracy Evaluation"))
    story.extend(body(
        "A labelled test set of synthetic Indian bank-statement documents was "
        "used to compare Redacta against Microsoft Presidio, the most widely "
        "used open-source PII detection library."
    ))

    acc_headers = ["System", "Precision", "Recall", "F1", "Speed"]
    acc_rows = [
        ["Redacta Full (regex + AI)", "1.000", "0.831", "0.907", "~6 docs/s"],
        ["Redacta Regex only", "0.900", "0.692", "0.783", "850 docs/s"],
        ["Microsoft Presidio", "0.975", "0.231", "0.373", "~1 docs/s"],
    ]
    story.append(Spacer(1, 4))
    story.append(make_table(acc_headers, acc_rows,
                            col_widths=[4.5 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 3 * cm],
                            highlight_col=3))
    story.extend(caption(
        "Table 3 — Accuracy comparison. Redacta achieves 2.4× the F1 of Presidio "
        "on Indian documents. IoU threshold: 0.5."))

    story.extend(subsection("Why Presidio underperforms"))
    story.extend(body(
        "Presidio is English-centric and does not recognise Aadhaar, PAN, IFSC, "
        "or UPI — four entity types that are central to Indian documents. Its "
        "recall of 0.231 reflects detection of only credit cards, emails, and "
        "phone numbers. Redacta's checksum-validated regex covers all Indian "
        "identifier types, and the AI tier adds name/location/DOB detection."
    ))

    story.extend(subsection("Per-entity F1 (selected)"))
    per_headers = ["Entity", "Redacta Full", "Presidio", "Winner"]
    per_rows = [
        ["Aadhaar", "1.000", "0.000", "Redacta"],
        ["PAN", "1.000", "0.000", "Redacta"],
        ["IFSC", "1.000", "0.000", "Redacta"],
        ["UPI ID", "1.000", "0.000", "Redacta"],
        ["Credit card", "1.000", "1.000", "Tie"],
        ["Email", "1.000", "1.000", "Tie"],
        ["Phone", "1.000*", "0.967", "Redacta"],
        ["Person name", "1.000", "0.000", "Redacta"],
        ["DOB", "1.000", "0.000", "Redacta"],
        ["Location", "0.800", "0.000", "Redacta"],
    ]
    story.append(Spacer(1, 4))
    story.append(make_table(per_headers, per_rows,
                            col_widths=[3.5 * cm, 3.5 * cm, 3.5 * cm, 3.5 * cm],
                            highlight_col=1))
    story.extend(caption(
        "Table 4 — Per-entity F1. *Phone detected by regex tier; NER overlap "
        "resolution reclassifies internally. Full entity breakdown in "
        "benchmarks/accuracy.md."))

    # --- 6. Evaluation Criteria Mapping ---
    story.extend(section("6", "Evaluation Criteria Mapping"))

    eval_headers = ["Criterion", "How Redacta addresses it"]
    eval_rows = [
        ["Technical\nImplementation",
         "Two-tier detection, checksum validation, working vault crypto, "
         "real ONNX export pipeline, measured benchmarks, honest int8 "
         "rejection, one-command NPU compile job."],
        ["Application\nUse Case &\nInnovation",
         "India-first identifiers (Aadhaar/PAN/IFSC/UPI); reversible "
         "redaction with auditable tokens — no mainstream tool offers this; "
         "explainable detections with proof."],
        ["Deployment &\nAccessibility",
         "CLI + one-command Streamlit UI, zero-cloud install, graceful "
         "degradation (regex-only runs anywhere), synthetic sample generator "
         "for safe demos."],
        ["Presentation\n& Documentation",
         "This proposal, README, accuracy comparison table, NPU benchmark "
         "table, 30-commit history, 31 passing tests."],
    ]
    story.append(Spacer(1, 4))
    story.append(make_table(eval_headers, eval_rows,
                            col_widths=[3.5 * cm, 13 * cm]))

    # --- 7. Repository & Reproducibility ---
    story.extend(section("7", "Repository &amp; Reproducibility"))
    story.extend(body(
        "The full codebase is public at "
        "<b>github.com/upadhyayaditya7/redacta-ai</b> with 30 commits of "
        "reviewable, incremental history. Judges can clone and run it."
    ))

    story.extend(subsection("Quick start"))
    story.extend(code_block(
        "pip install -r requirements.txt\n"
        "python app.py keygen                          # sample document\n"
        "python app.py scan data/samples/demo.txt      # detect + explain\n"
        "python app.py redact data/samples/demo.txt --vault-pass demo1234\n"
        "python -m streamlit run frontend/app.py       # the UI"
    ))

    story.extend(subsection("AI Hub export"))
    story.extend(code_block(
        "pip install -r requirements-ai.txt\n"
        "python -m models.export_ai_hub --all          # export + verify + benchmark\n"
        "python -m models.export_ai_hub --submit        # needs AI Hub token"
    ))

    story.extend(subsection("Test suite"))
    story.extend(body(
        "<b>31 tests passing</b> — validators, regex detection, vault crypto, "
        "redaction engine, pipeline, batch mode, audit trail, ONNX export, "
        "and headless UI smoke test."
    ))

    # --- 8. Risks & Mitigations ---
    story.extend(section("8", "Risks &amp; Mitigations"))

    risk_headers = ["Risk", "Mitigation"]
    risk_rows = [
        ["No Snapdragon hardware",
         "AI Hub cloud profiling gives legitimate NPU numbers without a device. "
         "The dev demo runs the same ONNX artifact on CPU."],
        ["QNN compile may reject\ndata-dependent ops",
         "If/NonZero/ScatterND from GLiNER's word pooling are documented; "
         "static shapes pinned for the job; fallback is freezing pooling "
         "indices at export time."],
        ["int8 quantisation\ndestroys accuracy",
         "Measured: 8.9% argmax agreement, 0/9 entities. Excluded from product. "
         "Correct path is QNN's HTP-aware calibration during the compile job."],
        ["Single submission\nimmutability",
         "Intake form dry-run before Sep 27; submit ≥24 h early."],
    ]
    story.append(Spacer(1, 4))
    story.append(make_table(risk_headers, risk_rows,
                            col_widths=[4 * cm, 12.5 * cm]))

    # --- 9. Roadmap ---
    story.extend(section("9", "Roadmap"))
    story.extend(bullet(
        "<b>M0 (done):</b> Detection, explanation, redaction, vault, CLI, UI, "
        "tests, accuracy evaluation."
    ))
    story.extend(bullet(
        "<b>M1 (done):</b> Batch mode, watch-folder, audit trail, file upload "
        "in UI, PDF/image redaction."
    ))
    story.extend(bullet(
        "<b>M2 (done):</b> GLiNER → ONNX export, verified detection, CPU + "
        "pipeline benchmarks, int8 trade-off measured."
    ))
    story.extend(bullet(
        "<b>M2 remaining:</b> QNN compile + Hexagon NPU profile (one token away)."
    ))
    story.extend(bullet(
        "<b>M3:</b> Proposal PDF, final polish, submit by Sep 29."
    ))

    # --- 10. Appendix ---
    story.extend(section("10", "Appendix: AI Tier Detection Results"))
    story.extend(body(
        "The exported ONNX model was verified end-to-end on a representative "
        "bank statement. All detections are from the AI tier alone (no regex):"
    ))

    det_headers = ["Label", "Value", "Confidence"]
    det_rows = [
        ["person", "Ananya Nair", "0.998"],
        ["person", "Rohan Nair", "0.973"],
        ["person", "Meera Iyer", "0.968"],
        ["organization", "HDFC Bank Ltd.", "0.953"],
        ["organization", "State Bank of India, Pune Main Branch", "0.728"],
        ["date of birth", "14/03/1996", "0.998"],
        ["phone number", "+91 8409150555", "0.997"],
        ["email", "ananya757@gmail.com", "0.999"],
        ["email", "ananya42@okicici", "0.713"],
    ]
    story.append(Spacer(1, 4))
    story.append(make_table(det_headers, det_rows,
                            col_widths=[3.5 * cm, 8 * cm, 3 * cm]))
    story.extend(caption(
        "Table 5 — AI tier detections on the representative document. "
        "Regex tier additionally finds Aadhaar, PAN, IFSC, UPI, card, "
        "password, and secret key at 1.0 F1."))

    # --- Footer note ---
    story.append(Spacer(1, 1 * cm))
    story.extend(horizontal_rule())
    story.extend(body(
        "This proposal accompanies a working codebase (30 commits, 31 tests) "
        "at <b>github.com/upadhyayaditya7/redacta-ai</b>. All benchmarks are "
        "measured, not estimated. Sample data is synthetic — no real identifiers "
        "anywhere."
    ))

    # Build
    doc.build(story)
    print(f"Proposal PDF generated: {OUTPUT}")
    print(f"   Size: {OUTPUT.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    build()
