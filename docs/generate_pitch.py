"""Generate the Snapdragon challenge short-pitch deck (PPTX + PDF via COM).

Run:  python docs/generate_pitch.py
Out:  docs/Redacta_AI_Pitch.pptx  (+ .pdf when PowerPoint is installed)
"""
from __future__ import annotations

import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

DOCS = Path(__file__).resolve().parent
OUT_PPTX = DOCS / "Redacta_AI_Pitch.pptx"

NAVY = RGBColor(0x0F, 0x17, 0x2A)
PANEL = RGBColor(0x1E, 0x29, 0x3B)
WHITE = RGBColor(0xF8, 0xFA, 0xFC)
GREY = RGBColor(0x94, 0xA3, 0xB8)
CYAN = RGBColor(0x22, 0xD3, 0xEE)
AMBER = RGBColor(0xFB, 0xBF, 0x24)
GREEN = RGBColor(0x34, 0xD3, 0x99)
RED = RGBColor(0xF8, 0x71, 0x71)

SW, SH = Inches(13.333), Inches(7.5)


def new_deck() -> Presentation:
    prs = Presentation()
    prs.slide_width = SW
    prs.slide_height = SH
    return prs


def base_slide(prs: Presentation) -> object:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = NAVY
    return slide


def textbox(slide, left, top, width, height):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    return tf


def set_runs(p, runs):
    """runs = list of (text, size, bold, color)."""
    p.text = ""
    for i, (text, size, bold, color) in enumerate(runs):
        r = p.add_run()
        r.text = text
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = color
        r.font.name = "Segoe UI"


def title_bar(slide, kicker: str, title: str):
    tf = textbox(slide, Inches(0.55), Inches(0.35), Inches(12.2), Inches(1.25))
    p = tf.paragraphs[0]
    set_runs(p, [(kicker.upper(), 13, True, CYAN)])
    p2 = tf.add_paragraph()
    set_runs(p2, [(title, 32, True, WHITE)])
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.6), Inches(1.62), Inches(1.6), Pt(3))
    line.fill.solid()
    line.fill.fore_color.rgb = CYAN
    line.line.fill.background()


def bullets(slide, items, left=Inches(0.7), top=Inches(2.0), width=Inches(12.0),
            height=Inches(4.9), size=17, gap=10):
    tf = textbox(slide, left, top, width, height)
    first = True
    for item in items:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.space_after = Pt(gap)
        head, rest = item if isinstance(item, tuple) else (None, item)
        runs = [("▪  ", size, True, CYAN)]
        if head:
            runs.append((head, size, True, WHITE))
        runs.append((rest, size, False, GREY))
        set_runs(p, runs)
    return tf


def panel(slide, left, top, width, height, fill=PANEL):
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    sh.line.color.rgb = RGBColor(0x33, 0x41, 0x55)
    sh.text_frame.word_wrap = True
    return sh


def stat_card(slide, left, top, width, height, big, small, color=CYAN):
    sh = panel(slide, left, top, width, height)
    tf = sh.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    set_runs(p, [(big, 30, True, color)])
    p2 = tf.add_paragraph()
    p2.alignment = PP_ALIGN.CENTER
    set_runs(p2, [(small, 12.5, False, GREY)])
    return sh


def make_table(slide, rows, cols, left, top, width, height):
    return slide.shapes.add_table(rows, cols, left, top, width, height).table


def style_cell(cell, text, bold=False, color=WHITE, size=14, fill=None):
    if fill is not None:
        cell.fill.solid()
        cell.fill.fore_color.rgb = fill
    tf = cell.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    set_runs(p, [(text, size, bold, color)])


def footer(slide, idx: int, total: int):
    tf = textbox(slide, Inches(11.9), Inches(7.05), Inches(1.2), Inches(0.35))
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.RIGHT
    set_runs(p, [(f"{idx:02d} / {total:02d}", 11, False, RGBColor(0x47, 0x55, 0x69))])


def notes(slide, text: str):
    slide.notes_slide.notes_text_frame.text = text


TOTAL = 10


def slide_title(prs):
    s = base_slide(prs)
    tf = textbox(s, Inches(0.9), Inches(2.2), Inches(11.5), Inches(1.1))
    p = tf.paragraphs[0]
    set_runs(p, [("REDACTA AI", 15, True, CYAN)])
    tf2 = textbox(s, Inches(0.9), Inches(2.7), Inches(11.6), Inches(2.2))
    p = tf2.paragraphs[0]
    set_runs(p, [("Your PII never leaves the machine.", 44, True, WHITE)])
    p2 = tf2.add_paragraph()
    set_runs(p2, [("On-device detection & redaction for Indian documents.", 24, False, GREY)])
    tf3 = textbox(s, Inches(0.9), Inches(5.9), Inches(11.5), Inches(1.2))
    p = tf3.paragraphs[0]
    set_runs(p, [
        ("Snapdragon® AI Lab Build & Present Challenge", 14, True, AMBER),
        ("   ·   Sudhanshu Upadhyay   ·   github.com/upadhyayaditya7/redacta-ai", 14, False, GREY),
    ])
    notes(s, "Open with the one-liner: everything — detection, redaction, the vault, the audit log — happens on the laptop. Zero bytes leave the machine.")


def slide_problem(prs):
    s = base_slide(prs)
    title_bar(s, "The problem", "The safest document is the one you never upload")
    bullets(s, [
        ("Cloud redaction is an oxymoron — ", "you send the most sensitive files you own to a third party to make them shareable."),
        ("Global tools miss India. ", "Microsoft Presidio scores F1 0.373 on Indian financial PII — it was never built for Aadhaar, IFSC or UPI."),
        ("Regex alone is not enough. ", "Names, organisations and dates of birth have no pattern — but a 12-digit order ID must not be flagged as an Aadhaar number either."),
        ("Redaction is usually a one-way door. ", "Teams need shareable copies and the original; permanent deletion breaks the workflow."),
    ])
    footer(s, 2, TOTAL)


def slide_solution(prs):
    s = base_slide(prs)
    title_bar(s, "The solution", "Drop a document. Get redactions you can undo.")
    bullets(s, [
        ("Detect with proof. ", "Every hit carries a score and a human-readable reason: Verhoeff checksum valid (UIDAI), Luhn valid, PAN holder-type letter valid."),
        ("Redact two ways. ", "Permanent scrub, or reversible «RDCT-…» tokens whose originals sit in a local encrypted vault (PBKDF2-HMAC-SHA256, 200k iterations + HMAC integrity)."),
        ("Text, PDF, screenshots. ", "One CLI and one Streamlit UI; batch mode watches a folder and redacts arrivals drop-box style."),
        ("Compliance-safe audit trail. ", "The JSON audit log records what was found and what was done — with hints, never raw values."),
    ])
    footer(s, 3, TOTAL)


def slide_architecture(prs):
    s = base_slide(prs)
    title_bar(s, "Architecture", "Two tiers: a deterministic floor, an AI ceiling")
    flow = [
        ("Input", "text · PDF · screenshot (OCR)"),
        ("Tier 1 — Regex + checksums", "12 rules · Verhoeff / Luhn / IFSC · 0.27 ms"),
        ("Tier 2 — GLiNER NER (ONNX)", "names · orgs · DOBs · 285 ms warm"),
        ("Redaction engine", "permanent scrub or encrypted vault"),
        ("Audit trail", "JSON log, hints only — no raw PII"),
    ]
    top = Inches(2.05)
    for i, (head, sub) in enumerate(flow):
        y = top + Inches(0.92) * i
        sh = panel(s, Inches(0.9), y, Inches(8.2), Inches(0.78))
        tf = sh.text_frame
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]
        set_runs(p, [(head + "   ", 15, True, WHITE), ("·  " + sub, 13, False, GREY)])
        if i < len(flow) - 1:
            ar = s.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, Inches(4.85), y + Inches(0.78), Inches(0.3), Inches(0.16))
            ar.fill.solid()
            ar.fill.fore_color.rgb = CYAN
            ar.line.fill.background()
    side = panel(s, Inches(9.5), top, Inches(3.3), Inches(4.45))
    tf = side.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    set_runs(p, [("Graceful by design", 15, True, CYAN)])
    for txt in (
        "No torch installed? Regex tier still runs.",
        "Model missing? Degrades and says so honestly.",
        "Vault lost the passphrase? That's the point — irreversible to everyone, including us.",
    ):
        p2 = tf.add_paragraph()
        p2.space_before = Pt(8)
        set_runs(p2, [(txt, 12.5, False, GREY)])
    footer(s, 4, TOTAL)


def slide_accuracy(prs):
    s = base_slide(prs)
    title_bar(s, "Accuracy (measured)", "2.4× Microsoft Presidio on Indian financial PII")
    tbl = make_table(s, 4, 4, Inches(0.9), Inches(2.1), Inches(11.5), Inches(1.9))
    heads = ["System", "Precision", "Recall", "F1"]
    rows = [
        ["Redacta Full (regex + GLiNER)", "1.000", "0.831", "0.907"],
        ["Redacta Regex only", "0.900", "0.692", "0.783"],
        ["Microsoft Presidio", "0.975", "0.231", "0.373"],
    ]
    for c, h in enumerate(heads):
        style_cell(tbl.cell(0, c), h, bold=True, color=CYAN, fill=PANEL)
    for r, row in enumerate(rows, start=1):
        for c, val in enumerate(row):
            style_cell(tbl.cell(r, c), val, bold=(c == 0 or r == 1),
                       color=WHITE if r == 1 else GREY)
    bullets(s, [
        ("Zero false positives on the eval set ", "(P = 1.000) — every flagged span explains itself, so a human can verify in seconds."),
        ("Fully reproducible: ", "python tests/test_accuracy.py regenerates the eval on 10 labelled synthetic documents."),
    ], top=Inches(4.5), size=15.5)
    footer(s, 5, TOTAL)
    notes(s, "If asked about the 0.831 recall: the misses are names and organisations the zero-shot model overlooks — precision-first is a deliberate policy for a redaction tool, because every flag is human-verifiable with its reason string.")


def slide_snapdragon(prs):
    s = base_slide(prs)
    title_bar(s, "Snapdragon & Qualcomm AI Hub", "Compiled and executed on the real X Elite NPU")
    bullets(s, [
        ("Pipeline: ", "GLiNER-PII → ONNX (static shapes) → QNN context binary → Snapdragon X Elite CRD (Hexagon v73) via AI Hub."),
        ("Real jobs, real device: ", "compile jgnz1qovg SUCCESS · on-device inference jp1no1r8g SUCCESS with genuine tokenized input."),
        ("Two upstream fixes contributed: ", "int64 ReduceMax lowered to fp32 casts; span-Einsum rewritten as Transpose+MatMul — proven bitwise-identical (9/9 entities, max diff 0.0)."),
        ("On-device load: ", "17.7 s cold → 1.2 s warm; every load/inference step logged by the device runtime."),
    ])
    footer(s, 6, TOTAL)


def slide_findings(prs):
    s = base_slide(prs)
    title_bar(s, "What the NPU taught us", "Honest benchmarks beat optimistic ones")
    stat_card(s, Inches(0.9), Inches(2.1), Inches(3.7), Inches(1.5), "0.959", "logits cosine, HTP fp16 vs CPU fp32", CYAN)
    stat_card(s, Inches(4.85), Inches(2.1), Inches(3.7), Inches(1.5), "0 / 9", "entities surviving fp16 execution", AMBER)
    stat_card(s, Inches(8.8), Inches(2.1), Inches(3.7), Inches(1.5), "0 / 9", "entities surviving naive int8 (CPU)", AMBER)
    bullets(s, [
        ("The finding: ", "GLiNER's LSTM span-scorer runs on razor-thin logit margins; fp16 accumulation across 4,950 nodes compresses the range (CPU [-68.6, +6.6] → HTP [-15.6, -1.5]) and erases the head."),
        ("The engineering answer: ", "the two-tier design absorbs it — the product ships the CPU AI tier (285 ms) and stays correct; every number is in benchmarks/npu.md §2b with job IDs."),
        ("The roadmap: ", "an NPU-viable span head (DeBERTa-style) and vocabulary pruning (the embedding table is 66% of the artifact) are the two moves queued next."),
    ], top=Inches(3.9), size=15)
    footer(s, 7, TOTAL)
    notes(s, "This is the credibility slide. Judges see dozens of 'it works on NPU' claims; this project measured it, found the edge, said so publicly, and designed the product so the answer doesn't depend on the NPU.")


def slide_deployment(prs):
    s = base_slide(prs)
    title_bar(s, "Deployment & accessibility", "Runs anywhere a pip install works")
    bullets(s, [
        ("Zero special hardware needed to run: ", "CPU-only ONNX Runtime, no torch at inference; the regex tier runs with the AI stack absent entirely."),
        ("Built on a conventional PC, validated on the target: ", "development happened on an Intel machine; NPU validation used AI Hub's cloud Snapdragon X Elite CRD — the exact hardware the challenge targets."),
        ("Two front doors: ", "python app.py … CLI for scripts and batch folders, Streamlit UI with file upload + download buttons for everyone else."),
        ("38 tests, every benchmark reproducible ", "with one command; sample data is synthetic — no real identifiers anywhere."),
    ])
    footer(s, 8, TOTAL)


def slide_roadmap(prs):
    s = base_slide(prs)
    title_bar(s, "Roadmap", "Next 90 days")
    bullets(s, [
        ("NPU-viable detector head. ", "Swap the LSTM span-scorer for a DeBERTa-style head that survives fp16, re-export, re-verify on the same AI Hub pipeline."),
        ("Vocabulary pruning. ", "250k multilingual tokens → the subset reachable from Indian financial documents; 1.1 GB artifact → ~400 MB, no accuracy cost on this domain."),
        ("Calibrated quantisation. ", "QNN PTQ with calibration data instead of CPU dynamic int8 — the path npu.md already points to."),
        ("Watch-mode daemon + policy packs. ", "Folder policies per department (finance, HR) with separate vaults."),
    ])
    footer(s, 9, TOTAL)


def slide_close(prs):
    s = base_slide(prs)
    tf = textbox(s, Inches(0.9), Inches(2.3), Inches(11.6), Inches(2.6))
    p = tf.paragraphs[0]
    set_runs(p, [("Nothing leaves the machine.", 40, True, WHITE)])
    p2 = tf.add_paragraph()
    set_runs(p2, [("Every claim in this deck is a command you can run.", 20, False, GREY)])
    tf2 = textbox(s, Inches(0.9), Inches(5.4), Inches(11.6), Inches(1.4))
    p = tf2.paragraphs[0]
    set_runs(p, [
        ("github.com/upadhyayaditya7/redacta-ai", 16, True, CYAN),
        ("   ·   38 tests   ·   benchmarks/npu.md   ·   docs/Redacta_AI_Proposal.pdf", 14, False, GREY),
    ])
    footer(s, 10, TOTAL)


def build() -> Presentation:
    prs = new_deck()
    slide_title(prs)
    slide_problem(prs)
    slide_solution(prs)
    slide_architecture(prs)
    slide_accuracy(prs)
    slide_snapdragon(prs)
    slide_findings(prs)
    slide_deployment(prs)
    slide_roadmap(prs)
    slide_close(prs)
    prs.save(str(OUT_PPTX))
    print(f"wrote {OUT_PPTX} ({OUT_PPTX.stat().st_size / 1024:.0f} KB, {TOTAL} slides)")
    return prs


def export_pdf() -> Path | None:
    try:
        import win32com.client  # noqa: PLC0415
    except ImportError:
        print("pywin32 not available - skipping PDF export")
        return None
    ppt_path = str(OUT_PPTX.resolve())
    pdf_path = str((DOCS / "Redacta_AI_Pitch.pdf").resolve())
    app = win32com.client.Dispatch("PowerPoint.Application")
    pres = app.Presentations.Open(ppt_path, True, False, False)  # ReadOnly, no window
    try:
        pres.SaveAs(pdf_path, 32)  # 32 = ppSaveAsPDF
    finally:
        pres.Close()
        app.Quit()
    out = Path(pdf_path)
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
    return out


if __name__ == "__main__":
    build()
    if "--pdf" in sys.argv:
        export_pdf()
