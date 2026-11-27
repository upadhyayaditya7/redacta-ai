"""Redacta AI — on-device PII redaction studio (Streamlit UI).

Run:  python -m streamlit run frontend/app.py

Design note: all custom HTML is rendered via st.markdown into the main
document so it inherits the single CSS block below.  No iframes, and no
external font/CDN requests — the zero-network claim survives a packet
sniffer.
"""

from __future__ import annotations

import html
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from core import __version__
from core.audit import build_record
from core.entities import ENTITY_SPECS, EntityType
from core.verify import verify_output, verify_text
from core.ner_detector import ONNX_DIR as NER_ONNX_DIR
from core.pipeline import RedactionPipeline
from core.redact import redact_image, redact_pdf, redact_text
from core.vault import Vault
from utils.samples import document

st.set_page_config(
    page_title="Redacta — on-device PII redaction",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------ theme ----
# One CSS block for the whole page.  System fonts only (no Google Fonts):
# the app must make ZERO network requests — that's the product's promise.
# Palette is deliberately neutral (ink + hairline) with a single lime
# signal colour, so nothing looks like a generated landing page.

CSS = """<style>
:root{
--bg:#08080a;--panel:#0e0e11;--panel2:#131317;--line:#1f1f24;--line2:#2c2c33;
--text:#ececee;--muted:#8c8c96;--dim:#61616b;--trace:#4a4a53;
--accent:#d8f24e;--ok:#5fd98d;--warn:#e5b145;--crit:#ef6f6f;
--mono:"Cascadia Mono","Cascadia Code",ui-monospace,Consolas,"SF Mono",monospace;
--ui:"Segoe UI Variable Text","Segoe UI",system-ui,-apple-system,Roboto,Arial,sans-serif;
--display:"Segoe UI Variable Display","Segoe UI Variable Text","Segoe UI",system-ui,sans-serif;
}
html,body,.stApp,[class*=css],.stMarkdown,p,h1,h2,h3,label,button,input,textarea{
font-family:var(--ui)}
.stApp{background-color:var(--bg);color:var(--text);
background-image:radial-gradient(rgba(255,255,255,.045) 1px,transparent 1px);
background-size:26px 26px;background-position:-2px -2px}
[data-testid=stHeader],#MainMenu,footer,[data-testid=stToolbar],[data-testid=stDecoration]{display:none!important}
[data-testid=stMainBlockContainer]{padding:0 32px 56px;max-width:1180px;margin:0 auto}
[data-testid=stMain]{background:transparent}
::selection{background:rgba(216,242,78,.26);color:#fff}
[data-testid=stHeaderActionElements],.stMarkdown h1 a{display:none!important}

/* ---------------------------------------------------------- top bar */
.topbar{display:flex;align-items:center;gap:11px;height:56px;margin:0 -32px;
padding:0 32px;border-bottom:1px solid var(--line);background:rgba(8,8,10,.82);
-webkit-backdrop-filter:blur(10px);backdrop-filter:blur(10px);
position:sticky;top:0;z-index:80}
.mark{width:22px;height:22px;border-radius:6px;flex:none;position:relative;
background:linear-gradient(180deg,#fff,#d5d5da);box-shadow:0 1px 0 rgba(255,255,255,.5) inset}
.mark .bar{position:absolute;left:4px;right:4px;top:8px;height:7px;border-radius:1.5px;
background:#09090b}
.wordmark{font-size:13.5px;font-weight:650;letter-spacing:-.01em;color:#f3f3f5}
.chip{font-family:var(--mono);font-size:10.5px;letter-spacing:.06em;color:var(--dim);
border:1px solid var(--line);border-radius:4px;padding:2px 6px;text-transform:uppercase}
.spacer{flex:1}
.live{display:inline-flex;align-items:center;gap:7px;font-family:var(--mono);
font-size:10.5px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted)}
.live i{width:6px;height:6px;border-radius:50%;background:var(--ok);flex:none;
box-shadow:0 0 0 3px rgba(95,217,141,.13)}
.live.sep{border-left:1px solid var(--line);padding-left:14px;margin-left:4px}

/* ------------------------------------------------------ hero block */
.hero{padding:46px 0 24px;max-width:780px}
.eyebrow{font-family:var(--mono);font-size:10.5px;letter-spacing:.22em;
text-transform:uppercase;color:var(--trace);margin:0 0 15px}
.hero h1{font-family:var(--display);font-size:clamp(27px,3vw,39px);line-height:1.07;
letter-spacing:-.035em;font-weight:600;color:#f5f5f7;margin:0 0 17px}
.hero h1 em{font-style:normal;color:#fff;
background:linear-gradient(transparent 61%,rgba(216,242,78,.26) 61%,rgba(216,242,78,.26) 94%,transparent 94%)}
.lede{font-size:14.5px;line-height:1.72;color:var(--muted);margin:0;max-width:660px}
.lede b{color:#dcdce3;font-weight:600}
.spec{display:flex;flex-wrap:wrap;align-items:center;border-top:1px solid var(--line);
border-bottom:1px solid var(--line);padding:11px 0;margin:0 0 26px}
.spec span{font-family:var(--mono);font-size:11px;letter-spacing:.08em;text-transform:uppercase;
color:var(--trace);padding:0 14px;border-left:1px solid var(--line);white-space:nowrap}
.spec span:first-child{padding-left:0;border-left:none}
.spec span:last-child{margin-left:auto;padding-right:0;border-left:none;color:var(--dim)}
.spec b{color:#b9b9c3;font-weight:500}

/* ----------------------------------------------------------- panels */
[data-testid=stMain] [data-testid=stVerticalBlockBorderWrapper]{
border:1px solid var(--line)!important;border-radius:13px!important;
background:var(--panel)!important;padding:2px!important}
[data-testid=stMain] [data-testid=stVerticalBlockBorderWrapper] [data-testid=stVerticalBlock]{gap:.55rem}
.panel-head{display:flex;align-items:baseline;justify-content:space-between;gap:18px;
margin:0 0 9px;flex-wrap:wrap}
.panel-head .eyebrow{margin:0;font-size:10px;color:var(--dim);letter-spacing:.18em}
.panel-head .meta{font-family:var(--mono);font-size:10.5px;letter-spacing:.04em;color:var(--trace)}
.count{font-family:var(--mono);font-size:11px;letter-spacing:.06em;color:var(--trace);
text-align:right;padding-top:9px}

/* ------------------------------------------------------- text input */
.stTextArea textarea{background:#09090c!important;border:1px solid var(--line)!important;
border-radius:9px!important;color:#e7e7ec!important;font-family:var(--mono)!important;
font-size:12.5px!important;line-height:1.8!important;padding:14px 15px!important;
min-height:190px!important}
.stTextArea textarea:focus{border-color:var(--line2)!important;outline:none!important;
box-shadow:0 0 0 3px rgba(216,242,78,.09)!important}
.stTextArea textarea::placeholder{color:#4c4c55!important}
.stTextArea [data-testid=stWidgetLabel]{display:none}

/* ----------------------------------------------------------- buttons */
.stButton>button,.stDownloadButton>button{
height:34px;min-height:34px;border-radius:8px!important;border:1px solid var(--line2)!important;
background:#15151a!important;color:#e2e2e8!important;font-size:12.5px!important;
font-weight:550!important;letter-spacing:-.005em;padding:0 14px!important;
box-shadow:none!important;transition:background .12s ease,border-color .12s ease}
/* Descendant, not child: a button carrying a `help` tooltip is wrapped in an extra
   div, so `> button` misses it. */
.stButton button p,.stDownloadButton button p{font-size:12.5px!important;
font-weight:inherit!important;line-height:1.1!important;color:inherit!important}
/* Streamlit wraps the label in an ellipsis div whose tall line box parks the text
   on its baseline, which reads a few pixels low inside a 34px button. */
.stButton button [data-testid=stMarkdownContainer],
.stDownloadButton button [data-testid=stMarkdownContainer]{
display:flex!important;align-items:center!important;justify-content:center!important;
line-height:1.1!important;height:auto!important;min-height:0!important}
.stButton button p,.stDownloadButton button p{vertical-align:baseline!important;
margin:0!important}
.stButton>button:hover,.stDownloadButton>button:hover{background:#1c1c22!important;
border-color:#3b3b45!important;color:#fff!important}
.stButton>button:active,.stDownloadButton>button:active{transform:translateY(1px)}
.stButton>button:focus-visible,.stDownloadButton>button:focus-visible{
outline:none!important;box-shadow:0 0 0 3px rgba(216,242,78,.18)!important}
[data-testid=stBaseButton-primary],.stButton>button[kind=primary]{
background:#ededf0!important;border-color:#ededf0!important;box-shadow:none!important}
[data-testid=stBaseButton-primary] p,.stButton>button[kind=primary] p{color:#08080a!important;
font-weight:650!important}
[data-testid=stBaseButton-primary]:hover,.stButton>button[kind=primary]:hover{
background:#fff!important;border-color:#fff!important}
[data-testid=stBaseButton-primary]:disabled,[data-testid=stBaseButton-secondary]:disabled{
opacity:.42!important}

/* -------------------------------------------------------------- tabs */
[role=tablist]{gap:26px!important;border-bottom:1px solid var(--line)!important;
background:transparent!important;padding:0!important}
[data-testid=stTab]{background:transparent!important;border:none!important;
border-radius:0!important;padding:0 0 11px!important;position:relative;
box-shadow:none!important;height:auto!important;min-height:0!important}
[data-testid=stTab] > div[data-rac]{display:none!important}
[data-testid=stTab] p{font-family:var(--mono)!important;font-size:11px!important;
letter-spacing:.14em;text-transform:uppercase;color:var(--trace)!important;
font-weight:500!important;line-height:1.4!important;margin:0!important}
[data-testid=stTab]:hover p{color:#a8a8b2!important}
[data-testid=stTab][aria-selected=true] p{color:#f1f1f4!important}
[data-testid=stTab][aria-selected=true]::after{content:"";position:absolute;left:0;right:0;
bottom:-1px;height:1.5px;background:#e9e9ee}

/* ------------------------------------------------------------ tables */
.ftable{border:1px solid var(--line);border-radius:10px;overflow:hidden;background:#0b0b0e}
.frow{display:grid;grid-template-columns:154px minmax(0,1.05fr) minmax(0,1.4fr) 58px;
gap:14px;align-items:center;padding:10px 14px;border-bottom:1px solid #17171c}
.frow:last-child{border-bottom:none}
.frow.body:hover{background:#101015}
.frow.hdr{background:#0f0f13;padding:8px 14px}
.frow.hdr span{font-family:var(--mono);font-size:9.5px;letter-spacing:.18em;
text-transform:uppercase;color:#5a5a64}
.f-type{font-family:var(--mono);font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;
color:#a3a3ae;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.f-val{font-family:var(--mono);font-size:12px;color:#f2f2f6;white-space:nowrap;
overflow:hidden;text-overflow:ellipsis}
.f-why{font-size:12px;color:var(--trace);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.f-why b{color:#a9a9b5;font-weight:500;font-family:var(--mono);font-size:10.5px}
.f-score{font-family:var(--mono);font-size:11px;text-align:right;color:var(--muted)}
.f-score.hi{color:var(--ok)}
.scroll{max-height:470px;overflow-y:auto}
.scroll::-webkit-scrollbar{width:6px}
.scroll::-webkit-scrollbar-thumb{background:#26262d;border-radius:6px}
.empty{color:var(--trace);font-size:12.5px;padding:18px 14px}

/* ------------------------------------------------------------- stats */
.stats{display:grid;grid-template-columns:repeat(4,1fr);border:1px solid var(--line);
border-radius:10px;overflow:hidden;background:#0b0b0e;margin:2px 0 16px}
.stats.three{grid-template-columns:repeat(3,1fr)}
.stat{padding:13px 16px;border-left:1px solid var(--line)}
.stat:first-child{border-left:none}
.stat-num{font-family:var(--mono);font-size:20px;font-weight:500;letter-spacing:-.02em;
line-height:1.25;color:#f2f2f5}
.stat-num.accent{color:var(--accent)}
.stat-lbl{font-family:var(--mono);font-size:9.5px;letter-spacing:.16em;text-transform:uppercase;
color:#5f5f69;margin-top:5px}

/* -------------------------------------------------------- verification */
.verify{display:flex;align-items:center;gap:11px;border:1px solid rgba(95,217,141,.26);
background:rgba(95,217,141,.055);border-radius:9px;padding:10px 14px;
font-size:12.5px;color:#c3e9d2;margin:0 0 14px;flex-wrap:wrap}
.verify .g{font-family:var(--mono);font-size:10px;letter-spacing:.16em;text-transform:uppercase;
color:#8fe3ae;border:1px solid rgba(95,217,141,.32);border-radius:4px;padding:2px 7px;flex:none}
.verify b{font-family:var(--mono);font-weight:500;color:#effff5}
.verify.bad{border-color:rgba(239,111,111,.32);background:rgba(239,111,111,.07);color:#f2bcbc}
.verify.bad .g{color:#f4a3a3;border-color:rgba(239,111,111,.38)}

/* --------------------------------------------------------- token rows */
.tok-row{display:grid;grid-template-columns:minmax(0,1fr) 16px minmax(0,1.25fr);
gap:10px;align-items:center;padding:8px 0;border-bottom:1px solid #16161b;
font-family:var(--mono);font-size:11.5px}
.tok-row:last-child{border-bottom:none}
.tok-row .tok{color:#e6e6ec}
.tok-row .arr{color:#3a3a44;text-align:center}
.tok-row .hint{color:var(--trace)}

/* -------------------------------------------------------------- alerts */
[data-testid=stAlert],[data-testid=stAlertContainer]{border-radius:9px!important;
background:var(--panel2)!important;border:1px solid var(--line)!important;
box-shadow:none!important}
[data-testid=stAlertContainer] p{font-size:12.5px!important;color:var(--muted)!important}
[data-testid=stAlertContainer] p b,[data-testid=stAlertContainer] p strong{color:#e6e6ec!important}

/* ---------------------------------------------------------- code / json */
[data-testid=stCode],[data-testid=stJson]{background:#09090c!important;
border:1px solid var(--line)!important;border-radius:9px!important}
[data-testid=stCode] pre,[data-testid=stCode] code{font-family:var(--mono)!important;
font-size:11.5px!important;line-height:1.7!important;color:#d6d6df!important;
background:transparent!important}
[data-testid=stJson] *{font-family:var(--mono)!important;font-size:11.5px!important}

/* -------------------------------------------------------- file upload */
[data-testid=stFileUploaderDropzone]{background:#09090c!important;
border:1px dashed #2b2b33!important;border-radius:10px!important;padding:16px 18px!important;
min-height:0!important;transition:border-color .12s ease,background .12s ease}
[data-testid=stFileUploaderDropzone]:hover{border-color:#3e3e49!important;background:#0b0b0e!important}
[data-testid=stFileUploaderDropzoneInstructions] span{font-family:var(--mono)!important;
font-size:11px!important;color:var(--muted)!important;letter-spacing:.02em}
[data-testid=stFileUploaderDropzoneInstructions] small{font-family:var(--mono)!important;
font-size:10.5px!important;color:var(--trace)!important;letter-spacing:.04em}
[data-testid=stFileUploaderDropzone] button{height:28px!important;min-height:28px!important;
font-size:11px!important;padding:0 11px!important;border-radius:7px!important;
background:#17171d!important;border:1px solid var(--line2)!important;color:#dcdce4!important}
[data-testid=stFileUploaderFile]{font-family:var(--mono)!important;font-size:11px!important}

/* ----------------------------------------------------------- section head */
.section-head{display:flex;align-items:baseline;justify-content:space-between;gap:16px;
margin:30px 0 12px;flex-wrap:wrap}
.section-head .eyebrow{margin:0}
.section-head .meta{font-family:var(--mono);font-size:10.5px;letter-spacing:.08em;
color:var(--trace);text-transform:uppercase}

/* --------------------------------------------------------------- rail */
[data-testid=stSidebar]{background:#0a0a0c;border-right:1px solid var(--line)}
[data-testid=stSidebar] [data-testid=stSidebarUserContent]{padding:42px 18px 26px}
[data-testid=stSidebar] [data-testid=stSidebarHeader]{display:none}
.rail-head{display:flex;align-items:center;gap:10px;padding:0 0 17px;
border-bottom:1px solid var(--line);margin:0 0 4px}
.rail-name{font-size:13px;font-weight:650;color:#f2f2f5;letter-spacing:-.01em}
.rail-ver{font-family:var(--mono);font-size:10px;letter-spacing:.1em;text-transform:uppercase;
color:var(--trace);margin-top:2px}
.rail-label{display:block;font-family:var(--mono);font-size:10px;letter-spacing:.2em;
text-transform:uppercase;color:#6b6b76;margin:18px 0 8px}
[data-testid=stSidebar] p.rail-help{font-size:11.5px!important;line-height:1.62;color:#90909b!important;
margin:0 0 10px!important;font-weight:400!important;max-width:100%}
.rail-note{display:flex;align-items:center;gap:7px;font-family:var(--mono);font-size:10.5px;
letter-spacing:.04em;color:var(--trace);margin:9px 0 0}
.rail-note i{width:5px;height:5px;border-radius:50%;background:var(--trace);flex:none}
.rail-note i.ok{background:var(--ok)}
.rail-note i.warn{background:var(--warn)}
/* Buttons carrying a `help` tooltip get a 40px min-height and 4px padding from
   Streamlit, which makes them sit taller than their neighbours in a row. */
[data-testid=stSidebar] .stButton button{width:100%;height:32px!important;min-height:32px!important;
font-size:12px!important;padding:0 14px!important;background:#14141a!important}
[data-testid=stSidebar] .stButton button p{font-size:12px!important}
.stTextInput input{background:#0c0c0f!important;border:1px solid var(--line)!important;
border-radius:8px!important;color:#e8e8ee!important;font-family:var(--mono)!important;
font-size:11.5px!important;height:34px!important;padding:0 11px!important}
.stTextInput input:focus{border-color:var(--line2)!important;outline:none!important;
box-shadow:0 0 0 3px rgba(216,242,78,.09)!important}
.stTextInput input::placeholder{color:#4c4c55!important}
[data-testid=stSidebar] [data-testid=stWidgetLabel] p{font-size:11px!important;color:var(--muted)!important}
[data-baseweb=tag]{background:#1a1a20!important;border:1px solid var(--line2)!important;
color:#c9c9d3!important;border-radius:6px!important;font-family:var(--mono)!important;
font-size:10.5px!important}
[data-baseweb=tag] svg{color:#71717c!important}
[data-testid=stExpander]{border:1px solid var(--line)!important;border-radius:9px!important;
background:transparent!important;margin-top:2px}
[data-testid=stExpander] summary{padding:9px 12px!important}
[data-testid=stExpander] summary p{font-family:var(--mono);font-size:10.5px!important;
letter-spacing:.14em;text-transform:uppercase;color:var(--muted)!important}
[data-testid=stExpander] svg{color:#565660}
[data-testid=stSidebar] hr{border-color:var(--line);margin:18px 0 12px}
[data-testid=stSidebar] .stCaption p,[data-testid=stSidebar] [data-testid=stCaptionContainer] p{
font-family:var(--mono);font-size:10.5px!important;color:var(--trace)!important;letter-spacing:.03em}
[data-testid=stSidebar] .stCaption a{color:#a8a8b4!important;text-decoration:none;
border-bottom:1px solid var(--line2)}
.toggle-label{font-size:12px;color:#dcdce3}

/* -------------------------------------------------------- narrow view */
@media (max-width:860px){
.hero{padding:34px 0 20px}
.frow{grid-template-columns:minmax(0,1fr) 44px;gap:8px;padding:10px 12px}
.frow.hdr{display:none}
.frow>.f-type{grid-column:1/-1;font-size:8.5px;color:var(--dim)}
.f-why{display:none}
.stats,.stats.three{grid-template-columns:repeat(2,1fr)}
.stat:nth-child(odd){border-left:none}
.stat:nth-child(n+3){border-top:1px solid var(--line)}
.spec span:last-child{display:none}
.foot{flex-direction:column;gap:6px}
}

/* --------------------------------------------------------------- foot */
.foot{display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap;
border-top:1px solid var(--line);margin-top:44px;padding:15px 0 0;
font-family:var(--mono);font-size:9.5px;letter-spacing:.07em;text-transform:uppercase;
color:#4c4c55}
.foot b{color:#8a8a94;font-weight:500}
</style>"""

st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------- session -----
if "pipeline" not in st.session_state:
    st.session_state.pipeline = RedactionPipeline(use_ner=True)
if "vault" not in st.session_state:
    st.session_state.vault = Vault()
if "ner_ready" not in st.session_state:
    st.session_state.ner_ready = importlib.util.find_spec("gliner") is not None
if "vault_unlocked" not in st.session_state:
    st.session_state.vault_unlocked = False
if "doc_input" not in st.session_state:
    st.session_state.doc_input = ""

vault_obj: Vault = st.session_state.vault
pipeline: RedactionPipeline = st.session_state.pipeline


def _load_sample() -> None:
    """Runs before the rerun renders, so the text_area picks the value up."""
    st.session_state.doc_input = document()


def _clear_doc() -> None:
    st.session_state.doc_input = ""


def _types_from_selection(selected: list[str]) -> list[EntityType] | None:
    if not selected or len(selected) == len(EntityType):
        return None
    return [EntityType(t) for t in selected]


def _verify_bar(check, bad_prefix: str = "") -> str:
    """Flat verification strip — the one claim the product makes out loud."""
    if check.ok:
        return (
            '<div class="verify"><span class="g">Verified</span>'
            f'<span><b>{check.checked_values}</b> redacted values absent from the output · '
            f'{check.checked_chars:,} chars re-read</span></div>'
        )
    return (
        f'<div class="verify bad"><span class="g">Leak</span>'
        f"<span>{bad_prefix}{html.escape(check.summary())}</span></div>"
    )


# ---------------------------------------------------------- top bar ----
st.markdown(
    f"""<div class="topbar">
<span class="mark"><span class="bar"></span></span>
<span class="wordmark">Redacta</span>
<span class="chip">v{__version__}</span>
<span class="spacer"></span>
<span class="live"><i></i>Local engine</span>
<span class="live sep">No network egress</span>
</div>""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------- hero ----
st.markdown(
    """<div class="hero">
<div class="eyebrow">PII redaction · local-first</div>
<h1>Detect the identifiers.<br>Redact them. <em>Prove they're gone.</em></h1>
<p class="lede">Drop a bank statement, ID scan or log export. Values are matched by
<b>checksum-validated rules</b> — Verhoeff Aadhaar, PAN structure, Luhn card numbers —
swapped for reversible vault tokens, then the finished file is <b>re-read</b> to confirm
nothing survived the redaction.</p>
</div>
<div class="spec">
<span><b>Aadhaar</b> · PAN</span>
<span><b>Cards</b> · Luhn</span>
<span><b>IFSC</b> · UPI</span>
<span><b>Keys</b> · tokens</span>
<span>Runs offline</span>
</div>""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------- sidebar ---
with st.sidebar:
    st.markdown(
        '<div class="rail-head">'
        '<span class="mark"><span class="bar"></span></span>'
        "<div><div class=\"rail-name\">Redacta</div>"
        f'<div class="rail-ver">v{__version__} · on-device</div></div>'
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<span class="rail-label">Detection</span>', unsafe_allow_html=True)
    use_ner = st.toggle(
        "Neural NER tier", value=True,
        help="Adds names/locations/orgs. Falls back to regex-only when unavailable.",
    )
    if st.session_state.ner_ready:
        backend = "ONNX" if (NER_ONNX_DIR / "model.onnx").exists() else "PyTorch"
        st.markdown(
            f'<div class="rail-note"><i class="ok"></i>GLiNER {backend} · NER on top of regex</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="rail-note"><i class="warn"></i>Regex-only · GLiNER not installed</div>',
            unsafe_allow_html=True,
        )
    if use_ner != (pipeline.ner is not None):
        st.session_state.pipeline = RedactionPipeline(use_ner=use_ner)

    st.markdown('<span class="rail-label">Vault</span>', unsafe_allow_html=True)
    st.markdown(
        '<p class="rail-help">Keeps every replaced value encrypted on this device. '
        'Unlock it to swap redactions back into a document later; leave it locked and '
        'redactions are permanent.</p>',
        unsafe_allow_html=True,
    )
    vault_pass = st.text_input(
        "Vault passphrase", type="password", label_visibility="collapsed",
        placeholder="Passphrase",
        help="Decrypts redactions locally — never leaves this machine.",
    )
    # Unlock and create are deliberately separate actions: a mistyped passphrase
    # must never be read as "there is no vault here" and overwrite the real one,
    # which would make every «RDCT-…» token ever issued unrecoverable.
    if vault_obj.exists():
        if st.button("Unlock vault", use_container_width=True):
            if not vault_pass:
                st.warning("Enter a passphrase first.")
            else:
                try:
                    vault_obj.load(vault_pass)
                    st.session_state.vault_unlocked = True
                    st.success("Vault unlocked.")
                except Exception as exc:  # noqa: BLE001 - show the real reason
                    st.session_state.vault_unlocked = False
                    st.error(f"Could not unlock this vault: {exc}")
    else:
        if st.button("Create vault", use_container_width=True):
            if not vault_pass:
                st.warning("Enter a passphrase first.")
            else:
                vault_obj.create(vault_pass)
                st.session_state.vault_unlocked = True
                st.success("Vault created — keep this passphrase safe.")

    if st.session_state.vault_unlocked:
        st.markdown(
            '<div class="rail-note"><i class="ok"></i>Unlocked · tokens reversible</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="rail-note"><i></i>Locked · redactions irreversible</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<span class="rail-label">Workspace</span>', unsafe_allow_html=True)
    st.markdown(
        '<p class="rail-help">Load a sample statement or clear the editor, then choose '
        'which identifier types to look for.</p>',
        unsafe_allow_html=True,
    )
    b_col1, b_col2 = st.columns(2)
    b_col1.button("Sample", use_container_width=True, on_click=_load_sample,
                  help="Fresh fake bank statement — checksum-valid, nothing real, nothing networked.")
    b_col2.button("Clear", use_container_width=True, on_click=_clear_doc)

    with st.expander("Entity types"):
        defaults = [t.value for t in EntityType]
        selected = st.multiselect("Detect", defaults, default=defaults, label_visibility="collapsed")

    st.divider()
    st.caption("CLI: `python app.py scan …` · [github.com/upadhyayaditya7/redacta-ai]"
               "(https://github.com/upadhyayaditya7/redacta-ai)")

# A passphrase typed but not unlocked (wrong passphrase, corrupt vault) must not
# be used for storage: redact irreversibly instead of raising mid-run.
vault_active = vault_obj if st.session_state.get("vault_unlocked") else None

# ------------------------------------------------------- document ------
with st.container(border=True):
    st.markdown(
        '<div class="panel-head"><span class="eyebrow">Document</span>'
        '<span class="meta">pasted text is never written to disk</span></div>',
        unsafe_allow_html=True,
    )
    input_text = st.text_area(
        "Document",
        key="doc_input",
        height=200,
        label_visibility="collapsed",
        placeholder="Paste a bank statement, ID copy, KYC form or log excerpt…",
    )
    b1, b2, b3 = st.columns([1.1, 1.1, 3.4], vertical_alignment="center")
    run_scan = b1.button("Scan", type="primary", use_container_width=True)
    run_redact = b2.button("Redact", use_container_width=True)
    b3.markdown(f'<div class="count">{len(input_text):,} characters</div>',
                unsafe_allow_html=True)

# ------------------------------------------------------ file upload ----
with st.container(border=True):
    st.markdown(
        '<div class="panel-head"><span class="eyebrow">File</span>'
        '<span class="meta">PDF · PNG · JPG · TXT · CSV · JSON</span></div>',
        unsafe_allow_html=True,
    )
    upload = st.file_uploader(
        "Upload", type=["pdf", "png", "jpg", "jpeg", "bmp", "webp", "txt", "md", "csv", "json"],
        label_visibility="collapsed",
        help="Processed locally — the file never leaves this machine.",
    )

if upload is not None:
    suffix = Path(upload.name).suffix.lower()
    tmp = Path("data") / "uploads"
    tmp.mkdir(parents=True, exist_ok=True)
    local_path = tmp / upload.name
    local_path.write_bytes(upload.getvalue())

    up_col1, up_col2 = st.columns([1.2, 3.8])
    if up_col1.button("Redact file", type="primary", use_container_width=True):
        with st.spinner("Redacting on-device…"):
            try:
                if suffix == ".pdf":
                    result, out_path = redact_pdf(local_path, pipeline, vault_active,
                                                  _types_from_selection(selected))
                elif suffix in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
                    result, out_path = redact_image(local_path, pipeline, vault_active,
                                                    _types_from_selection(selected))
                else:
                    text = local_path.read_text(encoding="utf-8", errors="replace")
                    report = pipeline.analyze(text, _types_from_selection(selected))
                    result = redact_text(text, report, vault_active,
                                         _types_from_selection(selected))
                    out_path = tmp / (local_path.stem + ".redacted.txt")
                    out_path.write_text(result.redacted_text, encoding="utf-8")
                # Re-read what we just produced: detection counts are a claim,
                # this is the check that the values are actually gone.
                check = verify_output(out_path, result.report)
                st.session_state["file_result"] = (upload.name, out_path, result)
                st.session_state["file_verify"] = check
                # Audit record for the file run.
                rec = build_record(
                    source=upload.name,
                    report=result.report,
                    result=result,
                    vault_hints={t: vault_obj.entry_hint(t) for t in result.token_map} if vault_active else None,
                    verification=check.as_dict(),
                )
                st.session_state["file_audit"] = rec.to_json()
            except Exception as exc:  # noqa: BLE001 — surface OCR/parse errors in the UI
                st.error(f"Could not process this file: {exc}")

    if "file_result" in st.session_state and st.session_state["file_result"][0] == upload.name:
        name, out_path, fresult = st.session_state["file_result"]
        out_name = Path(out_path).name
        data = Path(out_path).read_bytes()
        check = st.session_state.get("file_verify")
        st.markdown(
            '<div class="section-head"><span class="eyebrow">File output</span>'
            f'<span class="meta">{html.escape(out_name)}</span></div>',
            unsafe_allow_html=True,
        )
        if check is not None:
            st.markdown(_verify_bar(check, "Verification failed — "), unsafe_allow_html=True)
            if not check.ok:
                for leak in check.leaks[:5]:
                    st.caption(f"• {leak.label} still readable in {leak.where} ({leak.hint})")
        st.markdown(
            '<div class="stats three">'
            f'<div class="stat"><div class="stat-num">{fresult.redacted_count}</div>'
            '<div class="stat-lbl">Entities redacted</div></div>'
            f'<div class="stat"><div class="stat-num">{fresult.vaulted_count}</div>'
            '<div class="stat-lbl">Vaulted · reversible</div></div>'
            f'<div class="stat"><div class="stat-num">{fresult.metadata_dropped + fresult.annotations_removed}</div>'
            '<div class="stat-lbl">Metadata · annotations cleared</div></div>'
            "</div>",
            unsafe_allow_html=True,
        )
        if fresult.metadata_dropped or fresult.annotations_removed:
            st.caption(
                f"Also cleared {fresult.metadata_dropped} metadata field(s) and removed "
                f"{fresult.annotations_removed} annotation(s) from the copy."
            )
        dl1, dl2 = st.columns([1.3, 1.3])
        if check is None or check.ok:
            dl1.download_button("Download redacted file", data, file_name=out_name,
                                use_container_width=True)
        else:
            dl1.button("Download withheld", disabled=True, use_container_width=True)
        if "file_audit" in st.session_state:
            dl2.download_button("Audit log (JSON)", st.session_state["file_audit"],
                                file_name=out_name + ".audit.json", mime="application/json",
                                use_container_width=True)

if run_scan or run_redact:
    if not input_text.strip():
        st.warning("Nothing to analyse yet — paste text or click Load sample.")
        st.stop()

    types = _types_from_selection(selected)
    with st.spinner("Analysing on-device…"):
        report = pipeline.analyze(input_text, types)

    rows = ['<div class="frow hdr"><span>Type</span><span>Value</span>'
            '<span>Basis</span><span style="text-align:right">Conf</span></div>']
    for s in report.spans:
        sp = ENTITY_SPECS[s.entity_type]
        hi = "hi" if s.score >= 0.99 else ""
        rows.append(
            '<div class="frow body">'
            f'<span class="f-type">{html.escape(str(sp.label))}</span>'
            f'<span class="f-val">{html.escape(s.text)}</span>'
            f'<span class="f-why"><b>{html.escape(s.detector)}</b> · {html.escape(s.reason)}</span>'
            f'<span class="f-score {hi}">{s.score:.2f}</span>'
            "</div>"
        )
    if not report.spans:
        rows.append('<div class="empty">No identifiers matched in this text.</div>')
    findings_body = '<div class="ftable">' + "".join(rows) + "</div>"

    verified = sum(1 for s in report.spans if s.score >= 0.99)
    st.markdown(
        '<div class="section-head"><span class="eyebrow">Results</span>'
        f'<span class="meta">{len(report.spans)} entities · {report.total_ms:.1f} ms</span></div>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(
        [f"Findings · {len(report.spans)}", "Redacted output", "Engine"]
    )

    with tab1:
        st.markdown(
            '<div class="stats">'
            f'<div class="stat"><div class="stat-num">{len(report.spans)}</div>'
            '<div class="stat-lbl">Entities found</div></div>'
            f'<div class="stat"><div class="stat-num accent">{verified}</div>'
            '<div class="stat-lbl">Checksum-verified</div></div>'
            f'<div class="stat"><div class="stat-num">{report.total_ms:.1f}<span style="font-size:11px;color:#61616b"> ms</span></div>'
            '<div class="stat-lbl">Scan time</div></div>'
            f'<div class="stat"><div class="stat-num">{report.text_length:,}</div>'
            '<div class="stat-lbl">Characters</div></div>'
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown('<div class="scroll">' + findings_body + "</div>", unsafe_allow_html=True)
        if not report.ner_available:
            st.caption("AI NER tier not installed here — checksum-validated regex ran solo. Still 100% local.")

    with tab2:
        if run_redact:
            result = redact_text(input_text, report, vault_active, types)
            st.session_state["last_result"] = result
        if "last_result" in st.session_state:
            result = st.session_state["last_result"]
            tcheck = verify_text(result.redacted_text, result.report)
            st.markdown(
                '<div class="stats three">'
                f'<div class="stat"><div class="stat-num">{result.redacted_count}</div>'
                '<div class="stat-lbl">Redacted</div></div>'
                f'<div class="stat"><div class="stat-num accent">{result.vaulted_count}</div>'
                '<div class="stat-lbl">Vaulted · reversible</div></div>'
                f'<div class="stat"><div class="stat-num">{len(result.redacted_text):,}</div>'
                '<div class="stat-lbl">Output chars</div></div>'
                "</div>",
                unsafe_allow_html=True,
            )
            st.markdown(_verify_bar(tcheck, "Verification failed — "), unsafe_allow_html=True)
            st.markdown('<span class="rail-label">Sanitised document</span>',
                        unsafe_allow_html=True)
            st.code(result.redacted_text or "—", language="text")
            if result.token_map and vault_active:
                token_rows = "".join(
                    f'<div class="tok-row"><span class="tok">{tok}</span>'
                    f'<span class="arr">→</span><span class="hint">{vault_obj.entry_hint(tok)}</span></div>'
                    for tok in list(result.token_map)[:12]
                )
                st.markdown(
                    '<div class="panel-head" style="margin-top:16px">'
                    '<span class="eyebrow">Vault tokens</span>'
                    '<span class="meta">originals encrypted on this disk</span></div>'
                    + token_rows
                    + '<div class="rail-note">Reveal from the CLI: '
                      '<code>python app.py reveal «RDCT-…» --vault-pass …</code></div>',
                    unsafe_allow_html=True,
                )
            elif result.token_map:
                st.info("Vault locked — redactions were irreversible. Enter a passphrase in the rail "
                        "and redact again for reversible tokens.")

    with tab3:
        st.markdown('<span class="rail-label">Engine internals</span>', unsafe_allow_html=True)
        st.json({
            "regex_ms": round(report.regex_ms, 2),
            "ner_ms": round(report.ner_ms, 2),
            "ner_available": report.ner_available,
            "ner_backend": report.ner_backend,
            "ner_error": report.ner_error,
            "chars": report.text_length,
            "vault_path": str(vault_obj.path),
            "entity_types_selected": None if types is None else [t.value for t in types],
        })

st.markdown(
    '<div class="foot"><span><b>Redacta</b> · Snapdragon® AI Lab Build &amp; Present Challenge</span>'
    "<span>every byte processed on this device</span></div>",
    unsafe_allow_html=True,
)
