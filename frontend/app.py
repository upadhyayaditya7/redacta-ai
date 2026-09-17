"""Redacta AI — on-device PII redaction studio (Streamlit UI).

Run:  python -m streamlit run frontend/app.py

Design note: all custom HTML is rendered via st.markdown into the main
document so it inherits the single CSS block below.  No iframes, and no
external font/CDN requests — the zero-network claim survives a packet
sniffer.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from core.entities import ENTITY_SPECS, EntityType
from core.pipeline import RedactionPipeline
from core.redact import redact_text
from core.vault import Vault
from utils.samples import document

st.set_page_config(
    page_title="Redacta AI — on-device PII redaction",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="auto",
)

# ------------------------------------------------------------ theme ----
# One CSS block for the whole page.  System fonts only (no Google Fonts):
# the app must make ZERO network requests — that's the product's promise.

CSS = """<style>
:root{--accent:#8b5cf6;--accent2:#22d3ee;--good:#34d399;--warn:#fbbf24;
--text:#e8ebf5;--muted:#8b94ab;--line:rgba(255,255,255,.08)}
html,body,.stApp,[class*=css],.stMarkdown,p,h1,h2,h3{
font-family:"Segoe UI Variable Text","Segoe UI",system-ui,-apple-system,Roboto,Arial,sans-serif}
.stApp{background:
radial-gradient(1000px 480px at 10% -8%,rgba(139,92,246,.13),transparent 60%),
radial-gradient(760px 380px at 104% 0%,rgba(34,211,238,.09),transparent 55%),
#0a0e17;color:var(--text)}
#MainMenu,footer,.stDeployButton,header[data-testid=stHeader]{visibility:hidden}
.block-container{padding-top:2.4rem;max-width:1140px}
@keyframes rise{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}

/* editor + buttons */
.stTextArea textarea{background:linear-gradient(180deg,rgba(255,255,255,.045),rgba(255,255,255,.02))!important;
border:1px solid rgba(255,255,255,.10)!important;border-radius:14px!important;
color:var(--text)!important;font-size:.95rem!important;line-height:1.6!important;padding:16px!important}
.stTextArea textarea:focus{border-color:rgba(139,92,246,.6)!important;
box-shadow:0 0 0 3px rgba(139,92,246,.16)!important}
.stTextArea textarea::placeholder{color:#5d6680!important}
.stButton>button{border-radius:12px!important;font-weight:700!important;
border:1px solid rgba(255,255,255,.12)!important;color:var(--text)!important;
background:rgba(255,255,255,.04)!important;
transition:transform .12s ease,box-shadow .12s ease,border-color .12s ease}
.stButton>button:hover{transform:translateY(-1px);border-color:rgba(255,255,255,.24)!important}
.stButton>button[kind=primary]{background:linear-gradient(135deg,#8b5cf6,#6d28d9)!important;
border:none!important;color:#fff!important;box-shadow:0 8px 22px rgba(139,92,246,.32)!important}

/* sidebar */
[data-testid=stSidebar]{background:
radial-gradient(420px 240px at 0% 0%,rgba(139,92,246,.12),transparent 60%),
#0d1220;border-right:1px solid rgba(255,255,255,.06)}
[data-testid=stSidebar] .block-container{padding-top:1.6rem}
.stTextInput input{background:rgba(255,255,255,.05)!important;
border:1px solid rgba(255,255,255,.12)!important;border-radius:12px!important;color:var(--text)!important}
.stTextInput input:focus{border-color:rgba(139,92,246,.6)!important}
[data-baseweb=tag]{background:rgba(139,92,246,.16)!important;border:1px solid rgba(139,92,246,.4)!important;
color:#ddd6fe!important;border-radius:8px!important}

/* tabs */
.stTabs [data-baseweb=tab-list]{gap:4px;border-bottom:1px solid rgba(255,255,255,.08)}
.stTabs [data-baseweb=tab]{background:transparent;border:none;border-radius:10px;padding:9px 16px}
.stTabs [data-baseweb=tab] p{font-weight:700;font-size:.9rem;color:#8b94ab}
.stTabs [aria-selected=true]{background:rgba(139,92,246,.13)!important;border-radius:10px!important}
.stTabs [aria-selected=true] p{color:#ddd6fe!important}

/* stat cards */
.stat{background:linear-gradient(180deg,rgba(255,255,255,.05),rgba(255,255,255,.02));
border:1px solid rgba(255,255,255,.08);border-radius:16px;padding:14px 18px;height:100%}
.stat .num{font-size:1.9rem;font-weight:800;color:#f4f6fb;line-height:1.15}
.stat .lbl{font-size:.74rem;color:#8b94ab;font-weight:700;letter-spacing:.06em;text-transform:uppercase}
.stat.v .num{color:#c4b5fd}.stat.c .num{color:#67e8f9}.stat.g .num{color:#6ee7b7}

/* misc */
.glass{background:linear-gradient(180deg,rgba(255,255,255,.04),rgba(255,255,255,.015));
border:1px solid rgba(255,255,255,.08);border-radius:16px;padding:16px 18px}
.sec-title{font-size:.92rem;font-weight:700;color:#e8ebf5;margin:0 0 10px}
.stJson{background:rgba(255,255,255,.03)!important;border:1px solid rgba(255,255,255,.08)!important;border-radius:12px!important}
hr{border-color:rgba(255,255,255,.08)}
.foot{color:#6b7490;font-size:.84rem;text-align:center;padding:16px 0 2px}
.foot b{color:#9aa3b8}
</style>"""

st.markdown(CSS, unsafe_allow_html=True)

# --------------------------------------------------------- session -----
if "pipeline" not in st.session_state:
    st.session_state.pipeline = RedactionPipeline(use_ner=True)
if "vault" not in st.session_state:
    st.session_state.vault = Vault()
if "ner_ready" not in st.session_state:
    st.session_state.ner_ready = importlib.util.find_spec("gliner") is not None

vault_obj: Vault = st.session_state.vault
pipeline: RedactionPipeline = st.session_state.pipeline


def _types_from_selection(selected: list[str]) -> list[EntityType] | None:
    if not selected or len(selected) == len(EntityType):
        return None
    return [EntityType(t) for t in selected]


# [hero lands here]

# ----------------------------------------------------------- sidebar ---
with st.sidebar:
    st.markdown(
        '<div class="brand">🛡 Redacta<span class="grad">AI</span></div>'
        '<div class="brand-sub">On-device PII detection &amp; redaction.<br>Zero cloud. Full audit trail.</div>',
        unsafe_allow_html=True,
    )

    use_ner = st.toggle(
        "AI model (GLiNER NER)", value=True,
        help="Adds names/locations/orgs. Falls back to regex-only when unavailable.",
    )
    if st.session_state.ner_ready:
        st.markdown(
            '<div class="side-card">✅ <b style="color:#6ee7b7">GLiNER available</b> — '
            "NER tier active on top of the regex engine.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="side-card">ℹ️ <b style="color:#fcd34d">Regex-only mode</b> — GLiNER not '
            "installed here. Checksum-validated detection still runs, still 100% local.</div>",
            unsafe_allow_html=True,
        )
    if use_ner != (pipeline.ner is not None):
        st.session_state.pipeline = RedactionPipeline(use_ner=use_ner)

    st.markdown('<div class="side-label">Vault · reversible redaction</div>', unsafe_allow_html=True)
    vault_pass = st.text_input(
        "Vault passphrase", type="password", label_visibility="collapsed",
        placeholder="Passphrase…",
        help="Decrypts redactions locally — never leaves this machine.",
    )
    if st.button("🔐  Create / unlock vault", use_container_width=True):
        if not vault_pass:
            st.warning("Enter a passphrase first.")
        else:
            try:
                vault_obj.load(vault_pass)
                st.success("Vault unlocked.")
            except Exception:
                vault_obj.create(vault_pass)
                st.success("New vault created.")

    st.markdown('<div class="side-label">Try it</div>', unsafe_allow_html=True)
    b_col1, b_col2 = st.columns(2)
    if b_col1.button("🎲 Sample", use_container_width=True,
                     help="Fresh fake bank statement — checksum-valid, nothing real, nothing networked."):
        st.session_state["doc_input"] = document()
    if b_col2.button("🧽 Clear", use_container_width=True):
        st.session_state["doc_input"] = ""

    with st.expander("Entity types"):
        defaults = [t.value for t in EntityType]
        selected = st.multiselect("Detect", defaults, default=defaults, label_visibility="collapsed")

    st.divider()
    st.caption("CLI: `python app.py scan …` · [github.com/upadhyayaditya7/redacta-ai]"
               "(https://github.com/upadhyayaditya7/redacta-ai)")

# -------------------------------------------------------------- main ---
input_text = st.text_area(
    "Document",
    value=st.session_state.get("doc_input", ""),
    key="doc_input",
    height=200,
    label_visibility="collapsed",
    placeholder="Paste a bank statement, screenshot text, ID copy… (all processing stays on this device)",
)

b1, b2, b3 = st.columns([1, 1, 3])
run_scan = b1.button("🔍  Scan", type="primary", use_container_width=True)
run_redact = b2.button("🧹  Redact", use_container_width=True)

if run_scan or run_redact:
    if not input_text.strip():
        st.warning("Nothing to analyse yet — paste text or click 🎲 Sample.")
        st.stop()

    types = _types_from_selection(selected)
    with st.spinner("Analysing on-device…"):
        report = pipeline.analyze(input_text, types)

    tab1, tab2, tab3 = st.tabs(
        [f"📊 Findings · {len(report.spans)}", "🧹 Redacted output", "⚙ Engine"]
    )

    with tab1:
        st.success(f"**{len(report.spans)} entities** found in {report.total_ms:.1f} ms "
                   f"(regex {report.regex_ms:.1f} ms · NER {report.ner_ms:.1f} ms)")
        for s in report.spans:
            sp = ENTITY_SPECS[s.entity_type]
            st.markdown(
                f"{sp.icon} **{sp.label}** — `{s.text}`  \n"
                f"<span style='color:gray'>score {s.score:.2f} · {s.detector} · {s.reason}</span>",
                unsafe_allow_html=True,
            )
        if not report.spans:
            st.info("No PII detected.")
        if not report.ner_available:
            st.caption("ℹ️ AI NER tier not installed here — checksum-validated regex ran solo. Still 100% local.")

    with tab2:
        if run_redact:
            result = redact_text(input_text, report, vault_obj if vault_pass else None, types)
            st.session_state["last_result"] = result
        if "last_result" in st.session_state:
            result = st.session_state["last_result"]
            st.code(result.redacted_text or "—", language="text")
            c1, c2 = st.columns(2)
            c1.metric("Redacted", result.redacted_count)
            c2.metric("Vaulted (reversible)", result.vaulted_count)
            if result.token_map and vault_pass:
                st.caption("Reveal any token with its passphrase — original never left this machine.")
                for token in list(result.token_map)[:12]:
                    st.markdown(f"`{token}` → {vault_obj.entry_hint(token)}")
            elif result.token_map:
                st.info("Vault locked — redactions were irreversible. Enter a passphrase in the sidebar "
                        "and redact again for reversible tokens.")

    with tab3:
        st.markdown('<div class="sec-title">Engine internals</div>', unsafe_allow_html=True)
        st.json({
            "regex_ms": round(report.regex_ms, 2),
            "ner_ms": round(report.ner_ms, 2),
            "ner_available": report.ner_available,
            "ner_error": report.ner_error,
            "chars": report.text_length,
            "vault_path": str(vault_obj.path),
            "entity_types_selected": None if types is None else [t.value for t in types],
        })

st.markdown(
    '<div class="foot"><b>Redacta AI</b> · Snapdragon® AI Lab Build &amp; Present Challenge · '
    "every byte processed on this device — verify with any packet sniffer.</div>",
    unsafe_allow_html=True,
)
