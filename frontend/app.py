"""Redacta AI — on-device PII redaction studio (Streamlit UI).

Run:  streamlit run frontend/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from core.entities import ENTITY_SPECS, EntityType


def _ner_installed() -> bool:
    """Cheap check (no model download) — is the GLiNER package present?"""
    import importlib.util

    return importlib.util.find_spec("gliner") is not None
from core.pipeline import RedactionPipeline
from core.redact import redact_text
from core.vault import Vault
from utils.samples import document

st.set_page_config(page_title="Redacta AI", page_icon="🛡️", layout="wide")

# --- session state -----------------------------------------------------

if "pipeline" not in st.session_state:
    st.session_state.pipeline = RedactionPipeline(use_ner=True)
if "vault" not in st.session_state:
    st.session_state.vault = Vault()


def _types_from_selection(selected: list[str]) -> list[EntityType] | None:
    if not selected or len(selected) == len(ENTITY_SPECS):
        return None
    return [EntityType(t) for t in selected]


# --- sidebar -----------------------------------------------------------

with st.sidebar:
    st.title("🛡️ Redacta AI")
    st.caption("On-device PII detection & redaction — zero cloud.")

    use_ner = st.toggle("AI model (GLiNER NER)", value=True,
                        help="Adds names/locations/orgs. Falls back to regex-only when unavailable.")
    if _ner_installed():
        st.caption("✅ GLiNER package found — NER tier active.")
    else:
        st.caption("ℹ️ GLiNER not installed — regex-only mode (checksum-validated, still 100% local).")
    if use_ner != st.session_state.pipeline.ner is not None:
        st.session_state.pipeline = RedactionPipeline(use_ner=use_ner)

    st.divider()
    st.subheader("Vault (reversible redaction)")
    vault = st.session_state.vault
    vault_pass = st.text_input("Vault passphrase", type="password",
                               help="Decrypts redactions locally — never leaves this machine.")
    if st.button("Create / unlock vault", use_container_width=True):
        if not vault_pass:
            st.warning("Enter a passphrase first.")
        else:
            try:
                vault.load(vault_pass)
                st.success("Vault unlocked.")
            except Exception:
                vault.create(vault_pass)
                st.success("New vault created.")

    st.divider()
    st.subheader("Entity types")
    defaults = [t.value for t in EntityType]
    selected = st.multiselect("Detect", defaults, default=defaults, label_visibility="collapsed")

    st.divider()
    if st.button("🎲 Generate fake sample doc", use_container_width=True):
        st.session_state["demo_text"] = document()


# --- main --------------------------------------------------------------

st.markdown("### Paste text, or drop a document 👇")
input_text = st.text_area(
    "Document", value=st.session_state.get("demo_text", ""),
    height=220, label_visibility="collapsed",
    placeholder="Paste a bank statement, screenshot text, ID copy… (all processing is local)",
)

col1, col2, _ = st.columns([1, 1, 2])
run_scan = col1.button("🔍 Scan", type="primary", use_container_width=True)
run_redact = col2.button("🧹 Redact", use_container_width=True)

if run_scan or run_redact:
    if not input_text.strip():
        st.warning("Nothing to analyse yet.")
        st.stop()

    types = _types_from_selection(selected)
    with st.spinner("Analysing on-device…"):
        report = st.session_state.pipeline.analyze(input_text, types)

    tabs = st.tabs(["📊 Findings", "🧹 Redacted output", "⚙️ Engine"])
    with tabs[0]:
        st.success(f"**{len(report.spans)} entities** found in {report.total_ms:.1f} ms "
                   f"(regex {report.regex_ms:.1f} ms · NER {report.ner_ms:.1f} ms"
                   f"{' — regex-only mode' if not report.ner_available else ''})")
        for s in report.spans:
            sp = ENTITY_SPECS[s.entity_type]
            st.markdown(
                f"{sp.icon} **{sp.label}** — `{s.text}`  \n"
                f"<span style='color:gray'>score {s.score:.2f} · {s.detector} · {s.reason}</span>",
                unsafe_allow_html=True,
            )
        if not report.spans:
            st.info("No PII detected.")

    with tabs[1]:
        if run_redact:
            vault_obj = st.session_state.vault if vault_pass else None
            result = redact_text(input_text, report, vault_obj, types)
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
                    st.markdown(f"`{token}` → {st.session_state.vault.entry_hint(token)}")

    with tabs[2]:
        st.json({
            "regex_ms": round(report.regex_ms, 2),
            "ner_ms": round(report.ner_ms, 2),
            "ner_available": report.ner_available,
            "ner_error": report.ner_error,
            "chars": report.text_length,
            "vault_path": str(st.session_state.vault.path),
        })

st.divider()
st.caption("Redacta AI — Snapdragon® AI Lab Build & Present Challenge · everything runs on this device.")
