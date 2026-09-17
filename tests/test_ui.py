"""Smoke-test the Streamlit studio: boot, generate sample, scan, redact.

Uses Streamlit's official testing framework (streamlit.testing.v1), so the
whole UI is exercised headlessly — no browser, no server.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamlit.testing.v1 import AppTest


def test_studio_end_to_end() -> None:
    at = AppTest.from_file("frontend/app.py", default_timeout=30)
    at.run()
    assert not at.exception, f"App raised on boot: {at.exception}"

    # Generate a fake sample doc via the sidebar button.
    gen = [b for b in at.sidebar.button if "Sample" in b.label]
    assert gen, "sample-doc button missing from sidebar"
    gen[0].click()
    at.run()
    assert not at.exception, f"App raised on generate: {at.exception}"
    assert at.text_area[0].value.strip(), "sample doc did not populate the text area"

    # Scan.
    scan = [b for b in at.button if "Scan" in b.label]
    assert scan, "Scan button missing"
    scan[0].click()
    at.run()
    assert not at.exception, f"App raised on scan: {at.exception}"

    # Redact (no passphrase -> irreversible masking path).
    red = [b for b in at.button if "Redact" in b.label]
    assert red, "Redact button missing"
    red[0].click()
    at.run()
    assert not at.exception, f"App raised on redact: {at.exception}"

    # The sanitised output must exist and differ from the input.
    result = at.session_state["last_result"]
    assert result.redacted_count > 0, "nothing was redacted"
    assert result.redacted_text.strip()
    assert result.redacted_text != at.text_area[0].value, "output identical to input"
