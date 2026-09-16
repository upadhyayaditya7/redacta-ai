"""Shared configuration paths for Redacta AI."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path.home() / ".redacta"
VAULT_PATH = DATA_DIR / "vault.json"
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"
BENCH_DIR = PROJECT_ROOT / "benchmarks"
