"""Shared fixtures: one throwaway database and policy index per test session."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import db as db_module  # noqa: E402
from app.config import STORE_DB_PATH  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def built_database():
    """Build store.db once; the whole suite reads the same fixture."""
    if not STORE_DB_PATH.exists():
        db_module.rebuild_database()
    return STORE_DB_PATH
