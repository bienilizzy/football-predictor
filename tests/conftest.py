"""Pytest configuration.

Routes the app at an isolated, file-based SQLite test database. The env var
must be set *before* `config.settings`/`db.session` are first imported by any
test module, so this happens at conftest module-import time (pytest imports
conftest.py before collecting sibling test files).
"""
from __future__ import annotations

import os
import pathlib

_TEST_DB_PATH = pathlib.Path(__file__).parent / "_test.db"

# Use a path relative to the current working directory (pytest's invocation
# directory, normally the project root). An absolute `file:` URI built from
# `\\wsl.localhost\...` would be parsed as `//wsl.localhost/...`, and SQLite
# rejects "wsl.localhost" as a URI authority/host.
_relative_db_path = pathlib.Path(os.path.relpath(_TEST_DB_PATH, pathlib.Path.cwd())).as_posix()
os.environ["DATABASE_URL"] = f"sqlite:///file:./{_relative_db_path}?uri=true&nolock=1"

import pytest

from football_predictor.db.init_db import init_db
from football_predictor.db.session import engine


@pytest.fixture(scope="session", autouse=True)
def _test_database():
    if _TEST_DB_PATH.exists():
        _TEST_DB_PATH.unlink()
    init_db()
    yield
    # Close pooled connections first; Windows won't let us delete a file that
    # the engine still has open handles to.
    engine.dispose()
    if _TEST_DB_PATH.exists():
        _TEST_DB_PATH.unlink()
