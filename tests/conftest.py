"""Shared fixtures. Qt tests always run headless."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

from app.database.database import Database  # noqa: E402
from app.database.settings import Settings  # noqa: E402
from app.utils.i18n import Translator  # noqa: E402
from app.utils.paths import translations_dir  # noqa: E402


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "test.db")
    yield database
    database.close()


@pytest.fixture
def settings(db):
    return Settings(db)


@pytest.fixture
def translator(qapp):
    tr = Translator(translations_dir())
    tr.set_language("en")
    yield tr
    tr.set_language("en")
