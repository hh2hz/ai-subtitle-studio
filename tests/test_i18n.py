import json
import re
import string
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from app.core.modes import Mode
from app.utils import languages
from app.utils.i18n import TranslationError, Translator
from app.utils.paths import translations_dir

APP_DIR = Path(__file__).resolve().parents[1] / "app"
KEY_USE = re.compile(r"""\bt\(\s*["']([A-Za-z0-9_.]+)["']""")


def _fields(text):
    return {name for _, name, _, _ in string.Formatter().parse(text) if name}


def test_catalogues_have_identical_keys(translator):
    codes = translator.available_languages()
    assert codes[0] == "en" and "ar" in codes
    base = translator.keys("en")
    for code in codes:
        assert translator.keys(code) == base, code


def test_no_empty_strings_and_placeholders_match(translator):
    for code in translator.available_languages():
        for key in translator.keys("en"):
            text = translator.raw(code, key)
            assert text.strip(), (code, key)
            assert _fields(text) == _fields(translator.raw("en", key)), (code, key)


def test_all_keys_used_in_code_exist(translator):
    used = set()
    for path in APP_DIR.rglob("*.py"):
        used |= set(KEY_USE.findall(path.read_text(encoding="utf-8")))
    assert used, "key scan found nothing; regex is broken"
    missing = {k for k in used if not translator.has(k)}
    assert not missing


def test_dynamic_keys_exist(translator):
    for mode in Mode:
        assert translator.has(f"mode.{mode.value}")
        assert translator.has(f"mode.{mode.value}.tooltip")
    for code in languages.SUBTITLE_LANGUAGES:
        assert translator.has(f"lang.{code}")
    from app.core.pipeline import STAGES
    for stage in STAGES:
        assert translator.has(f"stage.{stage}")


def test_every_emitted_flag_has_a_string(translator):
    """D-106: no flag code may reach the review window without `flag.<code>` in both catalogues."""
    from app.core.confidence import ALL_FLAGS

    missing = sorted(code for code in ALL_FLAGS if not translator.has(f"flag.{code}"))
    assert missing == []
    # The catalogues must not carry flag strings for codes nobody emits either: the two sets must line up.
    catalogue = {key.split(".", 1)[1] for key in translator.keys("en") if key.startswith("flag.")}
    assert catalogue == set(ALL_FLAGS)


def _ui_error_keys():
    """Every literal ui_key handed to PipelineError/ProviderError in app/ (D-106)."""
    import ast

    classes = {"PipelineError", "ProviderError", "ProviderUnavailable", "ProviderAuthError"}
    found = {}
    for path in APP_DIR.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name not in classes:
                continue
            literal = None
            if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                literal = node.args[1].value
            for keyword in node.keywords:
                if keyword.arg == "ui_key" and isinstance(keyword.value, ast.Constant):
                    literal = keyword.value.value
            if isinstance(literal, str) and literal:
                found.setdefault(literal, []).append(f"{path.name}:{node.lineno}")
    return found


def test_error_keys_used_in_code_exist(translator):
    """A wrong ui_key shows the raw key to the user, so every literal error key must exist."""
    found = _ui_error_keys()
    assert found, "the ui_key scan found nothing; the scan is broken"
    missing = {key: where for key, where in found.items() if not translator.has(key)}
    assert missing == {}


def test_dynamic_keys_exist(translator):
    for mode in Mode:
        assert translator.has(f"mode.{mode.value}")
        assert translator.has(f"mode.{mode.value}.tooltip")
    for code in languages.SUBTITLE_LANGUAGES:
        assert translator.has(f"lang.{code}")
    from app.core.pipeline import STAGES
    for stage in STAGES:
        assert translator.has(f"stage.{stage}")


def test_switch_sets_rtl_and_back(translator, qtbot):
    app = QApplication.instance()
    with qtbot.waitSignal(translator.language_changed) as blocker:
        translator.set_language("ar")
    assert blocker.args == ["ar"]
    assert app.layoutDirection() == Qt.LayoutDirection.RightToLeft
    assert translator.t("button.start") != translator.raw("en", "button.start")
    translator.set_language("en")
    assert app.layoutDirection() == Qt.LayoutDirection.LeftToRight
    assert translator.t("button.start") == "Translate"


def test_unknown_language_rejected(translator):
    with pytest.raises(ValueError):
        translator.set_language("xx")


def test_fallbacks_and_formatting(tmp_path, qapp):
    (tmp_path / "en.json").write_text(json.dumps({
        "_meta": {"native_name": "English", "direction": "ltr"},
        "strings": {"a": "A {n}", "b": "B"}}), encoding="utf-8")
    (tmp_path / "xx.json").write_text(json.dumps({
        "_meta": {"native_name": "Xx", "direction": "ltr"},
        "strings": {"a": "XA {n}"}}), encoding="utf-8")
    tr = Translator(tmp_path)
    tr.set_language("xx")
    assert tr.t("a", n=1) == "XA 1"
    assert tr.t("b") == "B"                 # falls back to English
    assert tr.t("zzz") == "zzz"             # falls back to the key
    assert tr.t("a", wrong=1) == "XA {n}"   # bad format args do not raise


def test_invalid_catalogue_rejected(tmp_path):
    (tmp_path / "en.json").write_text('{"strings": {}}', encoding="utf-8")
    with pytest.raises(TranslationError):
        Translator(tmp_path)


def test_missing_fallback_rejected(tmp_path):
    with pytest.raises(TranslationError):
        Translator(tmp_path)


def test_real_catalogues_are_valid_json():
    for path in translations_dir().glob("*.json"):
        json.loads(path.read_text(encoding="utf-8"))
