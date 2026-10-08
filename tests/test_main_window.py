from PySide6.QtCore import Qt

from app.ui.main_window import MainWindow


def _make(qtbot, translator, settings):
    window = MainWindow(translator, settings)
    qtbot.addWidget(window)
    window.show()
    return window


def test_window_starts_with_defaults(qtbot, translator, settings):
    w = _make(qtbot, translator, settings)
    assert w.isVisible()
    assert w.windowTitle() == "AI Subtitle Studio"
    assert w.source_combo.currentData() == "auto"
    assert w.target_combo.currentData() == "ar"
    assert w.mode_combo.currentData() == "balanced"
    assert w.mode_combo.count() == 3


def test_language_switch_rtl_round_trip(qtbot, translator, settings):
    w = _make(qtbot, translator, settings)
    en_text = w.start_button.text()
    w._language_actions["ar"].trigger()
    assert translator.language == "ar"
    assert w.layoutDirection() == Qt.LayoutDirection.RightToLeft
    assert w.start_button.text() != en_text
    assert w._language_actions["ar"].isChecked()
    # Paths and URLs stay left-to-right in the RTL UI.
    assert w.input_edit.layoutDirection() == Qt.LayoutDirection.LeftToRight
    assert settings.get("ui_language") == "ar"

    w._language_actions["en"].trigger()
    assert w.layoutDirection() == Qt.LayoutDirection.LeftToRight
    assert w.start_button.text() == en_text
    assert settings.get("ui_language") == "en"


def test_retranslate_preserves_selection(qtbot, translator, settings):
    w = _make(qtbot, translator, settings)
    w.source_combo.setCurrentIndex(w.source_combo.findData("tr"))
    w.mode_combo.setCurrentIndex(w.mode_combo.findData("fast"))
    translator.set_language("ar")
    assert w.source_combo.currentData() == "tr"
    assert w.mode_combo.currentData() == "fast"
    assert settings.get("source_language") == "tr"
    assert settings.get("mode") == "fast"


def test_selections_restored_from_settings(qtbot, translator, settings):
    settings.set("target_language", "en")
    settings.set("mode", "maximum_accuracy")
    settings.set("output_dir", "/out")
    w = _make(qtbot, translator, settings)
    assert w.target_combo.currentData() == "en"
    assert w.mode_combo.currentData() == "maximum_accuracy"
    assert w.output_edit.text() == "/out"


def test_validation_messages(qtbot, translator, settings, tmp_path):
    w = _make(qtbot, translator, settings)
    assert w.validate_inputs() == translator.t("error.input_required")
    w.input_edit.setText(str(tmp_path / "missing.mp4"))
    assert "missing.mp4" in w.validate_inputs()
    video = tmp_path / "ep.mp4"
    video.write_bytes(b"")
    w.input_edit.setText(str(video))
    assert w.validate_inputs() is None
    w.input_edit.setText("https://www.youtube.com/watch?v=x")
    w.source_combo.setCurrentIndex(w.source_combo.findData("ar"))
    assert w.validate_inputs() == translator.t("error.same_language")


def test_start_reports_without_blocking(qtbot, translator, settings):
    w = _make(qtbot, translator, settings)
    w.start_button.click()
    assert translator.t("error.input_required") in w.log_view.toPlainText()


def test_start_label_follows_input_type(qtbot, translator, settings):
    w = _make(qtbot, translator, settings)
    w.input_edit.setText("C:/videos/ep1.mp4")
    assert w.start_button.text() == translator.t("button.start")
    w.input_edit.setText("https://www.youtube.com/watch?v=x")
    assert w.start_button.text() == translator.t("button.start_url")


def test_settings_dialog_saves(qtbot, translator, settings):
    from app.ui.settings_window import ProviderSettingsDialog

    dialog = ProviderSettingsDialog(translator, settings)
    qtbot.addWidget(dialog)
    dialog.review_check.setChecked(True)
    dialog.refine_check.setChecked(True)
    dialog.engine_combo.setCurrentIndex(dialog.engine_combo.findData("madlad"))
    dialog.local_model_edit.setText("gemma3:4b")
    dialog.subdl_edit.setText(" key ")
    # Site access: only the chosen source is editable.
    assert not dialog.cookies_browser_combo.isEnabled() and not dialog.cookies_file_edit.isEnabled()
    dialog.cookies_combo.setCurrentIndex(dialog.cookies_combo.findData("browser"))
    assert dialog.cookies_browser_combo.isEnabled() and not dialog.cookies_file_edit.isEnabled()
    dialog.cookies_browser_combo.setCurrentIndex(dialog.cookies_browser_combo.findData("firefox"))
    dialog.ipv4_check.setChecked(True)
    dialog.save()
    assert settings.get("llm_review") is True and settings.get("llm_refine") is True
    assert settings.get("translation_engine") == "madlad" and settings.get("local_model") == "gemma3:4b"
    assert settings.get("subdl_api_key") == "key"
    assert settings.get("cookies_source") == "browser"
    assert settings.get("cookies_browser") == "firefox"
    assert settings.get("force_ipv4") is True


def test_theme_switch(qtbot, translator, settings):
    from app.ui.main_window import MainWindow

    w = MainWindow(translator, settings)
    qtbot.addWidget(w)
    assert settings.get("theme") == "system"
    w.theme_button.click()
    assert settings.get("theme") == "light" and w._theme_actions["light"].isChecked()
    w.set_theme("dark")
    from PySide6.QtWidgets import QApplication
    assert QApplication.instance().property("resolvedTheme") == "dark"
    assert w.start_button.objectName() == "primaryButton"
