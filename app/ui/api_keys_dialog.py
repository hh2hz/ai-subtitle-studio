"""The API keys window: one field per cloud AI provider, opened on the first start and from Settings (D-117).

Every provider is optional. A provider left empty is skipped by the translation (it stays in the program, so it can
be added later). Keys are written to the key file only - never to the settings database and never to a log.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox, QGridLayout, QLabel, QLineEdit, QMessageBox,
                               QScrollArea, QVBoxLayout, QWidget)

from app.core.llm_providers import DEFAULT_ORDER, PROVIDERS
from app.services import api_keys

log = logging.getLogger(__name__)


def provider_names() -> list[str]:
    """Every provider the program knows, in the order of the translation preference."""
    return [name for name in DEFAULT_ORDER if name in PROVIDERS] + [n for n in PROVIDERS if n not in DEFAULT_ORDER]


class ApiKeysDialog(QDialog):
    """Edit the keys of all providers. `saved` is True after the user pressed Save and the file was written."""

    def __init__(self, translator, parent=None, first_run: bool = False, path=None):
        super().__init__(parent)
        self._tr = translator
        self._path = path
        self.saved = False
        self.setModal(True)
        self.setWindowTitle(translator.t("keys.title"))
        self.setMinimumSize(640, 520)

        layout = QVBoxLayout(self)
        intro = QLabel(translator.t("keys.intro_first" if first_run else "keys.intro"))
        intro.setWordWrap(True)
        layout.addWidget(intro)

        current = api_keys.load_keys([path] if path else None)
        self._fields: dict[str, QLineEdit] = {}
        self._accounts: dict[str, QLineEdit] = {}
        self._keyless: dict[str, QCheckBox] = {}

        inner = QWidget()
        grid = QGridLayout(inner)
        grid.setColumnStretch(1, 1)
        row = 0
        for name in provider_names():
            spec = PROVIDERS[name]
            entry = current.get(name, {})
            grid.addWidget(QLabel(f"<b>{name}</b>"), row, 0, Qt.AlignmentFlag.AlignTop)
            if spec.keyless:
                box = QCheckBox(translator.t("keys.keyless"))
                box.setChecked(not current or name in current)      # on by default; off only if the user turned it off
                self._keyless[name] = box
                grid.addWidget(box, row, 1)
            else:
                edit = QLineEdit(str(entry.get("api_key", "")))
                edit.setEchoMode(QLineEdit.EchoMode.Password)
                edit.setPlaceholderText(translator.t("keys.placeholder"))
                edit.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
                edit.setClearButtonEnabled(True)
                self._fields[name] = edit
                grid.addWidget(edit, row, 1)
                url = api_keys.SIGNUP_URLS.get(name, "")
                if url:
                    link = QLabel(f'<a href="{url}">{translator.t("keys.get_key")}</a>')
                    link.setOpenExternalLinks(True)
                    link.setToolTip(url)
                    grid.addWidget(link, row, 2)
                if name == "cloudflare":
                    row += 1
                    account = QLineEdit(str(entry.get("account_id", "")))
                    account.setPlaceholderText(translator.t("keys.account_id"))
                    account.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
                    self._accounts[name] = account
                    grid.addWidget(account, row, 1)
            row += 1
        grid.setRowStretch(row, 1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        layout.addWidget(scroll, 1)

        self.show_check = QCheckBox(translator.t("keys.show"))
        self.show_check.toggled.connect(self._toggle_visible)
        layout.addWidget(self.show_check)
        note = QLabel(translator.t("keys.note", path=str(path or api_keys.target_file())))
        note.setWordWrap(True)
        note.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(note)

        buttons = QDialogButtonBox()
        self.save_button = buttons.addButton(translator.t("keys.save"), QDialogButtonBox.ButtonRole.AcceptRole)
        self.skip_button = buttons.addButton(translator.t("keys.skip" if first_run else "keys.cancel"),
                                             QDialogButtonBox.ButtonRole.RejectRole)
        self.save_button.clicked.connect(self._save)
        self.skip_button.clicked.connect(self.reject)
        layout.addWidget(buttons)

    def _toggle_visible(self, visible: bool) -> None:
        mode = QLineEdit.EchoMode.Normal if visible else QLineEdit.EchoMode.Password
        for edit in self._fields.values():
            edit.setEchoMode(mode)

    def entries(self) -> dict[str, dict]:
        """What the dialog holds now: provider -> {"api_key", "account_id"}. Empty keys are included (= skipped)."""
        result: dict[str, dict] = {}
        for name, edit in self._fields.items():
            entry = {"api_key": edit.text().strip()}
            if name in self._accounts:
                entry["account_id"] = self._accounts[name].text().strip()
            result[name] = entry
        for name, box in self._keyless.items():
            result[name] = {"api_key": api_keys.KEYLESS if box.isChecked() else ""}
        return result

    def _save(self) -> None:
        entries = self.entries()
        missing = [n for n in self._accounts if entries[n]["api_key"] and not entries[n].get("account_id")]
        if missing:
            QMessageBox.warning(self, self._tr.t("keys.title"), self._tr.t("keys.need_account", provider=missing[0]))
            return
        try:
            api_keys.save_keys(entries, self._path)
        except OSError as exc:
            log.warning("Cannot write the API key file: %s", exc)
            QMessageBox.critical(self, self._tr.t("keys.title"),
                                 self._tr.t("keys.save_failed", path=str(self._path or api_keys.target_file()),
                                            error=str(exc)))
            return
        self.saved = True
        self.accept()


def maybe_prompt_first_run(translator, settings, parent=None, ask=None) -> bool:
    """On the very first start, ask for the keys once. Returns True when the window was shown.

    Nothing is asked when keys already exist (an upgrade, or a file the user prepared). The question is never
    asked again afterwards, whichever button was pressed; Settings -> API keys opens the same window any time.
    `ask` replaces the window in tests.
    """
    if settings.get("api_keys_prompted"):
        return False
    shown = False
    if not api_keys.load_keys():
        dialog = ApiKeysDialog(translator, parent, first_run=True) if ask is None else None
        if ask is not None:
            ask()
        else:
            dialog.exec()
        shown = True
    settings.set("api_keys_prompted", True)
    return shown
