"""Detecting series/season/episode from the input, and what the window does with it (D-077). No network."""

import pytest
from PySide6.QtWidgets import QLineEdit

from app.core.metadata import SeriesInfo


def test_file_name_is_parsed_immediately(qtbot, tmp_path):
    from app.ui.link_probe import LinkProbe

    media = tmp_path / "Kurtlar Vadisi Pusu 8. Sezon 198. Bolum.mp4"
    media.write_bytes(b"x")
    edit = QLineEdit()
    probe = LinkProbe(edit, delay_ms=0)
    seen = []
    probe.resolved.connect(lambda info, origin: seen.append((info, origin)))

    edit.setText(str(media))
    probe.start_now()

    assert seen, "nothing was reported for a file name"
    info, origin = seen[-1]
    assert origin == "filename"
    assert (info.series_name, info.season, info.episode) == ("Kurtlar Vadisi Pusu", 8, 198)


def test_url_is_probed_on_a_worker_with_a_fake_adapter(qtbot):
    from app.ui.link_probe import LinkProbe

    class FakeAdapter:
        def probe(self, url):
            return {"title": "Kurtlar Vadisi Pusu 198. Bölüm", "id": "x"}

    edit = QLineEdit()
    probe = LinkProbe(edit, adapter_factory=FakeAdapter, delay_ms=0)
    seen = []
    probe.resolved.connect(lambda info, origin: seen.append((info, origin)))

    edit.setText("https://www.youtube.com/watch?v=x")
    probe.start_now()
    probe.wait_for_lookup()
    qtbot.waitUntil(lambda: bool(seen), timeout=10000)

    info, origin = seen[-1]
    assert origin == "url"
    assert info.series_name == "Kurtlar Vadisi Pusu" and info.episode == 198


def test_a_failed_probe_is_silent(qtbot):
    from app.ui.link_probe import LinkProbe

    class BrokenAdapter:
        def probe(self, url):
            raise RuntimeError("network down")

    edit = QLineEdit()
    probe = LinkProbe(edit, adapter_factory=BrokenAdapter, delay_ms=0)
    seen = []
    probe.resolved.connect(lambda info, origin: seen.append((info, origin)))

    edit.setText("https://example.invalid/watch?v=x")
    probe.start_now()
    probe.wait_for_lookup()
    qtbot.wait(50)

    assert seen and seen[-1][0] == SeriesInfo()      # an empty result, never an exception


def test_window_fills_only_empty_fields(qtbot, translator, settings, db, tmp_path):
    from app.ui.main_window import MainWindow, RuntimeContext

    runtime = RuntimeContext(db_path=db.path, pipeline_factory=lambda *args, **kwargs: None,
                             cache_dir=tmp_path / "cache")
    window = MainWindow(translator, settings, runtime)
    qtbot.addWidget(window)

    window.series_edit.setText("My Own Name")        # the user already typed a name
    window._apply_detected_series(SeriesInfo(series_name="Kurtlar Vadisi Pusu", season=8, episode=198,
                                             source="metadata"), "url")
    assert window.series_edit.text() == "My Own Name"       # never overwritten
    assert window.season_spin.value() == 8 and window.episode_spin.value() == 198

    window.series_edit.clear()
    window._apply_detected_series(SeriesInfo(series_name="Kurtlar Vadisi Pusu"), "url")
    assert window.series_edit.text() == "Kurtlar Vadisi Pusu"
    window.close()


def test_window_ignores_an_empty_detection(qtbot, translator, settings, db, tmp_path):
    from app.ui.main_window import MainWindow, RuntimeContext

    runtime = RuntimeContext(db_path=db.path, pipeline_factory=lambda *args, **kwargs: None,
                             cache_dir=tmp_path / "cache")
    window = MainWindow(translator, settings, runtime)
    qtbot.addWidget(window)

    window._apply_detected_series(SeriesInfo(), "url")

    assert window.series_edit.text() == "" and window.season_spin.value() == 0
    window.close()
