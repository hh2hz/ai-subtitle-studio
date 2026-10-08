"""Review session and window (M4 acceptance: the review UI loads flagged lines with reasons)."""

import pytest

from app.core.exporter import read_srt
from app.core.modes import Mode
from app.core.pipeline import JobConfig, Pipeline
from app.core.review import ReviewError, ReviewSession
from app.ui.review_window import ReviewWindow
from tests.fakes import CPU_ASR, CPU_MT, Factory, FakeAsrEngine, FakeTranslator
from tests.media import make_tone_file, make_video_only_file

ARABIC_LINE = "\u0647\u0644 \u0623\u0646\u062a \u0645\u062a\u0623\u0643\u062f\u061f"


class ArabicTranslator(FakeTranslator):
    """Arabic output for every line except a Latin one on line 3, so some lines are flagged and some are not."""

    def translate(self, texts, source_language, target_language, context=None):
        super().translate(texts, source_language, target_language, context)
        return [ARABIC_LINE if "3" not in t else "untranslated text" for t in texts]


@pytest.fixture
def episode(tmp_path):
    media = make_tone_file(tmp_path / "Ep.m4a", seconds=20.0)
    config = JobConfig(media, "tr", "ar", Mode.BALANCED, output_dir=tmp_path / "out")
    result = Pipeline(config, tmp_path / "jobs", [CPU_ASR], [CPU_MT],
                      Factory({(CPU_ASR.model, "cpu"): FakeAsrEngine()}),
                      Factory({(CPU_MT.model, "cpu"): ArabicTranslator()})).run()
    return result.output_dir


def test_session_flags_edit_approve_save_reload(episode):
    session = ReviewSession(episode)
    flagged = [u for u in session.units if session.needs_review(u)]
    assert flagged and len(flagged) < len(session.units)
    line3 = session.unit(3)
    assert session.needs_review(line3) and "untranslated_or_wrong_script" in line3["flags"]
    session.set_text(3, ARABIC_LINE.replace("\u061f", " 5?"))
    assert session.unit(3)["final_text"].endswith("5\u061f")          # language fixes applied
    session.approve(3)
    assert not session.needs_review(session.unit(3))
    srt = session.save()
    texts = [c.text for c in read_srt(srt)]
    assert any(t.rstrip("\u200f").endswith("5\u061f") for t in texts)
    reloaded = ReviewSession(episode)
    assert reloaded.unit(3)["approved"] and reloaded.unit(3)["reviewed_text"]
    assert reloaded.counts()["approved"] == 1 and reloaded.counts()["edited"] == 1
    assert "#4 " not in (episode / "work" / "ReviewRequired.txt").read_text(encoding="utf-8")


def test_revert_to_ai_text(episode):
    session = ReviewSession(episode)
    original = session.unit(0)["translation"]
    session.set_text(0, "x")
    session.set_text(0, original)
    assert "reviewed_text" not in session.unit(0)


def test_not_an_episode_folder(tmp_path):
    with pytest.raises(ReviewError):
        ReviewSession(tmp_path)


@pytest.mark.parametrize("language", ["en", "ar"])
def test_window_loads_flagged_lines_with_reasons(qtbot, translator, episode, language):
    translator.set_language(language)          # Arabic UI = right-to-left layout
    window = ReviewWindow(episode, translator, enable_media=False)
    qtbot.addWidget(window)
    assert window.table.rowCount() >= 1
    ids = window._rows
    assert 3 in ids                                              # only lines needing review by default
    window.select_row(ids.index(3))
    assert translator.t("flag.untranslated_or_wrong_script") in window.reasons_label.text()
    assert window.table.item(ids.index(3), 5).text()
    window.text_edit.setPlainText(ARABIC_LINE)
    window.approve_and_next()
    assert window.session.unit(3)["approved"] and window.session.dirty
    path = window.save()
    assert not window.session.dirty and ARABIC_LINE in path.read_text(encoding="utf-8-sig")
    window.filter_combo.setCurrentIndex(1)
    assert window.table.rowCount() == len(window.session.units)


def test_window_with_video_file(qtbot, translator, episode):
    make_video_only_file(episode / "Ep.mp4")
    (episode / "Ep.m4a").unlink(missing_ok=True)
    window = ReviewWindow(episode, translator)
    qtbot.addWidget(window)
    assert window.session.video_path.name == "Ep.mp4"
    window.select_row(0)
    window.play_line()                        # must not raise even without an audio device
    window._on_position(int(window.session.units[0]["start"] * 1000) + 10)
    window.session.dirty = False


def test_retranslate_line_with_ai(qtbot, translator, episode):
    from tests.fake_llm_api import FakeLlmApi
    from tests.test_pipeline import _refiner_factory

    with FakeLlmApi() as server:
        window = ReviewWindow(episode, translator, _refiner_factory(server, review=False), enable_media=False)
        qtbot.addWidget(window)
        window.select_row(window._rows.index(3))
        window.retranslate_line()
        qtbot.waitUntil(lambda: window._thread is None, timeout=15000)
    assert window.text_edit.toPlainText().startswith("AR(")
    window.save()            # the new text is committed and saved, so closing asks nothing


def test_burn_ass_and_files(tmp_path):
    from app.core import burn
    from app.core.exporter import Cue, mark_rtl, write_srt

    episode = tmp_path / "Ep"
    episode.mkdir()
    (episode / "Ep 1.mp4").write_bytes(b"v")
    write_srt([Cue(1.0, 2.5, mark_rtl("a {x}\nb"))], episode / "Ep 1.ar.srt")
    video, srt = burn.episode_files(episode)
    assert video.name == "Ep 1.mp4" and srt.name == "Ep 1.ar.srt"
    assert burn.output_path(srt).name == "Ep 1.ar.subtitled.mp4"
    (episode / "Ep 1.ar.subtitled.mp4").write_bytes(b"v")
    assert burn.episode_files(episode)[0].name == "Ep 1.mp4"          # never the burned copy
    ass = tmp_path / "x.ass"
    burn.write_ass(burn.read_srt(srt), ass, 1920, 1080)
    text = ass.read_text(encoding="utf-8-sig")
    assert "PlayResY: 1080" in text and "Style: Default,Arial,59," in text and ",2," in text
    assert "Dialogue: 0,0:00:01.00,0:00:02.50,Default,,0,0,0,,‏a (x)‏\\N‏b‏" in text


def test_burn_with_real_ffmpeg(tmp_path):
    import shutil
    import subprocess

    import pytest

    from app.core import burn
    from app.core.exporter import Cue, write_srt

    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    try:
        burn.find_ffmpeg()
    except Exception:
        pytest.skip("ffmpeg without libass")
    video = tmp_path / "Ep.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=s=320x240:d=2", "-f", "lavfi",
                    "-i", "sine=d=2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(video)], check=True)
    write_srt([Cue(0.2, 1.5, "hello")], tmp_path / "Ep.ar.srt")
    seen = []
    out = burn.burn(video, tmp_path / "Ep.ar.srt", progress=seen.append, font="DejaVu Sans")
    assert out.is_file() and out.stat().st_size > 1000 and seen[-1] == 1.0
    w, h, duration = burn.video_info(out)
    assert (w, h) == (320, 240) and duration > 1.5


def test_the_window_asks_for_a_preview_copy_only_once(qtbot, translator, episode, tmp_path, monkeypatch):
    from app.ui import review_window

    window = review_window.ReviewWindow(episode, translator, enable_media=False)
    qtbot.addWidget(window)
    started = []
    monkeypatch.setattr(review_window.PreviewCopyRunner, "start", lambda self: started.append(self))
    window.player = type("P", (), {"stop": lambda self: None})()      # a player exists, so a copy can be requested
    window.session.video_path = tmp_path / "a.mkv"

    window._on_player_error(1, "format error")
    window._on_player_error(1, "format error")

    assert len(started) == 1 and window.overlay.text()
