import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_app_starts_offscreen_and_exits(tmp_path):
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", AISS_DATA_DIR=str(tmp_path))
    result = subprocess.run(
        [sys.executable, "-m", "app.main", "--smoke-test"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "studio.db").is_file()
    log_text = (tmp_path / "logs" / "app.log").read_text(encoding="utf-8")
    assert "Exiting with code 0" in log_text


def test_self_test_reports_ok(tmp_path):
    from app.main import main

    report = tmp_path / "r.txt"
    assert main(["--self-test", str(report)]) == 0
    assert report.read_text(encoding="utf-8").strip().endswith("self-test ok")
