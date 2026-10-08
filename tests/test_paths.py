from app.utils import paths


def test_env_override_and_ensure(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_DATA_DIR, str(tmp_path / "data"))
    root = paths.default_data_root()
    assert root == tmp_path / "data"
    app_paths = paths.AppPaths.from_root(root).ensure()
    for d in (app_paths.root, app_paths.logs_dir, app_paths.models_dir, app_paths.cache_dir, app_paths.jobs_dir):
        assert d.is_dir()
    app_paths.ensure()  # idempotent


def test_translations_dir_exists():
    assert (paths.translations_dir() / "en.json").is_file()
    assert (paths.translations_dir() / "ar.json").is_file()
