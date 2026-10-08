"""Estimated scores (name-based guesses) must never behave like measured ones (D-047, D-078)."""

import json

from app.core.llm_providers import ChatClient, ModelRanking, ProviderPool, ProviderSpec


def _ranking(tmp_path):
    estimates = tmp_path / "estimates.json"
    estimates.write_text(json.dumps({"floor": 78, "models": {
        "a:big": {"score": 90, "estimated": True},
        "a:measured": {"score": 99, "estimated": True},
        "a:other": {"score": 80, "estimated": True}}}), encoding="utf-8")
    measured = tmp_path / "measured.json"
    measured.write_text(json.dumps({"floor": 78, "models": {
        "a:measured": {"score": 85},
        "a:low": {"score": 79}}}), encoding="utf-8")
    return ModelRanking.load(estimates, measured)


def _pool(ranking):
    spec = ProviderSpec("a", "http://localhost", prefer=(), fallback_models=("big", "measured", "other", "low"))
    client = ChatClient(spec, "key")
    client.models = ["big", "measured", "other", "low"]
    return ProviderPool([client], ranking)


def test_measured_score_overrides_estimate(tmp_path):
    ranking = _ranking(tmp_path)
    assert ranking.scores["a:measured"] == 85.0
    assert "a:measured" not in ranking.estimates
    assert ranking.estimates["a:big"] == 90.0
    assert "a:big" not in ranking.scores


def test_estimated_models_rank_below_every_measured_model_above_floor(tmp_path):
    routes = _pool(_ranking(tmp_path)).routes()
    by_model = {r.model: r for r in routes}
    assert [r.model for r in routes][:2] == ["measured", "low"]
    assert by_model["big"].score < 78 and not by_model["big"].measured
    assert by_model["big"].score > by_model["other"].score          # the estimated order is kept


def test_estimated_models_cannot_correct(tmp_path):
    pool = _pool(_ranking(tmp_path))
    allowed = {r.model for r in pool.available(min_score=pool.ranking.floor)}
    assert allowed == {"measured", "low"}
