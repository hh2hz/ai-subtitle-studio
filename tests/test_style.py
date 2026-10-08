"""Per-language style guides (task 2.5, D-060): loader, fallback and the byte-identical Arabic prompts."""

import json
from pathlib import Path

from app.core import local_llm, style
from app.core.llm_providers import ModelRanking, ProviderPool
from app.core.llm_translation import LlmRefiner
from app.utils.languages import english_name
from app.utils.paths import resources_dir

GOLDEN = Path(__file__).parent / "fixtures" / "style_golden"
TARGETS = ("ar", "fr", "de", "ru", "es", "tr", "en", "he", "fa", "zh", "ja", "ko")


def test_arabic_cloud_prompts_are_byte_identical_to_golden():
    refiner = LlmRefiner(ProviderPool([], ModelRanking(floor=0)), "tr", "ar", {})
    rendered = {"translator": refiner._translator_system, "corrector": refiner._corrector_system,
                "judge": refiner._judge_system, "name_fix": refiner._name_fix_system}
    for name, text in rendered.items():
        assert text.encode("utf-8") == (GOLDEN / f"tr_ar_{name}.txt").read_bytes(), name


def test_arabic_local_prompt_is_byte_identical_to_golden():
    text = local_llm.INSTRUCT_GUIDELINES.format(src=english_name("tr"), tgt=english_name("ar"),
                                                style=style.local_style("ar"))
    assert text.encode("utf-8") == (GOLDEN / "tr_ar_local_instruct.txt").read_bytes()
    assert style.local_style("ar-SA") == style.local_style("ar")


def test_guides_cover_all_targets_with_one_line_rules():
    data = json.loads((resources_dir() / "style_guides.json").read_text(encoding="utf-8"))
    assert set(data["target"]) == set(TARGETS)
    for code, guide in data["target"].items():
        assert guide["rules"] and all("\n" not in r and r.strip() for r in guide["rules"]), code
        assert guide["source"] == "generic" and guide["verified"] is False, code
        low, high = guide["length_ratio"]
        assert 0 < low < high
    assert data["target"]["zh"]["max_line"] == data["target"]["ja"]["max_line"] == 16


def test_rules_render_per_language():
    assert "Put a space before ? ! : ;" in style.cloud_style("fr")
    assert "\u00bf" in style.cloud_style("es")
    assert "\uff0c" in style.cloud_style("zh")
    assert style.cloud_style("fr").count("\n") == 2
    assert style.local_style("fr").splitlines()[0].startswith("- ")


def test_unknown_language_falls_back_to_generic():
    generic = json.loads((resources_dir() / "style_guides.json").read_text(encoding="utf-8"))["generic"]
    assert style.cloud_style("xx") == "\n".join(generic["rules"]) != ""
    assert style.cloud_style(None) == style.cloud_style("xx")
    assert style.length_ratio("xx") == (0.4, 2.5)
    assert (style.cps("xx"), style.max_line("xx")) == (17.0, 42)
    assert style.source_guide("xx") == {"notes": [], "negation": []}


def test_source_turkish_notes_and_negation():
    guide = style.source_guide("tr")
    assert any("Kinship" in n for n in guide["notes"])
    assert {"de\u011fil", "yok", "-ma", "-me"} <= set(guide["negation"])


def test_missing_resource_gives_generic_rules(monkeypatch, tmp_path):
    monkeypatch.setattr(style, "resources_dir", lambda: tmp_path)
    monkeypatch.setattr(style, "_cache", None)
    try:
        assert style.cloud_style("ar") == ""
        assert style.length_ratio("ar") == (0.4, 2.5)
    finally:
        monkeypatch.setattr(style, "_cache", None)


def test_non_arabic_prompt_contains_style_rules():
    refiner = LlmRefiner(ProviderPool([], ModelRanking(floor=0)), "tr", "fr", {})
    assert "Put a space before ? ! : ;" in refiner._translator_system
    assert "Put a space before ? ! : ;" in refiner._corrector_system
