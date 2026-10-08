"""Tests for deterministic name normalisation (Task 2.6, DECISIONS D-058)."""

from __future__ import annotations

import logging
from app.core import names


POLAT_CANONICAL = "\u0628\u0648\u0644\u0627\u062a"  # Polat canonical Arabic
POLAT_VARIANT = "\u0628\u0648\u0644\u0627\u0637"    # Polat variant with tah instead of teh
ASLAN_CANONICAL = "\u0623\u0635\u0644\u0627\u0646"  # Aslan canonical Arabic with sad
ASLAN_VARIANT = "\u0623\u0633\u0644\u0627\u0646"    # Aslan variant with seen


def test_variant_replaced():
    """Spelling variant with matching consonant skeleton is replaced by glossary spelling."""
    source = "Polat geldi."
    target = f"{POLAT_VARIANT} \u062c\u0627\u0621."
    glossary = {"Polat": POLAT_CANONICAL}

    result, reps = names.normalize_target_text(
        source, target, glossary, user_names=set(), source_language="tr", target_language="ar"
    )
    assert result == f"{POLAT_CANONICAL} \u062c\u0627\u0621."
    assert len(reps) == 1
    assert reps[0]["old"] == POLAT_VARIANT
    assert reps[0]["new"] == POLAT_CANONICAL
    assert reps[0]["name"] == "Polat"


def test_prefix_kept():
    """Attached Arabic prefixes (waw, ba, lam, al, ya, wal, etc.) are preserved upon replacement."""
    glossary = {"Polat": POLAT_CANONICAL}

    # waw prefix
    target_waw = f"\u0648{POLAT_VARIANT} \u062c\u0627\u0621."
    result_waw, _ = names.normalize_target_text(
        "Polat geldi.", target_waw, glossary, source_language="tr", target_language="ar"
    )
    assert result_waw == f"\u0648{POLAT_CANONICAL} \u062c\u0627\u0621."

    # lam prefix
    target_lam = f"\u0642\u0644 \u0644{POLAT_VARIANT}."
    result_lam, _ = names.normalize_target_text(
        "Polat'a soyle.", target_lam, glossary, source_language="tr", target_language="ar"
    )
    assert result_lam == f"\u0642\u0644 \u0644{POLAT_CANONICAL}."

    # al prefix
    target_al = f"\u0627\u0644{POLAT_VARIANT} \u0647\u0646\u0627."
    result_al, _ = names.normalize_target_text(
        "Polat burada.", target_al, glossary, source_language="tr", target_language="ar"
    )
    assert result_al == f"\u0627\u0644{POLAT_CANONICAL} \u0647\u0646\u0627."

    # wal prefix
    target_wal = f"\u0648\u0627\u0644{POLAT_VARIANT} \u0647\u0646\u0627."
    result_wal, _ = names.normalize_target_text(
        "Ve Polat burada.", target_wal, glossary, source_language="tr", target_language="ar"
    )
    assert result_wal == f"\u0648\u0627\u0644{POLAT_CANONICAL} \u0647\u0646\u0627."

    # ya prefix
    target_ya = f"\u064a\u0627{POLAT_VARIANT}!"
    result_ya, _ = names.normalize_target_text(
        "Polat!", target_ya, glossary, source_language="tr", target_language="ar"
    )
    assert result_ya == f"\u064a\u0627{POLAT_CANONICAL}!"


def test_stoplisted_word_untouched():
    """Common words that are also names (e.g. aslan = lion) are not normalised unless in user."""
    source = "Aslan gibi dovustu."
    target = f"\u0642\u0627\u062a\u0644 \u0645\u062b\u0644 {ASLAN_VARIANT}."
    glossary = {"Aslan": ASLAN_CANONICAL}

    result, reps = names.normalize_target_text(
        source, target, glossary, user_names=set(), source_language="tr", target_language="ar"
    )
    assert result == target
    assert reps == []


def test_stoplisted_word_normalised_when_pinned_in_user():
    """Names in the stoplist are normalised if pinned in series_glossary.user."""
    source = "Aslan gibi dovustu."
    target = f"\u0642\u0627\u062a\u0644 \u0645\u062b\u0644 {ASLAN_VARIANT}."
    glossary = {"Aslan": ASLAN_CANONICAL}

    result, reps = names.normalize_target_text(
        source, target, glossary, user_names={"Aslan"}, source_language="tr", target_language="ar"
    )
    assert result == f"\u0642\u0627\u062a\u0644 \u0645\u062b\u0644 {ASLAN_CANONICAL}."
    assert len(reps) == 1
    assert reps[0]["name"] == "Aslan"


def test_name_absent_from_source_untouched():
    """Target words matching a glossary name skeleton are not modified if the name is not in source."""
    source = "Merhaba dostum."
    target = f"\u0645\u0631\u062d\u0628\u0627 {POLAT_VARIANT}."
    glossary = {"Polat": POLAT_CANONICAL}

    result, reps = names.normalize_target_text(
        source, target, glossary, user_names=set(), source_language="tr", target_language="ar"
    )
    assert result == target
    assert reps == []


def test_setting_false_no_replacement(tmp_path, caplog):
    """When name_normalization setting is false, pipeline _normalize_names does nothing."""
    from app.core.pipeline import JobConfig, Mode, Pipeline

    config = JobConfig(tmp_path / "dummy.mp4", "tr", "ar", Mode.FAST)
    pipeline = Pipeline(
        config, tmp_path / "jobs",
        [object()], [object()],
        lambda p, o, c: None, lambda p, b, c: None,
        name_normalization=False,
    )
    units = [
        {"id": 0, "text": "Polat geldi.", "translation": f"{POLAT_VARIANT} \u062c\u0627\u0621.", "flags": []}
    ]
    translation = {
        "source_language": "tr",
        "target_language": "ar",
        "units": units,
        "summary": {"glossary": {"Polat": POLAT_CANONICAL}},
    }
    with caplog.at_level(logging.INFO):
        res = pipeline._normalize_names(translation)
    assert res["units"][0]["translation"] == f"{POLAT_VARIANT} \u062c\u0627\u0621."
    assert "Name normalisation disabled by setting" in caplog.text


def test_normalize_units_resolves_name_mismatch_flag():
    """When a line has name_mismatch and normalisation fixes the name, the flag is removed."""
    units = [
        {"id": 0, "text": "Polat geldi.", "translation": f"{POLAT_VARIANT} \u062c\u0627\u0621.",
         "flags": ["name_mismatch", "weak_ai_model"]}
    ]
    glossary = {"Polat": POLAT_CANONICAL}
    normalized = names.normalize_units(
        units, glossary, user_names=set(), source_language="tr", target_language="ar"
    )
    assert normalized[0]["translation"] == f"{POLAT_CANONICAL} \u062c\u0627\u0621."
    assert "name_mismatch" not in normalized[0]["flags"]
    assert "weak_ai_model" in normalized[0]["flags"]


def test_pipeline_normalisation_enabled_replaces_variant(tmp_path):
    """When name_normalization is True, pipeline _normalize_names performs replacements."""
    from app.core.pipeline import JobConfig, Mode, Pipeline

    config = JobConfig(tmp_path / "dummy.mp4", "tr", "ar", Mode.FAST)
    pipeline = Pipeline(
        config, tmp_path / "jobs",
        [object()], [object()],
        lambda p, o, c: None, lambda p, b, c: None,
        name_normalization=True,
    )
    units = [
        {"id": 0, "text": "Polat geldi.", "translation": f"{POLAT_VARIANT} \u062c\u0627\u0621.", "flags": []}
    ]
    translation = {
        "source_language": "tr",
        "target_language": "ar",
        "units": units,
        "summary": {"glossary": {"Polat": POLAT_CANONICAL}},
    }
    res = pipeline._normalize_names(translation)
    assert res["units"][0]["translation"] == f"{POLAT_CANONICAL} \u062c\u0627\u0621."


def test_ordinary_words_are_not_turned_into_names():
    """Regression (review of 2.6): a loose consonant match replaced common words with names."""
    cases = [
        ("Ali", "\u0639\u0644\u064a", "Ali, otur.", "\u0627\u062c\u0644\u0633 \u0639\u0644\u0649 \u0627\u0644\u0643\u0631\u0633\u064a"),
        ("Polat", POLAT_CANONICAL, "Polat nerede?", "\u0623\u064a\u0646 \u0628\u0644\u062f\u062a\u0643\u061f"),
        ("Memati", "\u0645\u0645\u0627\u062a\u064a", "Memati bak.", "\u0645\u0627\u062a \u0645\u064a\u062a\u0629"),
    ]
    for name, spelling, source, target in cases:
        result, reps = names.normalize_target_text(source, target, {name: spelling},
                                                   source_language="tr", target_language="ar")
        assert result == target and reps == [], name


def test_line_with_the_right_spelling_is_untouched():
    target = f"{POLAT_CANONICAL} \u0648{POLAT_VARIANT}"
    result, reps = names.normalize_target_text("Polat geldi.", target, {"Polat": POLAT_CANONICAL},
                                               source_language="tr", target_language="ar")
    assert result == target and reps == []


# -- swapped-name repair (D-079, task B) ----------------------------------------------------------
# Real line 36 of the Pusu 198 run: "Polat'tan" was translated with the glossary spelling of Murat.
MURAT_CANONICAL = "\u0645\u0631\u0627\u062f"
HUSNU = "\u0633\u064a\u062f \u062d\u0633\u0646\u064a\u060c"                       # "Sayyid Husni,"
TASHAQQAL = "\u062a\u0641\u0631\u0642 \u0627\u0644\u062c\u0645\u064a\u0639 \u0628\u0639\u062f"   # "everyone scattered after"
LINE36_SOURCE = "H\u00fcsn\u00fc Bey, Polat'tan sonra herkes bir yerlere da\u011f\u0131ld\u0131."
LINE36_TARGET = f"{HUSNU} {TASHAQQAL} {MURAT_CANONICAL}."
LINE36_FIXED = f"{HUSNU} {TASHAQQAL} {POLAT_CANONICAL}."


def test_swapped_name_is_repaired():
    """One source name missing from the translation and one wrong glossary spelling present -> swap them."""
    result, reps = names.repair_swapped_names(
        LINE36_SOURCE, LINE36_TARGET, {"Polat": POLAT_CANONICAL, "Murat": MURAT_CANONICAL},
        source_language="tr", target_language="ar")

    assert result == LINE36_FIXED
    assert len(reps) == 1
    assert reps[0]["name"] == "Polat" and reps[0]["replaced"] == "Murat"
    assert reps[0]["old"] == MURAT_CANONICAL and reps[0]["new"] == POLAT_CANONICAL


def test_repair_keeps_a_longer_name_variant_from_blocking_it():
    """The glossary also holds "Polat Alemdar" and "Murat Argun"; only the names really in the line may count."""
    glossary = {"Polat": POLAT_CANONICAL, "Polat Alemdar": f"{POLAT_CANONICAL} \u0639\u0644\u0645\u062f\u0627\u0631",
                "Murat": MURAT_CANONICAL, "Murat Argun": f"{MURAT_CANONICAL} \u0623\u0631\u063a\u0648\u0646"}
    result, reps = names.repair_swapped_names(LINE36_SOURCE, LINE36_TARGET, glossary,
                                              source_language="tr", target_language="ar")
    assert result == LINE36_FIXED and len(reps) == 1


def test_untouched_when_the_wrong_name_is_also_in_the_source():
    """Murat really speaks in this line, so his spelling is not a mistake: nothing may be swapped."""
    source = "Polat'tan sonra Murat geldi."
    target = f"{TASHAQQAL} {MURAT_CANONICAL} \u062c\u0627\u0621."
    result, reps = names.repair_swapped_names(source, target,
                                              {"Polat": POLAT_CANONICAL, "Murat": MURAT_CANONICAL},
                                              source_language="tr", target_language="ar")
    assert result == target and reps == []


def test_untouched_when_two_names_are_missing():
    """Two missing names and one wrong spelling: ambiguous, the reviewer decides."""
    source = "Polat ve Elif geldi."
    target = f"{MURAT_CANONICAL} \u062c\u0627\u0621."
    result, reps = names.repair_swapped_names(
        source, target, {"Polat": POLAT_CANONICAL, "Elif": "\u0625\u064a\u0644\u064a\u0641", "Murat": MURAT_CANONICAL},
        source_language="tr", target_language="ar")
    assert result == target and reps == []


def test_untouched_when_the_missing_name_is_still_there_phonetically():
    """The Pusu 198 line 412 case: the glossary spelling differs by one letter, the name is not missing."""
    source = "Aksa\u00e7l\u0131 ba\u015fka t\u00fcrl\u00fcs\u00fcn\u00fc d\u00fc\u015f\u00fcnm\u00fcyor."
    target = "\u0627\u0644\u0623\u0643\u0633\u0627\u0643\u0644\u064a \u0644\u0627 \u064a\u0641\u0643\u0631 \u0628\u063a\u064a\u0631 \u0630\u0644\u0643."
    result, reps = names.repair_swapped_names(
        source, target, {"Aksa\u00e7l\u0131": "\u0623\u0643\u0633\u0627\u062c\u0644\u064a",
                         "Murat": MURAT_CANONICAL}, source_language="tr", target_language="ar")
    assert result == target and reps == []


def test_repair_keeps_an_attached_prefix():
    target_waw = f"\u0648{MURAT_CANONICAL} {TASHAQQAL}."
    result, reps = names.repair_swapped_names(
        "Polat'tan sonra.", target_waw, {"Polat": POLAT_CANONICAL, "Murat": MURAT_CANONICAL},
        source_language="tr", target_language="ar")
    assert result == f"\u0648{POLAT_CANONICAL} {TASHAQQAL}."
    assert reps[0]["old"] == f"\u0648{MURAT_CANONICAL}" and reps[0]["new"] == f"\u0648{POLAT_CANONICAL}"


def test_swapped_name_repair_disabled_by_setting(tmp_path, caplog):
    """The repair lives in the same step as normalisation, so the same setting turns it off."""
    from app.core.pipeline import JobConfig, Mode, Pipeline

    config = JobConfig(tmp_path / "dummy.mp4", "tr", "ar", Mode.FAST)
    pipeline = Pipeline(
        config, tmp_path / "jobs",
        [object()], [object()],
        lambda p, o, c: None, lambda p, b, c: None,
        name_normalization=False,
    )
    units = [{"id": 36, "text": LINE36_SOURCE, "translation": LINE36_TARGET, "flags": []}]
    translation = {
        "source_language": "tr", "target_language": "ar", "units": units,
        "summary": {"glossary": {"Polat": POLAT_CANONICAL, "Murat": MURAT_CANONICAL}},
    }
    with caplog.at_level(logging.INFO):
        res = pipeline._normalize_names(translation)

    assert res["units"][0]["translation"] == LINE36_TARGET
    assert MURAT_CANONICAL in res["units"][0]["translation"]


def test_swapped_name_repair_clears_the_flag_and_logs(caplog):
    """A repaired line drops its name_mismatch flag, and every replacement is logged (B2)."""
    units = [{"id": 36, "text": LINE36_SOURCE, "translation": LINE36_TARGET,
              "flags": ["name_mismatch", "weak_ai_model"]}]
    with caplog.at_level(logging.INFO):
        out = names.normalize_units(units, {"Polat": POLAT_CANONICAL, "Murat": MURAT_CANONICAL},
                                    user_names=set(), source_language="tr", target_language="ar")

    assert out[0]["translation"] == LINE36_FIXED
    assert "name_mismatch" not in out[0]["flags"]
    assert "weak_ai_model" in out[0]["flags"]
    assert "Repaired swapped name in line 36" in caplog.text


def test_multi_word_wrong_name_is_replaced_once_not_per_word():
    glossary = {"Polat Alemdar": "\u0628\u0648\u0644\u0627\u062a \u0639\u0644\u0645\u062f\u0627\u0631",
                "Memati Ba\u015f": "\u0645\u064a\u0645\u0627\u062a\u064a \u0628\u0627\u0634"}
    source = "Polat Alemdar'\u0131 g\u00f6rd\u00fc."
    target = "\u0631\u0623\u0649 \u0645\u064a\u0645\u0627\u062a\u064a \u0628\u0627\u0634."
    new_text, replacements = names.repair_swapped_names(source, target, glossary, None, "tr", "ar")
    assert new_text == "\u0631\u0623\u0649 \u0628\u0648\u0644\u0627\u062a \u0639\u0644\u0645\u062f\u0627\u0631."
    assert len(replacements) == 1
