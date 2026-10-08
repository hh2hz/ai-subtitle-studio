"""Per-line confidence category (HIGH / MEDIUM / LOW).

Formula (configurable through ConfidenceRules, covered by tests):
- LOW when any severe flag is present (meaning likely wrong or missing) or the audio confidence of the line's
  ASR segment is below low_audio.
- MEDIUM when any other flag is present or the audio confidence is below medium_audio.
- HIGH otherwise.
Audio confidence is the mean word probability from the ASR (see master_transcript.audio_confidence).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core import language_qa, subtitle_qa, verification

HIGH, MEDIUM, LOW = "HIGH", "MEDIUM", "LOW"

# Flags the pipeline and the finalizer set themselves. Every flag code the app can emit lives in ALL_FLAGS, so a
# missing user-facing string is caught by one test instead of showing the raw key (D-106).
NOT_REFINED_BY_AI = "not_refined_by_ai"
WEAK_AI_MODEL = "weak_ai_model"
AI_DISAGREEMENT = "ai_disagreement"
REVIEWER_SUGGESTION = "reviewer_suggestion"
NAME_MISMATCH = "name_mismatch"
LOW_AUDIO_CONFIDENCE = "low_audio_confidence"

PIPELINE_FLAGS = frozenset({NOT_REFINED_BY_AI, WEAK_AI_MODEL, AI_DISAGREEMENT, REVIEWER_SUGGESTION,
                            NAME_MISMATCH, LOW_AUDIO_CONFIDENCE})

#: Every flag code that can appear in a unit's "flags" list, from every module that emits one.
ALL_FLAGS = frozenset(PIPELINE_FLAGS | language_qa.FLAGS | subtitle_qa.FLAGS | verification.FLAGS)

SEVERE_DEFAULT = frozenset({
    "empty_translation", "untranslated_or_wrong_script", "numbers_differ", "repetition_loop",
    "not_refined_by_ai", "invalid_timing", "overlap", "empty_cue", "name_mismatch",
})
# Purely technical, auto-fixed or cosmetic flags that should not lower confidence in the meaning.
NEUTRAL_DEFAULT = frozenset({"extra_whitespace", "three_dots", "tatweel", "arabic_indic_digits",
                             "latin_punctuation", "space_before_punctuation"})


@dataclass(frozen=True)
class ConfidenceRules:
    low_audio: float = 0.5
    medium_audio: float = 0.75
    severe: frozenset = field(default=SEVERE_DEFAULT)
    neutral: frozenset = field(default=NEUTRAL_DEFAULT)


def category(flags: list[str], audio_confidence: float | None, rules: ConfidenceRules = ConfidenceRules()) -> str:
    relevant = [f for f in flags if f not in rules.neutral]
    if any(f in rules.severe for f in relevant) or (audio_confidence is not None and audio_confidence < rules.low_audio):
        return LOW
    if relevant or (audio_confidence is not None and audio_confidence < rules.medium_audio):
        return MEDIUM
    return HIGH
