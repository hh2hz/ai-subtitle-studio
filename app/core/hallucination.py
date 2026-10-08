"""Removal of text Whisper invents on music, noise or silence (D-044).

Whisper was trained on subtitled web video, so on non-speech it often writes subtitle credits or channel
outros ("Altyazi M.K.", "Thanks for watching", "Subtitles by the Amara.org community"). A segment is dropped when
its whole text is such a phrase, when it contains only symbols or music notes, or when the model itself rated it
as probably not speech (no_speech_prob >= 0.6) and was unsure of the words (avg_logprob <= -1.0).
"""

from __future__ import annotations

import unicodedata

# Lowercase, without punctuation. Matched against the whole segment text (also with a trailing name/number).
_PHRASES = (
    # Turkish
    "altyazı m k", "altyazı mk", "altyazı", "izlediğiniz için teşekkürler", "izlediğiniz için teşekkür ederim",
    "abone olmayı unutmayın", "altyazılar amara org topluluğu tarafından", "beğenmeyi ve abone olmayı unutmayın",
    # English
    "thanks for watching", "thank you for watching", "subtitles by the amara org community",
    "please subscribe", "like and subscribe", "subtitles by", "transcription by",
    # Arabic
    "\u062a\u0631\u062c\u0645\u0629 \u0646\u0627\u0646\u0633\u064a \u0642\u0646\u0642\u0631",
    "\u0627\u0634\u062a\u0631\u0643\u0648\u0627 \u0641\u064a \u0627\u0644\u0642\u0646\u0627\u0629",
    "\u0634\u0643\u0631\u0627 \u0644\u0644\u0645\u0634\u0627\u0647\u062f\u0629",
    # German / French / Spanish / Portuguese / Italian / Russian
    "untertitel der amara org community", "untertitel im auftrag des zdf", "vielen dank fürs zuschauen",
    "sous titres réalisés par la communauté d amara org", "sous titrage st 501", "merci d avoir regardé",
    "subtítulos realizados por la comunidad de amara org", "gracias por ver el video", "obrigado por assistir",
    "legendas pela comunidade amara org", "sottotitoli creati dalla comunità amara org", "grazie per la visione",
    "редактор субтитров",
    "субтитры сделала",
    "продолжение следует",
)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    text = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text)
    return " ".join(text.split())


# Credit lines that Whisper writes with a name or number after them ("Altyazi M.K.", "Subtitles by X Y").
_CREDITS = (
    "altyazı m k", "altyazı mk", "altyazılar amara org topluluğu tarafından", "subtitles by the amara org community",
    "subtitles by", "transcription by", "untertitel der amara org community", "untertitel im auftrag des zdf",
    "sous titres réalisés par la communauté d amara org", "sous titrage st 501",
    "subtítulos realizados por la comunidad de amara org", "legendas pela comunidade amara org",
    "sottotitoli creati dalla comunità amara org", "редактор субтитров", "субтитры сделала",
    "\u062a\u0631\u062c\u0645\u0629 \u0646\u0627\u0646\u0633\u064a \u0642\u0646\u0642\u0631",
)


def _is_phrase(text: str) -> bool:
    """The whole segment is a known phrase; credit phrases may be followed by up to three capitalised words or
    numbers (a name), never by ordinary words or a question ("Subtitles by tomorrow, okay?" is dialogue)."""
    norm = normalize(text)
    if not norm:
        return False
    if norm in _PHRASES:
        return True
    if text.rstrip().endswith("?"):
        return False
    for phrase in _CREDITS:
        if norm.startswith(phrase + " "):
            extra = text.split()[len(text.split()) - (len(norm.split()) - len(phrase.split())):]
            if len(extra) <= 3 and all(w[:1].isupper() or w[:1].isdigit() for w in extra):
                return True
    return False


def _loop(text: str) -> bool:
    """A 2-4 word group repeated at least 4 times in a row, or one word at least 8 times, covering most of the
    segment ("I mean I mean I mean I mean"). "no, no, no, no, no!" is ordinary speech."""
    words = normalize(text).split()
    for size in range(1, 5):
        need = 8 if size == 1 else 4
        for start in range(0, max(0, len(words) - size * need) + 1):
            gram = words[start:start + size]
            count = 1
            while words[start + count * size:start + (count + 1) * size] == gram:
                count += 1
            if count >= need and count * size >= 0.8 * len(words):
                return True
    return False


def reason(segment: dict) -> str | None:
    """Why a segment is considered invented, or None to keep it."""
    text = segment.get("text", "")
    if not any(ch.isalnum() for ch in text):
        return "no words"
    if _is_phrase(text):
        return "known non-speech phrase"
    no_speech, logprob = segment.get("no_speech_prob"), segment.get("avg_logprob")
    if no_speech is not None and logprob is not None and no_speech >= 0.6 and logprob <= -1.0:
        return "probably not speech"
    ratio = segment.get("compression_ratio")
    if _loop(text) or (ratio is not None and ratio > 2.4 and len(text) > 40):
        return "repetition"
    return None
