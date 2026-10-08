"""Person-name consistency between the source and the translation, for any pair of languages.

LLMs know dubbed and localized versions of popular works and sometimes replace names with the localized ones
(Kurtlar Vadisi in Arabic: "Polat" became the dubbed name "Murad", "Nazife" became "Nazik"; DECISIONS D-039).
A subtitle of the original must keep the original names, transliterated. The check compares consonant skeletons:
both the source name and each word of the translation are reduced to consonant classes (Arabic script with a
hand-made table, every other script through `anyascii` romanization) and must share most consonants in order
("Polat" -> b-l-t matches its Arabic, Cyrillic or Greek transliteration; "Murad" -> m-r-d is rejected).
Chinese and Japanese targets are not checked (names are written with meaning-bearing characters there).
"""

from __future__ import annotations

import json
import logging
import re
from difflib import SequenceMatcher
from pathlib import Path

log = logging.getLogger(__name__)

try:
    from anyascii import anyascii as _romanize
except ImportError:                       # optional at runtime; without it only Latin and Arabic script are checked
    _romanize = None

# Consonant classes shared by all scripts (vowels, glottal stops and long-vowel letters are ignored).
_LATIN_DIGRAPHS = (("sch", "S"), ("sh", "S"), ("ch", "C"), ("ph", "F"), ("th", "T"), ("kh", "H"), ("gh", "G"))
_LATIN = {
    "b": "B", "p": "B", "t": "T", "d": "T", "c": "C", "ç": "C", "j": "C", "s": "S", "ş": "S", "z": "S",
    "x": "KS", "f": "F", "v": "", "w": "", "g": "G", "ğ": "", "k": "K", "q": "K", "h": "H", "l": "L", "m": "M",
    "n": "N", "r": "R", "y": "",
}
# Arabic script (Arabic, Persian, Urdu letters).
_ARABIC = {
    "\u0628": "B", "\u067e": "B", "\u062a": "T", "\u0637": "T", "\u062f": "T", "\u0636": "T", "\u0679": "T",
    "\u0688": "T", "\u062b": "Q", "\u0633": "S", "\u0635": "S", "\u0634": "S", "\u0632": "S", "\u0630": "S",
    "\u0638": "S", "\u0698": "S", "\u062c": "C", "\u0686": "C", "\u0641": "F", "\u06a4": "F", "\u0642": "K",
    "\u0643": "K", "\u06a9": "K", "\u06af": "G", "\u063a": "G", "\u062d": "H", "\u062e": "H", "\u0647": "H",
    "\u06be": "H", "\u0644": "L", "\u0645": "M", "\u0646": "N", "\u06ba": "N", "\u0631": "R", "\u0691": "R",
}
_ARABIC_SCRIPT = re.compile(r"[\u0600-\u06ff]")
# Titles and address words are not part of the name ("Mehmet Bey", "Mr Smith").
_TITLES = {"bey", "hanım", "hanim", "ağa", "aga", "anne", "abi", "abla", "efendi", "paşa", "pasa", "hoca", "baba",
           "amca", "dayı", "dayi", "teyze", "bay", "bayan", "usta", "reis", "mr", "mrs", "ms", "miss", "dr", "sir",
           "madam", "herr", "frau", "señor", "señora", "senor", "senora", "don", "doña", "monsieur", "madame",
           "mademoiselle", "signor", "signora", "san", "kun", "chan", "sama"}
_ARABIC_PREFIXES = (
    "\u0648\u0627\u0644", "\u0641\u0627\u0644", "\u0628\u0627\u0644", "\u0644\u0644", "\u0627\u0644",
    "\u064a\u0627", "\u0648\u0644", "\u0648\u0628", "\u0641\u0644", "\u0641\u0628",
    "\u0648", "\u0628", "\u0644", "\u0641",
)
_TRAILING_VOWEL = ("\u0647", "\u0629")
# Consonants that scripts write interchangeably (p/f in Hebrew and Arabic-script languages, k/g).
_EQUIVALENT = str.maketrans({"F": "B", "G": "K"})
UNCHECKED_TARGETS = {"zh", "ja", "yue"}


def supported_target(language: str) -> bool:
    return language.split("-")[0].lower() not in UNCHECKED_TARGETS


def latin_skeleton(name: str) -> str:
    text = name.lower()
    for digraph, sound in _LATIN_DIGRAPHS:
        text = text.replace(digraph, f"[{sound}]")
    out = []
    for part in re.split(r"(\[[A-Z]+\])", text):
        if part.startswith("["):
            out.append(part[1:-1])
        else:
            out.extend(_LATIN.get(ch, "") for ch in part)
    # c/ç/j are written with jeem or sheen ("Celebi" -> sheen-lam-ba): one class with the other sibilants.
    return _collapse("".join(out).replace("C", "S"))


def arabic_skeleton(word: str) -> str:
    word = re.sub(r"[\u064b-\u065f\u0670\u0640]", "", word)          # harakat, tatweel
    if len(word) > 2 and word.endswith(_TRAILING_VOWEL):              # final -e/-a written with heh/teh marbuta
        word = word[:-1]
    skeleton = "".join(_ARABIC.get(ch, "") for ch in word).replace("TS", "S")
    return _collapse(skeleton.replace("C", "S"))


def skeleton(word: str) -> str:
    """Consonant skeleton of a word in any script."""
    if _ARABIC_SCRIPT.search(word):
        return arabic_skeleton(word)
    if all(ord(ch) < 0x250 for ch in word):                           # Latin (incl. Turkish letters)
        return latin_skeleton(word)
    if _romanize is None:
        return ""
    return latin_skeleton(_romanize(word))


def _collapse(skeleton_text: str) -> str:
    return re.sub(r"(.)\1+", r"\1", skeleton_text)


def core_tokens(name: str) -> list[str]:
    """Name tokens without titles; empty for a title alone ("Abi", "Bey"), which is never treated as a name."""
    return [t for t in re.split(r"[\s'’\-.]+", name) if t and t.lower() not in _TITLES]


def is_name(name: str) -> bool:
    return bool(core_tokens(name))


def certain_in(name: str, source: str) -> bool:
    """The name clearly appears as a proper noun in the source: a capitalised token in the middle of a sentence.
    A capital at the start of a sentence proves nothing ("Aslan gibi..." = "like a lion"), and scripts without
    letter case give no signal; such lines are flagged for review instead of rewritten automatically (D-047)."""
    for token in core_tokens(name):
        if not token[:1].isupper():
            continue
        for match in re.finditer(rf"(?<![\w]){re.escape(token)}", source):
            before = source[:match.start()].rstrip()
            if before and before[-1] not in ".!?…:;-–—\"'«»“”(":
                return True
    return False


def _similar(a: str, b: str) -> bool:
    if not a or not b:
        return a == b
    # Checked on a real 891-line episode: "Polat" b-l-t must not match unrelated words such as b-n-t ("my daughter")
    # or b-t ("after"), so the ratio is 0.75 and the target word may not lose a consonant of the name. Turkish "v"
    # is ignored (often written with a vowel letter such as waw).
    a, b = a.translate(_EQUIVALENT), b.translate(_EQUIVALENT)
    return a[0] == b[0] and len(b) >= len(a) and SequenceMatcher(None, a, b).ratio() >= 0.75


def _candidates(text: str) -> list[str]:
    """Skeletons of every word of a translation, of Arabic words without attached prefixes, and of word pairs
    (names written as two words, e.g. Zulfikar as "dhu al-fiqar")."""
    words = re.findall(r"[^\W\d_]+", text)
    result = []
    for i, word in enumerate(words):
        variants = {word}
        if _ARABIC_SCRIPT.search(word):
            for prefix in _ARABIC_PREFIXES:
                if word.startswith(prefix) and len(word) - len(prefix) >= 2:
                    variants.add(word[len(prefix):])
        if i + 1 < len(words):
            variants.add(word + words[i + 1])
        for variant in variants:
            sk = skeleton(variant)
            # theh ("Q") stands for "s" in Turkish names and for "th" in English ones.
            result.extend({sk.replace("Q", "S"), sk.replace("Q", "T")})
    return result


def token_ok(token: str, target_text: str, candidates: list[str] | None = None) -> bool:
    token_skeleton = skeleton(token)
    if not token_skeleton:
        return True
    candidates = candidates if candidates is not None else _candidates(target_text)
    return any(_similar(token_skeleton, c) for c in candidates)


def transliteration_ok(source_name: str, target: str) -> bool:
    """True when every core token of the source name has a phonetically matching word in `target`."""
    candidates = _candidates(target)
    return all(token_ok(token, target, candidates) for token in core_tokens(source_name))


def _present(token: str, text: str, longer: set[str] = frozenset()) -> bool:
    """Occurrence at a word start; suffixes may follow ("Polat'ın", "Polatla"), but not when the word is another,
    longer known name ("Kara" inside "Karahanlı")."""
    for match in re.finditer(rf"(?<![\w]){re.escape(token)}(\w*)", text):
        word = token + match.group(1)
        if not any(word.startswith(other) and len(other) > len(token) for other in longer):
            return True
    return False


def _name_like(token: str) -> bool:
    """Capitalised in scripts with letter case; any token in scripts without case (Arabic, CJK, Hebrew...)."""
    first = token[:1]
    return first.isupper() or not first.islower()


def line_issues(source: str, target: str, names: set[str] | dict, rejected: dict[str, set[str]]) -> list[str]:
    """Names in `source` that the translation renders wrongly: a rejected rendering (e.g. a dubbed name) is
    present, or no word of the translation matches the name phonetically."""
    issues = []
    candidates = None
    all_tokens = {t for n in names for t in core_tokens(n)}
    for name in names:
        tokens = [t for t in core_tokens(name) if _name_like(t) and _present(t, source, all_tokens)]
        if not tokens:
            continue
        if any(bad and bad in target for bad in rejected.get(name, ())):
            issues.append(name)
            continue
        if candidates is None:
            candidates = _candidates(target)
        if not all(token_ok(t, target, candidates) for t in tokens):
            issues.append(name)
    return issues


class SeriesGlossary:
    """Name spellings kept per series and target language across episodes: <data>/glossaries/<series>.<lang>.json.

    {"names": {"Polat": "<spelling in the target language>"}, "user": ["Polat"]} - names listed in "user" were set by the user
    (edit the file) and are never replaced; learned names are added only when they pass the transliteration check.
    """

    def __init__(self, path: Path | None, names: dict[str, str] | None = None, user: set[str] | None = None):
        self.path = path
        self.names = dict(names or {})
        self.user = set(user or ())

    @classmethod
    def for_series(cls, data_dir: Path, series_name: str | None, target_language: str = "ar") -> "SeriesGlossary":
        slug = re.sub(r"[^\w]+", "-", (series_name or "").strip().lower()).strip("-")
        if not slug:
            return cls(None)
        path = Path(data_dir) / "glossaries" / f"{slug}.{target_language.lower()}.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls(path)
        names = {k: v for k, v in (data.get("names") or {}).items() if isinstance(k, str) and isinstance(v, str)}
        user = {n for n in data.get("user") or () if n in names}
        # Learned names are re-checked on load (older versions accepted dubbed names); user names are kept as set.
        names = {k: v for k, v in names.items() if k in user or transliteration_ok(k, v)}
        return cls(path, names, user)

    def learn(self, names: dict[str, str]) -> int:
        added = 0
        for source, target in names.items():
            if (source in self.user or source in self.names or not is_name(source)
                or not transliteration_ok(source, target)):
                continue
            self.names[source] = target
            added += 1
        return added

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"names": dict(sorted(self.names.items())), "user": sorted(self.user)}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.path)


# -- deterministic name normalisation (D-058, task 2.6) --------------------------------------------

_STOPLIST: dict[str, set[str]] | None = None


def load_stoplist(source_language: str | None = None) -> set[str]:
    """Load the stoplist of common words that are also names from app/resources/name_stoplist.json."""
    global _STOPLIST
    if _STOPLIST is None:
        try:
            from app.utils.paths import resources_dir
            path = resources_dir() / "name_stoplist.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            _STOPLIST = {lang.lower(): {w.casefold() for w in words} for lang, words in data.items()}
        except Exception:
            _STOPLIST = {}
    if source_language is None:
        return set().union(*_STOPLIST.values()) if _STOPLIST else set()
    lang_key = source_language.split("-")[0].lower()
    return _STOPLIST.get(lang_key, set())


def _lower_lang(text: str, language: str) -> str:
    lang_prefix = language.split("-")[0].lower()
    if lang_prefix in ("tr", "az"):
        text = text.replace("I", "\u0131").replace("\u0130", "i")
    return text.casefold()


def is_stoplisted(name: str, source_language: str) -> bool:
    """Check if the name is a common word in the source language."""
    stoplist = load_stoplist(source_language)
    if not stoplist:
        return False
    tokens = core_tokens(name)
    if len(tokens) == 1:
        if _lower_lang(tokens[0], source_language) in stoplist:
            return True
    return _lower_lang(name.strip(), source_language) in stoplist


_NORMALIZE_AR = str.maketrans({"\u0623": "\u0627", "\u0625": "\u0627", "\u0622": "\u0627", "\u0649": "\u064a",
                               "\u0629": "\u0647", "\u0640": None})


def _edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def _match_target_word(word: str, spelling: str) -> tuple[int, str] | None:
    """(distance, replacement) when `word`, or `word` without an attached Arabic prefix, is a near spelling of the
    glossary `spelling`; None otherwise. "Near" is strict, because a loose consonant match turns ordinary words
    into names (checked: "on" -> "Ali", "died" -> "Memati", "your town" -> "Polat"): the spelling has at least
    4 letters, at most 1 letter differs (2 for 7+ letters) after alef/yaa/taa-marbuta folding, and the consonant
    skeletons still match."""
    target = spelling.translate(_NORMALIZE_AR)
    if len(target) < 4 or word == spelling:
        return None
    limit = 2 if len(target) >= 7 else 1
    best = None
    prefixes = ("",) + (_ARABIC_PREFIXES if _ARABIC_SCRIPT.search(word) else ())
    for prefix in prefixes:
        if prefix and (not word.startswith(prefix) or spelling.startswith(prefix)):
            continue
        stem = word[len(prefix):]
        if len(stem) < 3:
            continue
        if stem == spelling:
            return None                                   # already correct with a prefix
        distance = _edit_distance(stem.translate(_NORMALIZE_AR), target)
        if distance <= limit and _similar(skeleton(spelling), skeleton(stem)) and \
                (best is None or distance < best[0]):
            best = (distance, prefix + spelling)
    return best


def _name_present(name: str, source_text: str, all_tokens: set[str]) -> bool:
    tokens = [t for t in core_tokens(name) if _name_like(t)]
    if not tokens:
        tokens = [name]
    return any(_present(t, source_text, all_tokens) for t in tokens)


def normalize_target_text(
    source_text: str,
    target_text: str,
    glossary: dict[str, str],
    user_names: set[str] | None = None,
    source_language: str = "tr",
    target_language: str = "ar",
) -> tuple[str, list[dict]]:
    """Deterministically normalise name spellings in target_text based on glossary (Task 2.6).

    Returns (new_target_text, list_of_replacements).
    """
    if not supported_target(target_language) or not glossary or not target_text.strip():
        return target_text, []

    user = set(user_names or ())
    # user names first, then learned names
    ordered_names = [n for n in user if n in glossary] + [n for n in glossary if n not in user]
    all_tokens = {t for n in ordered_names for t in core_tokens(n)}

    replacements_log: list[dict] = []
    current_text = target_text

    for source_name in ordered_names:
        # Names in the stoplist are never auto-enforced or normalised unless pinned in user.
        if is_stoplisted(source_name, source_language) and source_name not in user:
            continue

        if not _name_present(source_name, source_text, all_tokens):
            continue

        glossary_spelling = glossary[source_name]
        if not glossary_spelling:
            continue

        spans_to_replace: list[tuple[int, int, str, str]] = []
        for part in glossary_spelling.split():           # each word of a multi-word spelling on its own
            words = list(re.finditer(r"[^\W\d_]+", current_text))
            if any(m.group(0) == part or (m.group(0).endswith(part) and m.group(0)[:-len(part)] in _ARABIC_PREFIXES)
                   for m in words):
                continue                                  # already spelled right in this line: touch nothing
            found = [(m, res) for m in words if (res := _match_target_word(m.group(0), part)) is not None]
            if len(found) == 1:                           # two candidates: ambiguous, leave the line alone
                m, (_, replacement) = found[0]
                spans_to_replace.append((m.start(), m.end(), m.group(0), replacement))

        if spans_to_replace:
            # Replace in reverse order so character offsets stay valid
            for start, end, old_word, new_word in sorted(spans_to_replace, key=lambda s: s[0], reverse=True):
                current_text = current_text[:start] + new_word + current_text[end:]
                replacements_log.append({
                    "name": source_name,
                    "old": old_word,
                    "new": new_word,
                })

    return current_text, replacements_log


def _spelling_hits(spelling: str, text: str) -> list[tuple[int, int, str, str]]:
    """Whole-word occurrences of a glossary spelling, as (start, end, matched word, attached Arabic prefix)."""
    if not spelling or not text:
        return []
    words = list(re.finditer(r"[^\W\d_]+", text))
    hits: list[tuple[int, int, str, str]] = []
    for part in spelling.split():
        for match in words:
            word = match.group(0)
            if word == part:
                hits.append((match.start(), match.end(), word, ""))
                continue
            for prefix in _ARABIC_PREFIXES:
                if word == prefix + part:
                    hits.append((match.start(), match.end(), word, prefix))
                    break
    return hits


def repair_swapped_names(
    source_text: str,
    target_text: str,
    glossary: dict[str, str],
    user_names: set[str] | None = None,
    source_language: str = "tr",
    target_language: str = "ar",
) -> tuple[str, list[dict]]:
    """Replace a name the translation swapped for a different one (D-079, task B).

    The condition is deliberately narrow, because a wrong replacement is worse than a flagged line: exactly ONE
    glossary name that the source contains is missing from the translation (its tokens have no phonetically
    matching word - the same test `line_issues` flags with), and exactly ONE glossary spelling of a name the source
    does NOT contain appears in the translation as a whole word. Two candidates, none, or an ambiguous line change
    nothing. The attached Arabic prefix of the replaced word is kept.

    Returns (new_target_text, replacements); replacements is empty when nothing was changed.
    """
    if (not glossary or not supported_target(target_language)
            or not source_text.strip() or not target_text.strip()):
        return target_text, []

    user = set(user_names or ())
    ordered = [n for n in user if n in glossary] + [n for n in glossary if n not in user]
    all_tokens = {t for n in ordered for t in core_tokens(n)}

    missing: list[str] = []
    wrong: list[tuple[str, list[tuple[int, int, str, str]]]] = []
    candidates: list[str] | None = None
    for name in ordered:
        spelling = glossary.get(name) or ""
        parts = spelling.split()
        if not parts or (is_stoplisted(name, source_language) and name not in user):
            continue
        tokens = core_tokens(name)
        # "Present in the source" means the whole name is there (suffix-aware): "Polat Alemdar" is not present in a
        # line that only says "Polat'tan", and counting it as a second missing name would block every repair.
        if not tokens or any(not _name_like(t) or not _present(t, source_text, all_tokens) for t in tokens):
            hits = _spelling_hits(spelling, target_text)
            if hits and len(hits) >= len(parts):        # the whole wrong spelling is in the translation
                wrong.append((name, hits))
            continue
        if candidates is None:
            candidates = _candidates(target_text)
        if not all(token_ok(token, target_text, candidates) for token in tokens):
            missing.append(name)

    if len(missing) != 1 or len(wrong) != 1:
        return target_text, []

    right_name, wrong_name = missing[0], wrong[0][0]
    right_spelling = glossary[right_name]
    replacements: list[dict] = []
    new_text = target_text
    wrong_parts = glossary[wrong_name].split()
    if len(wrong_parts) > 1:
        # A multi-word wrong spelling is replaced as ONE phrase (each word replaced separately would repeat the
        # right name). Words that are not adjacent in the line: change nothing.
        letter = r"[^\W\d_]"
        pattern = re.compile(
            rf"(?<!{letter})((?:{'|'.join(re.escape(p) for p in _ARABIC_PREFIXES)})?)"
            + r"\s+".join(re.escape(part) for part in wrong_parts) + rf"(?!{letter})")
        found = list(pattern.finditer(target_text))
        if not found:
            return target_text, []
        for match in reversed(found):
            new_word = match.group(1) + right_spelling
            new_text = new_text[:match.start()] + new_word + new_text[match.end():]
            replacements.append({"name": right_name, "old": match.group(0), "new": new_word,
                                 "replaced": wrong_name})
        return new_text, replacements
    # Replace from the end so the offsets of the remaining hits stay valid.
    for start, end, old_word, prefix in sorted(wrong[0][1], key=lambda hit: hit[0], reverse=True):
        new_word = prefix + right_spelling
        new_text = new_text[:start] + new_word + new_text[end:]
        replacements.append({"name": right_name, "old": old_word, "new": new_word, "replaced": wrong_name})
    return new_text, replacements


def normalize_units(
    units: list[dict],
    glossary: dict[str, str],
    user_names: set[str] | None = None,
    source_language: str = "tr",
    target_language: str = "ar",
) -> list[dict]:
    """Apply deterministic name normalisation to all translation units."""
    if not supported_target(target_language) or not glossary:
        return units

    for unit in units:
        source = unit.get("text", "")
        target = unit.get("translation", "")
        if not source or not target:
            continue

        new_target, replacements = normalize_target_text(
            source, target, glossary, user_names, source_language, target_language
        )
        if replacements:
            for rep in replacements:
                log.info("Normalized name in line %d: %s -> %s for %s",
                         unit.get("id", 0), rep["old"], rep["new"], rep["name"])
            unit["translation"] = new_target
            if unit.get("llm_translation"):
                unit["llm_translation"] = new_target

            if "name_mismatch" in unit.get("flags", []):
                remaining = line_issues(source, new_target, set(glossary), {})
                if not remaining:
                    unit["flags"] = [f for f in unit["flags"] if f != "name_mismatch"]
                    log.info("Line %d name_mismatch resolved by normalisation", unit.get("id", 0))

        # A name the translation swapped for another glossary name (D-079): same step, same setting.
        repaired, swaps = repair_swapped_names(
            source, unit.get("translation", ""), glossary, user_names, source_language, target_language)
        if swaps:
            for swap in swaps:
                log.info("Repaired swapped name in line %d: %s -> %s (was %s)",
                         unit.get("id", 0), swap["old"], swap["new"], swap["replaced"])
            unit["translation"] = repaired
            if unit.get("llm_translation"):
                unit["llm_translation"] = repaired
            if "name_mismatch" in unit.get("flags", []) and not line_issues(source, repaired, set(glossary), {}):
                unit["flags"] = [f for f in unit["flags"] if f != "name_mismatch"]
                log.info("Line %d name_mismatch resolved by the swap repair", unit.get("id", 0))

    return units
