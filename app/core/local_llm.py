"""Context-aware translation with a local LLM (TranslateGemma) running in the built-in llama.cpp server.

Two prompt styles:
- "translategemma": Google's TranslateGemma models, using their published prompt template. Context is
  provided by translating the previous lines together with the new ones (their output is discarded).
- "instruct": any general instruction model (e.g. gemma3, qwen3), with explicit subtitle guidelines,
  previous translated lines as context and JSON-schema constrained output.
Every block is validated; lines missing from the model output are retranslated one by one.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Protocol

from app.core import style as style_guides
from app.utils.languages import english_name

log = logging.getLogger(__name__)

_MARKER = re.compile(r"^\s*\[(\d+)\]\s*(.*)$")
_THINK = re.compile(r"<think>.*?</think>", re.S)


class LocalModelError(RuntimeError):
    pass


class CompletionClient(Protocol):
    """Anything that can answer chat messages (the built-in llama.cpp server)."""

    def chat(self, model: str, messages: list[dict], fmt: dict | None = None, num_ctx: int = 4096) -> str: ...

    def unload(self, model: str) -> None: ...


def clean_line(text: str) -> str:
    text = _THINK.sub("", text).strip()
    match = _MARKER.match(text)
    if match:
        text = match.group(2)
    text = text.strip().strip('"').strip("\u201c\u201d\u00ab\u00bb").strip()
    return " ".join(text.split())


def prompt_style(model: str) -> str:
    return "translategemma" if model.lower().startswith("translategemma") else "instruct"


def translategemma_prompt(text: str, source: str, target: str) -> str:
    """Published TranslateGemma template (two blank lines before the text)."""
    src_name, tgt_name = english_name(source), english_name(target)
    return (
        f"You are a professional {src_name} ({source}) to {tgt_name} ({target}) translator. "
        f"Your goal is to accurately convey the meaning and nuances of the original {src_name} text while "
        f"adhering to {tgt_name} grammar, vocabulary, and cultural sensitivities.\n"
        f"Produce only the {tgt_name} translation, without any additional explanations or commentary. "
        f"Please translate the following {src_name} text into {tgt_name}:\n\n\n{text}"
    )


INSTRUCT_GUIDELINES = """You translate film and TV dialogue subtitles from {src} into {tgt}.
Rules:
- Translate the meaning and intent of each line, not word by word. Idioms, slang and insults must become natural {tgt} equivalents.
- Keep each line short and readable as a subtitle. Never merge lines and never split them: exactly one translation per id.
- Keep person names as names (transliterate them consistently); never translate a name as a word.
- Keep imperatives, questions, negation, numbers and the speaker's tone.
- Use the previous lines only as context; do not translate them again.
{style}
Return JSON only."""

# Per-target style rules live in app/resources/style_guides.json (loaded by app/core/style.py, D-060).

_SCHEMA = {
    "type": "object",
    "properties": {"translations": {"type": "array", "items": {
        "type": "object", "properties": {"id": {"type": "integer"}, "text": {"type": "string"}},
        "required": ["id", "text"]}}},
    "required": ["translations"],
}


class LocalTranslationBackend:
    def __init__(self, client: CompletionClient, model: str, context_lines: int = 3):
        self.client = client
        self.model = model
        self.style = prompt_style(model)
        self.context_lines = context_lines
        self.name = f"local:{model}"

    def close(self) -> None:
        self.client.unload(self.model)

    # -- public ------------------------------------------------------------------------------

    def translate(self, texts: list[str], source_language: str, target_language: str,
                  context: list[tuple[str, str]] | None = None) -> list[str]:
        context = (context or [])[-self.context_lines:]
        if self.style == "translategemma":
            result = self._translategemma_block(texts, context, source_language, target_language)
        else:
            result = self._instruct_block(texts, context, source_language, target_language)
        for i, text in enumerate(texts):
            if not result.get(i):
                log.info("Line %d missing from block output; translating it alone", i)
                result[i] = self._single(text, source_language, target_language)
        return [result[i] for i in range(len(texts))]

    # -- styles ------------------------------------------------------------------------------

    def _translategemma_block(self, texts, context, source, target) -> dict[int, str]:
        lines = [src for src, _ in context] + list(texts)
        block = "\n".join(f"[{i + 1}] {line}" for i, line in enumerate(lines))
        output = self.client.chat(self.model, [{"role": "user", "content": translategemma_prompt(block, source, target)}])
        parsed: dict[int, str] = {}
        for raw in output.splitlines():
            match = _MARKER.match(raw)
            if match:
                index = int(match.group(1)) - 1 - len(context)
                if 0 <= index < len(texts) and index not in parsed:
                    parsed[index] = clean_line(match.group(2))
        return parsed

    def _instruct_block(self, texts, context, source, target) -> dict[int, str]:
        system = INSTRUCT_GUIDELINES.format(src=english_name(source), tgt=english_name(target),
                                            style=style_guides.local_style(target))
        payload = {
            "previous_lines": [{"source": s, "translation": t} for s, t in context],
            "lines": [{"id": i + 1, "text": t} for i, t in enumerate(texts)],
        }
        output = self.client.chat(self.model, [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ], fmt=_SCHEMA)
        parsed: dict[int, str] = {}
        try:
            items = json.loads(output).get("translations", [])
        except (ValueError, AttributeError):
            log.warning("Model returned invalid JSON; falling back to line-by-line")
            return parsed
        for item in items:
            try:
                index = int(item["id"]) - 1
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= index < len(texts) and index not in parsed:
                parsed[index] = clean_line(str(item.get("text", "")))
        return parsed

    def _single(self, text: str, source: str, target: str) -> str:
        if self.style == "translategemma":
            prompt = translategemma_prompt(text, source, target)
            return clean_line(self.client.chat(self.model, [{"role": "user", "content": prompt}]))
        parsed = self._instruct_block([text], [], source, target)
        return parsed.get(0, "")
