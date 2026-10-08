"""Review session for one episode folder: load results, edit/approve lines, save the final subtitle.

Edits are kept in work/review.json (so the original AI translation is never lost) and applied on save:
the final .<lang>.srt, MasterTranscript.json and ReviewRequired.txt are rewritten.
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.core import finalize
from app.core.audio_processor import SAMPLE_RATE, load_wav
from app.core.exporter import safe_basename, write_srt
from app.utils.atomic import atomic_write_json, read_json

VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v")
REVIEW_FILE = "review.json"
# Extra audio kept around a re-transcribed cue: Whisper needs a little run-up and decay to hear it.
SPAN_MARGIN = 0.3

log = logging.getLogger(__name__)


class ReviewError(RuntimeError):
    pass


class ReviewSession:
    def __init__(self, episode_dir: Path):
        self.episode_dir = Path(episode_dir)
        self.work_dir = self.episode_dir / "work"
        master = self.work_dir / "MasterTranscript.json"
        if not master.is_file():
            raise ReviewError(f"No work/MasterTranscript.json in {self.episode_dir}")
        self.doc = read_json(master)
        self.units: list[dict] = self.doc.get("translation_units", [])
        if not self.units:
            raise ReviewError("The transcript has no translated lines")
        self.target_language = self.doc["job"]["target_language"]
        self.source_language = self.doc["language"]["code"]
        self.segments = self.doc.get("segments", [])
        self.target_srt = self._find_target_srt()
        self.video_path = self._find_video()
        self._deleted_ids: set[int] = set()
        self._apply_saved_edits()
        self._finalize()
        self.dirty = False

    def _finalize(self):
        media = self.doc.get("media") or {}
        return finalize.finalize_units(self.units, self.segments, self.target_language,
                                       fps=media.get("fps"), shots=media.get("shot_changes"))

    # -- discovery -----------------------------------------------------------------------------

    def _find_target_srt(self) -> Path:
        candidates = sorted(self.episode_dir.glob(f"*.{self.target_language}.srt"))
        if candidates:
            return candidates[0]
        title = (self.doc.get("media_info") or {}).get("title") or self.episode_dir.name
        return self.episode_dir / f"{safe_basename(str(title))}.{self.target_language}.srt"

    def _find_video(self) -> Path | None:
        base = self.target_srt.name[: -len(f".{self.target_language}.srt")]
        for ext in VIDEO_EXTENSIONS:
            candidate = self.episode_dir / f"{base}{ext}"
            if candidate.is_file():
                return candidate
        others = [p for p in self.episode_dir.iterdir()
                  if p.suffix.lower() in VIDEO_EXTENSIONS and not p.name.endswith(".subtitled.mp4")]
        return others[0] if others else None

    def _apply_saved_edits(self) -> None:
        path = self.work_dir / REVIEW_FILE
        if not path.is_file():
            return
        saved = read_json(path)
        deleted = set(saved.get("deleted", []))
        if deleted:
            self.units = [u for u in self.units if u["id"] not in deleted]
            self._deleted_ids = deleted
        units_state = saved.get("units", {})
        for unit in self.units:
            state = units_state.get(str(unit["id"]))
            if state:
                if state.get("text") is not None:
                    unit["reviewed_text"] = state["text"]
                unit["approved"] = bool(state.get("approved"))

    # -- queries --------------------------------------------------------------------------------

    def unit(self, unit_id: int) -> dict:
        if 0 <= unit_id < len(self.units) and self.units[unit_id]["id"] == unit_id:
            return self.units[unit_id]
        return next(u for u in self.units if u["id"] == unit_id)

    def needs_review(self, unit: dict) -> bool:
        return finalize.needs_review(unit)

    def counts(self) -> dict:
        return {"total": len(self.units),
                "needs_review": sum(1 for u in self.units if finalize.needs_review(u)),
                "approved": sum(1 for u in self.units if u.get("approved")),
                "edited": sum(1 for u in self.units if u.get("reviewed_text") is not None)}

    def unit_at(self, seconds: float) -> dict | None:
        for unit in self.units:
            if unit["start"] <= seconds <= unit["end"]:
                return unit
        return None

    def current_text(self, unit: dict) -> str:
        return unit.get("final_text", unit.get("reviewed_text") or unit["translation"])

    # -- edits ------------------------------------------------------------------------------------

    def set_text(self, unit_id: int, text: str) -> None:
        unit = self.unit(unit_id)
        clean = text.strip()
        if clean == unit["translation"].strip():
            unit.pop("reviewed_text", None)
        else:
            unit["reviewed_text"] = clean
        self._refresh()

    def delete_unit(self, unit_id: int) -> None:
        unit = self.unit(unit_id)
        self.units.remove(unit)
        self._deleted_ids.add(unit_id)
        self._refresh()

    def split_unit(self, unit_id: int, split_time: float | None = None,
                   split_text: tuple[str, str] | None = None) -> tuple[dict, dict]:
        idx = next(i for i, u in enumerate(self.units) if u["id"] == unit_id)
        unit = self.units[idx]
        if split_time is None or not (unit["start"] < split_time < unit["end"]):
            split_time = round(unit["start"] + (unit["end"] - unit["start"]) / 2.0, 3)
        if split_text is not None:
            t1, t2 = split_text
        else:
            cur = self.current_text(unit)
            lines = cur.split("\n", 1)
            if len(lines) == 2 and lines[0].strip() and lines[1].strip():
                t1, t2 = lines[0].strip(), lines[1].strip()
            else:
                words = cur.split()
                if len(words) > 1:
                    half = len(words) // 2
                    t1, t2 = " ".join(words[:half]), " ".join(words[half:])
                else:
                    t1, t2 = cur, cur

        src = unit.get("text", "")
        src_lines = src.split("\n", 1)
        if len(src_lines) == 2 and src_lines[0].strip() and src_lines[1].strip():
            s1, s2 = src_lines[0].strip(), src_lines[1].strip()
        else:
            src_words = src.split()
            if len(src_words) > 1:
                shalf = len(src_words) // 2
                s1, s2 = " ".join(src_words[:shalf]), " ".join(src_words[shalf:])
            else:
                s1, s2 = src, src

        # Never reuse the id of a deleted or merged cue: review.json would drop the new cue on reload.
        new_id = max([u["id"] for u in self.units] + list(self._deleted_ids), default=0) + 1
        old_end = unit["end"]
        unit["end"] = split_time
        unit["text"] = s1
        unit["reviewed_text"] = t1
        unit["translation"] = t1
        unit["approved"] = False

        unit2 = {
            **unit,
            "id": new_id,
            "start": split_time,
            "end": old_end,
            "text": s2,
            "translation": t2,
            "reviewed_text": t2,
            "approved": False,
        }
        self.units.insert(idx + 1, unit2)
        self._refresh()
        return unit, unit2

    def merge_units(self, unit_id1: int, unit_id2: int) -> dict:
        idx1 = next(i for i, u in enumerate(self.units) if u["id"] == unit_id1)
        idx2 = next(i for i, u in enumerate(self.units) if u["id"] == unit_id2)
        u1, u2 = self.units[idx1], self.units[idx2]
        u1["start"] = min(u1["start"], u2["start"])
        u1["end"] = max(u1["end"], u2["end"])
        u1["text"] = f"{u1.get('text', '')} {u2.get('text', '')}".strip()
        t1, t2 = self.current_text(u1), self.current_text(u2)
        u1["reviewed_text"] = f"{t1} {t2}".strip()
        u1["translation"] = f"{u1.get('translation', '')} {u2.get('translation', '')}".strip()
        u1["segment_ids"] = sorted(set(u1.get("segment_ids", []) + u2.get("segment_ids", [])))
        u1["approved"] = False
        self.units.remove(u2)
        self._deleted_ids.add(u2["id"])
        self._refresh()
        return u1

    def shift_cues(self, delta: float, unit_ids: list[int] | None = None) -> None:
        targets = self.units if unit_ids is None else [u for u in self.units if u["id"] in set(unit_ids)]
        for u in targets:
            u["start"] = max(0.0, round(u["start"] + delta, 3))
            u["end"] = max(round(u["start"] + 0.1, 3), round(u["end"] + delta, 3))
        self._refresh()

    def find_replace_episode(self, find_text: str, replace_text: str, match_case: bool = False) -> int:
        if not find_text:
            return 0
        import re
        flags = 0 if match_case else re.IGNORECASE
        pattern = re.compile(re.escape(find_text), flags)
        count = 0
        for u in self.units:
            txt = self.current_text(u)
            if pattern.search(txt):
                new_txt, n = pattern.subn(replace_text, txt)
                if n > 0:
                    self.set_text(u["id"], new_txt)
                    count += n
        return count

    def find_replace_series(self, find_text: str, replace_text: str, match_case: bool = False) -> dict[str, int]:
        parent = self.episode_dir.parent
        results: dict[str, int] = {}
        for ep_dir in sorted(parent.iterdir()):
            if not ep_dir.is_dir() or not (ep_dir / "work" / "MasterTranscript.json").is_file():
                continue
            if ep_dir.resolve() == self.episode_dir.resolve():
                n = self.find_replace_episode(find_text, replace_text, match_case)
                if n > 0:
                    results[ep_dir.name] = n
            else:
                try:
                    other_sess = ReviewSession(ep_dir)
                    n = other_sess.find_replace_episode(find_text, replace_text, match_case)
                    if n > 0:
                        other_sess.save()
                        results[ep_dir.name] = n
                except Exception as exc:
                    log.warning("Could not process episode %s: %s", ep_dir.name, exc)
        return results

    def retranscribe_unit(self, unit_id: int, asr_factory=None) -> str:
        """Re-transcribe one cue's span with the real ASR engine and return the new source text.

        The span [start - SPAN_MARGIN, end + SPAN_MARGIN], clamped to the job's audio, is decoded and
        handed to the engine as an array; only the cue's source text is replaced (the translation is not
        redone). Returns "" when the span holds no speech, in which case the old text is kept. Raises
        ReviewError when no engine can be built or the job's audio is missing.
        """
        unit = self.unit(unit_id)
        audio_path = self.work_dir / "audio.wav"
        if not audio_path.is_file():
            raise ReviewError(f"Missing {audio_path}: re-transcription needs the audio of the job")
        engine = asr_factory() if callable(asr_factory) else asr_factory
        if engine is None:
            raise ReviewError("No speech recognition engine could be loaded")
        try:
            audio = load_wav(audio_path)
            first = max(0, round((unit["start"] - SPAN_MARGIN) * SAMPLE_RATE))
            last = min(len(audio), round((unit["end"] + SPAN_MARGIN) * SAMPLE_RATE))
            if last <= first:
                # A cue shifted past the end of the audio has nothing to decode: treat it as no speech.
                log.warning("Re-transcription of unit %s has an empty span; keeping the old text", unit_id)
                return ""
            offset = first / SAMPLE_RATE
            new_text = " ".join(segment.text.strip()
                                for segment in engine.transcribe(audio[first:last], self.source_language,
                                                                 offset, None)
                                if segment.text.strip())
        finally:
            # The design GPU has only 4 GB: never hold an ASR engine while the user is just editing.
            release = getattr(asr_factory, "release", None)
            if callable(release):
                release()
        if not new_text:
            log.warning("Re-transcription of unit %s found no speech; keeping the old text", unit_id)
            return ""
        unit["text"] = new_text
        self._refresh()
        return new_text

    def approve(self, unit_id: int, approved: bool = True) -> None:
        self.unit(unit_id)["approved"] = approved
        self._refresh()

    def _refresh(self) -> None:
        self._finalize()
        self.dirty = True

    def save(self) -> Path:
        state = {str(u["id"]): {"text": u.get("reviewed_text"), "approved": bool(u.get("approved"))}
                 for u in self.units if u.get("reviewed_text") is not None or u.get("approved")}
        payload = {"units": state}
        if self._deleted_ids:
            payload["deleted"] = sorted(self._deleted_ids)
        atomic_write_json(self.work_dir / REVIEW_FILE, payload)
        cues = self._finalize()
        write_srt(cues, self.target_srt)
        self.doc["translation_units"] = self.units
        atomic_write_json(self.work_dir / "MasterTranscript.json", self.doc)
        finalize.write_review_file(self.work_dir / "ReviewRequired.txt", self.units)
        self.dirty = False
        return self.target_srt
