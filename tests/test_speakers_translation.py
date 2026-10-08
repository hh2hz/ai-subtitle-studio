"""Task 3.2: speakers in the AI translation (D-099)."""

import json

from app.core import brief as br
from app.core.llm_translation import CORRECTOR_SYSTEM, TRANSLATOR_SYSTEM, LlmRefiner
from tests.fake_llm_api import FakeLlmApi
from tests.fakes import FakeAsrEngine, FakeTranslator
from tests.test_brief import _pool
from tests.test_pipeline import _pipeline, _refiner_factory, media  # noqa: F401

UNITS = [{"id": 0, "start": 0.0, "text": "Nevzat abi geldi mi?", "speaker": "S1"},
         {"id": 1, "start": 2.0, "text": "Gelmedi.", "speaker": "S2"},
         {"id": 2, "start": 4.0, "text": "Ben buradayim, sen neredeydin kardesim?", "speaker": "S1"},
         {"id": 3, "start": 6.0, "text": "Yok.", "speaker": None}]


def test_samples_are_grouped_by_voice_longest_first_in_time_order():
    samples = br.speaker_samples(UNITS)
    assert [s["id"] for s in samples] == ["S1", "S2"]
    assert samples[0]["samples"] == ["Nevzat abi geldi mi?", "Ben buradayim, sen neredeydin kardesim?"]
    assert br.speaker_samples([{"id": 0, "start": 0.0, "text": "x"}]) == []


def test_validate_speakers_keeps_known_ids_and_fixes_enums():
    data = {"speakers": [{"id": "S1", "name": "Nevzat", "gender": "male"}, {"id": "S9", "name": "X"},
                         {"id": "S2", "name": "unknown", "gender": "robot"}, "bad"]}
    assert br.validate_speakers(data, {"S1", "S2"}) == {"S1": {"name": "Nevzat", "gender": "male"},
                                                        "S2": {"name": "", "gender": "unknown"}}
    assert br.validate_speakers(None, {"S1"}) == {}


def test_speaker_fields_use_the_name_else_the_voice_id_and_skip_unknown_gender():
    speakers = {"S1": {"name": "Nevzat", "gender": "male"}, "S2": {"name": "", "gender": "unknown"}}
    assert br.speaker_fields("S1", speakers) == {"speaker": "Nevzat", "speaker_gender": "male"}
    assert br.speaker_fields("S2", speakers) == {"speaker": "S2"}
    assert br.speaker_fields("S3", None) == {"speaker": "S3"} and br.speaker_fields(None, speakers) == {}


def test_speaker_map_is_one_request_and_failure_returns_none():
    with FakeLlmApi() as server:
        result = br.build_speaker_map(_pool(server), "tr", None, UNITS, sleep=lambda s: None)
    assert result == {"S1": {"name": "Nevzat", "gender": "male"}, "S2": {"name": "", "gender": "female"}}
    assert len(server.requests) == 1
    sent = json.loads(server.requests[0]["messages"][1]["content"])
    assert [s["id"] for s in sent["speakers"]] == ["S1", "S2"]
    with FakeLlmApi(behaviour="garbage") as bad:
        assert br.build_speaker_map(_pool(bad), "tr", None, UNITS, sleep=lambda s: None) is None
    assert br.build_speaker_map(_pool(), "tr", None, [{"id": 0, "start": 0.0, "text": "x"}]) is None


def test_prompts_tell_the_model_to_use_speaker_gender():
    assert "speaker_gender" in TRANSLATOR_SYSTEM and "speaker_gender" in CORRECTOR_SYSTEM


def test_speaker_fields_reach_the_block_payload():
    with FakeLlmApi() as server:
        refiner = LlmRefiner(_pool(server), "tr", "ar", {"title": "t"}, sleep=lambda s: None, review=False)
        refiner.translate_block(
            [{"id": 0, "source": "Geldi mi?", "draft": "", "speaker": "Elif", "speaker_gender": "female"}],
            [{"source": "Evet.", "translation": "x", "speaker": "S2"}], [], {})
    payload = json.loads(server.requests[-1]["messages"][1]["content"])
    assert payload["lines"][0]["speaker"] == "Elif" and payload["lines"][0]["speaker_gender"] == "female"
    assert payload["previous_lines"][0]["speaker"] == "S2"


class _Voices:
    """A diarizer that gives the first half of the audio to one voice and the rest to another."""

    def __call__(self, audio, windows, partial, progress=None, cancel=None):
        return [(0.0, 10.0, 0), (10.0, 20.0, 1)]


def test_pipeline_sends_speakers_and_caches_the_map(tmp_path, media):  # noqa: F811
    with FakeLlmApi() as server:
        result = _pipeline(tmp_path, media, FakeAsrEngine(), FakeTranslator(), diarizer=_Voices(),
                           refiner_factory=_refiner_factory(server, mode="translate", review=False))[0].run()
        translate_calls = [json.loads(r["messages"][1]["content"]) for r in server.requests
                           if "lines" in json.loads(r["messages"][1]["content"])]
        map_calls = [r for r in server.requests if "match the speakers" in r["messages"][0]["content"]]
    assert len(map_calls) == 1
    first = translate_calls[0]["lines"][0]
    assert first["speaker"] == "Nevzat" and first["speaker_gender"] == "male"
    assert any(l.get("speaker") == "S2" and l.get("speaker_gender") == "female" for c in translate_calls
               for l in c["lines"])
    assert any(p.name.startswith("speakers.") for p in result.job_dir.iterdir())
