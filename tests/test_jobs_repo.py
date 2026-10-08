import json

from app.database.jobs import JobsRepo


def test_job_lifecycle(db):
    repo = JobsRepo(db)
    job_id = repo.create(input_type="file", input_value="C:/v.mkv", source_language="tr",
                         target_language="ar", mode="balanced", output_dir=None)
    repo.update(job_id, status="running", input_hash="abc")
    repo.record_stage(job_id, "audio", "running", "k1", None)
    repo.record_stage(job_id, "audio", "completed", None, None)
    stages = repo.stages(job_id)
    assert stages["audio"]["status"] == "completed"
    assert stages["audio"]["cache_key"] == "k1"              # kept when not re-supplied
    assert stages["audio"]["started_at"] and stages["audio"]["finished_at"]
    repo.update(job_id, status="completed", outputs={"srt": "x.srt"}, warnings=["w"], stats={"a": 1})
    job = repo.get(job_id)
    assert job["status"] == "completed" and json.loads(job["output_paths"]) == {"srt": "x.srt"}
    assert repo.recent()[0]["id"] == job_id


def test_mark_interrupted(db):
    repo = JobsRepo(db)
    a = repo.create(input_type="file", input_value="a", source_language="tr", target_language="ar",
                    mode="fast", output_dir=None)
    b = repo.create(input_type="file", input_value="b", source_language="tr", target_language="ar",
                    mode="fast", output_dir=None)
    repo.update(a, status="running")
    repo.update(b, status="completed")
    assert repo.mark_interrupted() == 1
    assert repo.get(a)["status"] == "paused" and repo.get(b)["status"] == "completed"
