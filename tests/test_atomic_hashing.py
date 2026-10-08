import json

from app.utils.atomic import JsonlWriter, atomic_write_json, load_jsonl, read_json
from app.utils.hashing import file_fingerprint, stable_hash


def test_atomic_write_replaces_and_leaves_no_temp(tmp_path):
    target = tmp_path / "a" / "x.json"
    atomic_write_json(target, {"v": 1})
    atomic_write_json(target, {"v": 2})
    assert read_json(target) == {"v": 2}
    assert [p.name for p in target.parent.iterdir()] == ["x.json"]


def test_jsonl_truncates_torn_tail_and_appends_cleanly(tmp_path):
    path = tmp_path / "p.jsonl"
    with JsonlWriter(path) as w:
        w.write({"i": 0})
        w.write({"i": 1})
    with open(path, "ab") as f:
        f.write(b'{"i": 2, "trunc')            # simulated kill mid-write
    assert load_jsonl(path) == [{"i": 0}, {"i": 1}]
    with JsonlWriter(path) as w:
        w.write({"i": 2})
    assert load_jsonl(path) == [{"i": 0}, {"i": 1}, {"i": 2}]


def test_jsonl_stops_at_invalid_line(tmp_path):
    path = tmp_path / "p.jsonl"
    path.write_bytes(b'{"i": 0}\nnot json\n{"i": 2}\n')
    assert load_jsonl(path) == [{"i": 0}]
    assert path.read_bytes() == b'{"i": 0}\n'


def test_load_missing_jsonl(tmp_path):
    assert load_jsonl(tmp_path / "none.jsonl") == []


def test_stable_hash_ignores_key_order():
    assert stable_hash({"a": 1, "b": [1, 2]}) == stable_hash({"b": [1, 2], "a": 1})
    assert stable_hash({"a": 1}) != stable_hash({"a": 2})


def test_file_fingerprint(tmp_path):
    a = tmp_path / "a.bin"
    big = bytes(range(256)) * (40 * 1024)   # 10 MiB, larger than head + tail samples
    a.write_bytes(big)
    b = tmp_path / "b.bin"
    b.write_bytes(big)
    assert file_fingerprint(a) == file_fingerprint(b)
    b.write_bytes(b"x" + big[1:])
    assert file_fingerprint(a) != file_fingerprint(b)
    b.write_bytes(big[:-1] + b"x")
    assert file_fingerprint(a) != file_fingerprint(b)
    b.write_bytes(big + b"x")
    assert file_fingerprint(a) != file_fingerprint(b)
    small = tmp_path / "s.bin"
    small.write_bytes(b"abc")
    assert len(file_fingerprint(small)) == 64
