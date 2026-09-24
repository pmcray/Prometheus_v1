import json

import pytest

from prometheus.common.eventlog import EventLog, read_log, verify_log


def test_append_and_verify(tmp_path):
    log = EventLog(tmp_path / "agent.jsonl")
    for i in range(5):
        log.append("episode", {"i": i})
    assert len(log) == 5
    result = verify_log(log.path)
    assert result.ok and result.records == 5


def test_reopen_resumes_chain(tmp_path):
    p = tmp_path / "agent.jsonl"
    log = EventLog(p)
    log.append("a", {})
    head = log.head_hash
    log2 = EventLog(p)
    rec = log2.append("b", {})
    assert rec["seq"] == 1 and rec["prev_hash"] == head
    assert verify_log(p).ok


def _rewrite(path, fn):
    lines = path.read_text().splitlines()
    lines = fn(lines)
    path.write_text("\n".join(lines) + "\n")


def test_detects_edited_payload(tmp_path):
    log = EventLog(tmp_path / "l.jsonl")
    for i in range(3):
        log.append("score", {"value": i})

    def edit(lines):
        rec = json.loads(lines[1])
        rec["payload"]["value"] = 99
        lines[1] = json.dumps(rec)
        return lines

    _rewrite(log.path, edit)
    r = verify_log(log.path)
    assert not r.ok and r.first_bad_seq == 1


def test_detects_deleted_record(tmp_path):
    log = EventLog(tmp_path / "l.jsonl")
    for i in range(3):
        log.append("x", {"i": i})
    _rewrite(log.path, lambda lines: [lines[0], lines[2]])
    assert not verify_log(log.path).ok


def test_detects_rehashed_forgery(tmp_path):
    """Even recomputing the edited record's own hash breaks the next link."""
    from prometheus.common.eventlog import _record_hash

    log = EventLog(tmp_path / "l.jsonl")
    for i in range(3):
        log.append("x", {"i": i})

    def forge(lines):
        rec = json.loads(lines[1])
        rec["payload"]["i"] = 42
        rec["hash"] = _record_hash(rec)
        lines[1] = json.dumps(rec)
        return lines

    _rewrite(log.path, forge)
    r = verify_log(log.path)
    assert not r.ok and r.first_bad_seq == 2


def test_refuses_to_append_to_corrupted_log(tmp_path):
    log = EventLog(tmp_path / "l.jsonl")
    log.append("x", {})
    log.append("y", {})
    _rewrite(log.path, lambda lines: [lines[1]])
    with pytest.raises(ValueError):
        EventLog(log.path)
