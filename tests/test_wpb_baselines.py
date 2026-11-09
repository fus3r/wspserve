"""Report-command regression, with a tiny local scorer and no benchmark download."""
import hashlib
import json
import sys

from wpb import baselines


def test_official_report_command_writes_provenance_and_preserves_existing_rows(tmp_path, monkeypatch):
    runtime = tmp_path / "snapshot" / "runtime"
    runtime.mkdir(parents=True)
    (runtime / "verify.py").write_text(
        "def grade_answer(game, source, answer, episodes):\n"
        "    return 0.5, {'raw_score': 5.0, 'score_cap': 10.0, 'mean_episode_length': 2.0}\n"
    )
    task = {"name": "freeway", "game": "freeway", "split": "test", "source": "original", "episodes": 1000}
    (runtime / "tasks.jsonl").write_text(json.dumps(task) + "\n")
    source = {"source_revision": "fixture-source", "game_sha256": {},
              "verify_sha256": baselines.sha256(runtime / "verify.py"),
              "tasks_sha256": baselines.sha256(runtime / "tasks.jsonl")}
    (runtime / "source.json").write_text(json.dumps(source))
    monkeypatch.setattr(baselines, "SOURCE_SHA256", hashlib.sha256((runtime / "source.json").read_bytes()).hexdigest())

    answers = tmp_path / "answers"
    answers.mkdir()
    (answers / "freeway__rule.txt").write_text("```edits\nb3[2] = 20.0\n```\n")
    output = tmp_path / "output" / "report.json"
    output.parent.mkdir()
    previous = {"task": "retained", "policy": "zero", "episodes": 1000, "raw_score": 3.0}
    output.write_text(json.dumps({"rows": [previous]}))
    monkeypatch.setattr(sys, "argv", ["baselines.py", "--wpb", str(runtime.parent),
                                    "--answers", str(answers), "--episodes", "1", "--workers", "1",
                                    "--out", str(output)])

    baselines.main()

    report = json.loads(output.read_text())
    assert report["revision"] == "cb8d15ea0d2c1970f6d8b590c7b7fcb774acd426"
    assert report["runtime_source_revision"] == "fixture-source"
    assert previous in report["rows"]
    scored = [row for row in report["rows"] if row["task"] == "freeway"]
    assert {row["policy"] for row in scored} == {"zero", "rule", *(f"const_{i}" for i in range(6))}
    assert all(row["episodes"] == 1 and row["raw_score"] == 5.0 and row["error"] is None for row in scored)
