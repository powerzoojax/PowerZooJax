import json
from pathlib import Path

from benchmarks.common.experiment_ops import benchmark_preflight, benchmark_preflight_main


def _write_min_task(root: Path, *, doc_text: str = "iid", summary_split: str = "iid") -> Path:
    task_dir = root / "unit_task"
    (task_dir / "configs").mkdir(parents=True)
    (task_dir / "results" / "summary").mkdir(parents=True)
    (task_dir / "results").mkdir(exist_ok=True)
    (task_dir / "configs" / "task.yaml").write_text(
        "\n".join(
            [
                "task: unit_task",
                "eval_splits: [iid]",
                "primary_split: iid",
                "seeds: [0, 1, 2]",
                "num_envs: 1",
            ]
        ),
        encoding="utf-8",
    )
    (task_dir / "configs" / "provenance.json").write_text("{}", encoding="utf-8")
    (task_dir / "results" / "manifest.json").write_text("[]", encoding="utf-8")
    (task_dir / "results" / "summary" / "latest.json").write_text(
        json.dumps(
            {
                "task": "unit_task",
                "rows": [{"algo": "ppo", "split": summary_split, "n_seeds": 1}],
                "protocol_status": {"current_campaign_submission_ready": False},
            }
        ),
        encoding="utf-8",
    )
    (task_dir / "README.md").write_text(doc_text, encoding="utf-8")
    return task_dir


def test_benchmark_preflight_passes_minimal_consistent_task(tmp_path):
    task_dir = _write_min_task(tmp_path)

    report = benchmark_preflight(task="unit_task", task_dir=task_dir)

    assert report["ok"] is True
    assert report["errors"] == []


def test_benchmark_preflight_flags_doc_split_conflict(tmp_path):
    task_dir = _write_min_task(tmp_path, doc_text="mentions summer_ood")

    report = benchmark_preflight(task="unit_task", task_dir=task_dir)

    assert report["ok"] is False
    assert any("doc_mentions_unconfigured_split" in err for err in report["errors"])


def test_benchmark_preflight_enforce_returns_nonzero(tmp_path, capsys):
    task_dir = _write_min_task(tmp_path, summary_split="summer_ood")

    code = benchmark_preflight_main(
        ["--task", "unit_task", "--task-dir", str(task_dir), "--enforce"]
    )

    assert code == 2
    out = capsys.readouterr().out
    assert "summary_split_not_configured" in out

