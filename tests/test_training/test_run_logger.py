"""Tests for RunLogger (DuckDB-backed experiment tracker)."""

import pytest
import torch

from meandre.training.run_logger import RunLogger


def test_start_end_run(tmp_path):
    """Basic run lifecycle."""
    rl = RunLogger(tmp_path / "runs.duckdb")
    run_id = rl.start_run("my_run")
    assert run_id is not None
    assert rl.last_run_id == run_id
    rl.end_run()
    assert rl.last_run_id is None


def test_log_params(tmp_path):
    """Params round-trip."""
    rl = RunLogger(tmp_path / "runs.duckdb")
    rl.start_run("p")
    rl.log_params({"lr": 5e-4, "n_epochs": 100, "tag": "test"})
    run_id = rl.last_run_id
    rl.end_run()

    df = rl.params_df(run_id)
    vals = dict(zip(df["key"], df["value"]))
    assert vals["lr"] == str(5e-4)
    assert vals["n_epochs"] == "100"
    assert vals["tag"] == "test"


def test_log_metric(tmp_path):
    """Single metric logging."""
    rl = RunLogger(tmp_path / "runs.duckdb")
    rl.start_run("m")
    rl.log_metric("val_nse", 0.42, step=5)
    run_id = rl.last_run_id
    rl.end_run()

    df = rl.metrics_df(run_id, key="val_nse")
    assert len(df) == 1
    assert abs(df["value"].iloc[0] - 0.42) < 1e-6
    assert df["step"].iloc[0] == 5


def test_log_metrics_multiple_steps(tmp_path):
    """log_metrics over several epochs."""
    rl = RunLogger(tmp_path / "runs.duckdb")
    rl.start_run("multi")
    for epoch in range(5):
        rl.log_metrics({"loss": 1.0 - epoch * 0.1, "nse": epoch * 0.05}, step=epoch)
    run_id = rl.last_run_id
    rl.end_run()

    df = rl.metrics_df(run_id, key="loss")
    assert len(df) == 5
    assert df["step"].tolist() == list(range(5))


def test_runs_df(tmp_path):
    """runs_df returns all runs."""
    rl = RunLogger(tmp_path / "runs.duckdb")
    rl.start_run("run_a"); rl.end_run()
    rl.start_run("run_b"); rl.end_run()

    df = rl.runs_df()
    assert len(df) == 2
    assert set(df["name"]) == {"run_a", "run_b"}
    assert all(df["status"] == "FINISHED")


def test_best_run(tmp_path):
    """best_run returns the run with highest metric."""
    rl = RunLogger(tmp_path / "runs.duckdb")

    rl.start_run("low"); rl.log_metric("nse", 0.3, 0); rl.end_run()
    rl.start_run("high"); rl.log_metric("nse", 0.8, 0); rl.end_run()
    rl.start_run("mid"); rl.log_metric("nse", 0.5, 0); rl.end_run()

    best = rl.best_run("nse", mode="max")
    assert best["name"] == "high"
    assert abs(best["best_value"] - 0.8) < 1e-6

    worst = rl.best_run("nse", mode="min")
    assert worst["name"] == "low"


def test_best_run_missing_metric(tmp_path):
    """best_run raises KeyError for unknown metric."""
    rl = RunLogger(tmp_path / "runs.duckdb")
    rl.start_run("r"); rl.end_run()
    with pytest.raises(KeyError):
        rl.best_run("nonexistent_metric")


def test_log_without_run_raises(tmp_path):
    """Logging without start_run raises RuntimeError."""
    rl = RunLogger(tmp_path / "runs.duckdb")
    with pytest.raises(RuntimeError, match="No active run"):
        rl.log_metric("x", 1.0)


def test_file_created(tmp_path):
    """DuckDB file is created on init."""
    db = tmp_path / "sub" / "runs.duckdb"
    RunLogger(db)
    assert db.exists()


def test_multiple_instances_same_file(tmp_path):
    """Two RunLogger instances on the same file work correctly."""
    db = tmp_path / "shared.duckdb"
    r1 = RunLogger(db)
    r1.start_run("first"); r1.log_metric("loss", 1.0, 0); r1.end_run()

    r2 = RunLogger(db)
    r2.start_run("second"); r2.log_metric("loss", 0.5, 0); r2.end_run()

    df = r1.runs_df()
    assert len(df) == 2
