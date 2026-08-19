"""Benchmark recording tests — BenchmarkRun.record() and write_results()."""

from __future__ import annotations

import subprocess

from f1_pit_window.monitoring.benchmark import CSV_COLUMNS, BenchmarkRun, _git_commit, write_results


class TestRecord:
    def test_computes_rows_per_second(self):
        run = BenchmarkRun()
        run.record("feature_build", rows_processed=1000, duration_seconds=2.0)
        assert run.rows[-1]["rows_per_second"] == 500.0

    def test_zero_duration_does_not_raise_and_reports_zero_throughput(self):
        run = BenchmarkRun()
        run.record("train", rows_processed=1000, duration_seconds=0.0)
        assert run.rows[-1]["rows_per_second"] == 0.0

    def test_notes_default_to_empty_string(self):
        run = BenchmarkRun()
        run.record("db_load", rows_processed=10, duration_seconds=1.0)
        assert run.rows[-1]["notes"] == ""

    def test_all_stage_rows_in_a_run_share_run_id_and_git_commit(self):
        run = BenchmarkRun()
        run.record("db_load", rows_processed=10, duration_seconds=1.0)
        run.record("feature_build", rows_processed=10, duration_seconds=1.0)
        assert run.rows[0]["run_id"] == run.rows[1]["run_id"] == run.run_id
        assert run.rows[0]["git_commit"] == run.rows[1]["git_commit"] == run.git_commit

    def test_two_runs_get_different_run_ids(self):
        assert BenchmarkRun().run_id != BenchmarkRun().run_id


class TestGitCommit:
    def test_returns_unknown_when_git_fails(self, monkeypatch):
        def _raise(*args, **kwargs):
            raise subprocess.CalledProcessError(1, "git")

        monkeypatch.setattr(subprocess, "check_output", _raise)
        assert _git_commit() == "unknown"

    def test_returns_a_string_in_this_real_repo(self):
        commit = _git_commit()
        assert isinstance(commit, str)
        assert commit != ""


class TestWriteResults:
    def test_first_write_creates_header_plus_data_rows(self, tmp_path):
        run = BenchmarkRun()
        run.record("db_load", rows_processed=100, duration_seconds=1.0)
        run.record("feature_build", rows_processed=100, duration_seconds=0.5)
        output_path = tmp_path / "benchmark_results.csv"

        write_results(run, output_path)

        lines = output_path.read_text().splitlines()
        assert lines[0] == ",".join(CSV_COLUMNS)
        assert len(lines) == 1 + len(run.rows)

    def test_second_write_appends_without_duplicate_header(self, tmp_path):
        output_path = tmp_path / "benchmark_results.csv"
        write_results(BenchmarkRun(), output_path)
        run1_lines = output_path.read_text().splitlines()

        run2 = BenchmarkRun()
        run2.record("train", rows_processed=50, duration_seconds=1.0)
        write_results(run2, output_path)
        run2_lines = output_path.read_text().splitlines()

        assert len(run2_lines) == len(run1_lines) + 1
        assert sum(1 for line in run2_lines if line == ",".join(CSV_COLUMNS)) == 1

    def test_creates_parent_directory_if_missing(self, tmp_path):
        output_path = tmp_path / "nested" / "metrics" / "benchmark_results.csv"
        run = BenchmarkRun()
        run.record("train", rows_processed=1, duration_seconds=1.0)

        write_results(run, output_path)

        assert output_path.exists()

    def test_notes_containing_a_comma_round_trips_correctly(self, tmp_path):
        import csv

        run = BenchmarkRun()
        run.record("train", rows_processed=10, duration_seconds=1.0, notes="train=8,test=2")
        output_path = tmp_path / "benchmark_results.csv"

        write_results(run, output_path)

        with open(output_path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["notes"] == "train=8,test=2"

    def test_run_with_no_recorded_rows_still_creates_header_only_file(self, tmp_path):
        output_path = tmp_path / "benchmark_results.csv"
        write_results(BenchmarkRun(), output_path)
        lines = output_path.read_text().splitlines()
        assert lines == [",".join(CSV_COLUMNS)]
