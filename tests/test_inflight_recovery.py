from __future__ import annotations

import json
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from scrapy.utils.test import get_testenv

if TYPE_CHECKING:
    from tests.mockserver.http import MockServer

SCRIPT = Path(__file__).parent / "inflight_recovery" / "inflight_spider.py"
# Barrier files appear quickly; this only guards against a hung subprocess.
WAIT_TIMEOUT = 15


def _wait_for(
    path: Path,
    proc: subprocess.Popen[Any] | None = None,
    timeout: float = WAIT_TIMEOUT,
) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() > deadline:
            detail = ""
            if proc is not None and proc.poll() is not None:
                detail = "\n" + (proc.stderr.read() if proc.stderr else "")
            pytest.fail(f"Timed out waiting for barrier {path.name}{detail}")
        time.sleep(0.02)


def _run_spider(
    tmp_path: Path,
    mockserver: MockServer,
    *,
    mode: str = "basic",
    kill_stage: str | None = None,
    wait_stage: str | None = None,
    pipeline: bool = False,
    drop_items: bool = False,
    concurrent_requests: int | None = None,
    retry_times: int | None = None,
    extra_settings: dict[str, Any] | None = None,
) -> subprocess.Popen[Any]:
    jobdir = tmp_path / "job"
    barrier_dir = tmp_path / "barriers"
    barrier_dir.mkdir(parents=True, exist_ok=True)
    item_log = tmp_path / "items.jsonl"
    config: dict[str, Any] = {
        "mockserver": mockserver.url("/"),
        "jobdir": str(jobdir),
        "barrier_dir": str(barrier_dir),
        "item_log": str(item_log),
        "mode": mode,
        "kill_stage": kill_stage,
        "wait_stage": wait_stage,
        "pipeline": pipeline,
        "drop_items": drop_items,
    }
    if concurrent_requests is not None:
        config["concurrent_requests"] = concurrent_requests
    if retry_times is not None:
        config["retry_times"] = retry_times
    config.update(extra_settings or {})
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return subprocess.Popen(
        [sys.executable, str(SCRIPT), str(config_path)],
        env=get_testenv(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _kill(proc: subprocess.Popen[Any]) -> None:
    proc.send_signal(signal.SIGKILL)
    proc.wait(timeout=WAIT_TIMEOUT)
    # The process can never flush or close its pipes itself.
    if proc.stdout is not None:
        proc.stdout.close()
    if proc.stderr is not None:
        proc.stderr.close()


def _finish(proc: subprocess.Popen[Any]) -> tuple[int, str]:
    _, stderr = proc.communicate(timeout=WAIT_TIMEOUT)
    assert proc.returncode is not None
    return proc.returncode, stderr


def _inflight_rows(jobdir: Path) -> int:
    path = jobdir / "requests.inflight.db"
    if not path.exists() and not path.with_name(path.name + "-wal").exists():
        return 0
    db = sqlite3.connect(path)
    try:
        # The table may not be visible yet from another connection mid-write.
        if not db.execute(
            "SELECT 1 FROM sqlite_schema WHERE name = 'inflight'"
        ).fetchone():
            return 0
        row = db.execute("SELECT COUNT(*) FROM inflight").fetchone()
        return int(row[0]) if row is not None else 0
    finally:
        db.close()


def _read_items(item_log: Path) -> list[dict[str, Any]]:
    if not item_log.exists():
        return []
    return [
        json.loads(line) for line in item_log.read_text(encoding="utf-8").splitlines()
    ]


class TestInflightRecovery:
    def test_kill_during_download(self, tmp_path: Path, mockserver: MockServer) -> None:
        proc = _run_spider(tmp_path, mockserver)
        # The first request targets a 2-second delay endpoint, so once its
        # in-flight row exists while the response has not arrived, the request
        # is provably still downloading.
        jobdir = tmp_path / "job"
        deadline = time.monotonic() + WAIT_TIMEOUT
        while time.monotonic() < deadline:
            if _inflight_rows(jobdir):
                break
            time.sleep(0.02)
        else:
            pytest.fail("No in-flight record appeared")
        # Give the row a moment to be committed while the download is still
        # running; the response barrier cannot exist before the 2s delay ends.
        assert not (tmp_path / "barriers" / "callback_entered_main").exists()
        _kill(proc)
        assert _inflight_rows(jobdir) == 1

        proc = _run_spider(tmp_path, mockserver, pipeline=True)
        _, stderr = _finish(proc)
        assert proc.returncode == 0, stderr
        assert "Spider closed (finished)" in stderr
        assert "Resuming 1 in-flight request" in stderr
        assert _inflight_rows(jobdir) == 0
        items = _read_items(tmp_path / "items.jsonl")
        assert len(items) == 1

    def test_pending_queue_survives_alongside_inflight(
        self, tmp_path: Path, mockserver: MockServer
    ) -> None:
        # With concurrency limited to one, a second request stays in the
        # downloader or scheduler while the first one is in flight. After a
        # restart every request is processed exactly once, whether it came
        # from the in-flight store or from a regular queue.
        proc = _run_spider(
            tmp_path,
            mockserver,
            pipeline=True,
            extra_settings={"with_pending": True},
        )
        jobdir = tmp_path / "job"
        deadline = time.monotonic() + WAIT_TIMEOUT
        while time.monotonic() < deadline:
            if _inflight_rows(jobdir):
                break
            time.sleep(0.02)
        else:
            pytest.fail("No in-flight record appeared")
        _kill(proc)

        proc = _run_spider(
            tmp_path,
            mockserver,
            pipeline=True,
            extra_settings={"with_pending": True},
        )
        _, stderr = _finish(proc)
        assert proc.returncode == 0, stderr
        assert "Spider closed (finished)" in stderr
        assert "Resuming" in stderr
        urls = {item["url"] for item in _read_items(tmp_path / "items.jsonl")}
        assert len(urls) == 2

    def test_kill_during_callback(self, tmp_path: Path, mockserver: MockServer) -> None:
        proc = _run_spider(
            tmp_path, mockserver, kill_stage="callback", wait_stage="callback"
        )
        _wait_for(tmp_path / "barriers" / "callback_entered_main", proc)
        _kill(proc)

        proc = _run_spider(tmp_path, mockserver, pipeline=True)
        _, stderr = _finish(proc)
        assert proc.returncode == 0, stderr
        assert "Resuming 1 in-flight request" in stderr
        assert len(_read_items(tmp_path / "items.jsonl")) == 1

    def test_kill_during_pipeline(self, tmp_path: Path, mockserver: MockServer) -> None:
        proc = _run_spider(
            tmp_path,
            mockserver,
            kill_stage="pipeline",
            pipeline=True,
            wait_stage="pipeline",
        )
        _wait_for(tmp_path / "barriers" / "pipeline_entered", proc)
        _kill(proc)

        proc = _run_spider(tmp_path, mockserver, pipeline=True)
        _, stderr = _finish(proc)
        assert proc.returncode == 0, stderr
        assert "Resuming 1 in-flight request" in stderr
        assert len(_read_items(tmp_path / "items.jsonl")) == 1

    def test_completed_request_not_recovered(
        self, tmp_path: Path, mockserver: MockServer
    ) -> None:
        proc = _run_spider(tmp_path, mockserver, pipeline=True)
        _, stderr = _finish(proc)
        assert proc.returncode == 0, stderr
        assert "Resuming" not in stderr
        assert _inflight_rows(tmp_path / "job") == 0
        assert len(_read_items(tmp_path / "items.jsonl")) == 1

    def test_duplicate_inflight_instances(
        self, tmp_path: Path, mockserver: MockServer
    ) -> None:
        # Two identical dont_filter requests must be tracked and recovered
        # independently, never as one fingerprinted unit.
        proc = _run_spider(
            tmp_path,
            mockserver,
            mode="duplicate",
            concurrent_requests=2,
            pipeline=True,
            wait_stage="pipeline",
        )
        _wait_for(tmp_path / "barriers" / "pipeline_entered", proc)
        jobdir = tmp_path / "job"
        deadline = time.monotonic() + WAIT_TIMEOUT
        while time.monotonic() < deadline and _inflight_rows(jobdir) < 2:
            time.sleep(0.02)
        assert _inflight_rows(jobdir) == 2
        _kill(proc)

        proc = _run_spider(tmp_path, mockserver, mode="duplicate", pipeline=True)
        _, stderr = _finish(proc)
        assert proc.returncode == 0, stderr
        assert "Resuming 2 in-flight requests" in stderr
        items = _read_items(tmp_path / "items.jsonl")
        # Both instances were recovered independently (not deduplicated by
        # fingerprint); an item already written before the kill may repeat.
        assert len(items) >= 2
        assert {item["kind"] for item in items} == {"duplicate"}

    def test_redirect_handoff(self, tmp_path: Path, mockserver: MockServer) -> None:
        # Kill right as the redirect replacement request is being scheduled:
        # whichever side survives, the chain must continue on restart.
        proc = _run_spider(tmp_path, mockserver, mode="redirect")
        _wait_for(tmp_path / "barriers" / "callback_entered_main", proc)
        _kill(proc)

        proc = _run_spider(tmp_path, mockserver, mode="redirect")
        _, stderr = _finish(proc)
        assert proc.returncode == 0, stderr
        assert "Spider closed (finished)" in stderr

    def test_retry_handoff(self, tmp_path: Path, mockserver: MockServer) -> None:
        proc = _run_spider(
            tmp_path,
            mockserver,
            mode="retry",
            retry_times=5,
            kill_stage="errback",
        )
        _wait_for(tmp_path / "barriers" / "errback_entered", proc)
        _kill(proc)

        proc = _run_spider(tmp_path, mockserver, mode="retry", retry_times=5)
        _, stderr = _finish(proc)
        assert proc.returncode == 0, stderr
        assert "Spider closed (finished)" in stderr

    def test_consecutive_restarts(self, tmp_path: Path, mockserver: MockServer) -> None:
        for _ in range(3):
            proc = _run_spider(
                tmp_path,
                mockserver,
                kill_stage="callback",
                wait_stage="callback",
            )
            _wait_for(tmp_path / "barriers" / "callback_entered_main", proc)
            _kill(proc)
            # Fresh barriers for the next attempt.
            for barrier in (tmp_path / "barriers").iterdir():
                barrier.unlink()

        proc = _run_spider(tmp_path, mockserver, pipeline=True)
        _, stderr = _finish(proc)
        assert proc.returncode == 0, stderr
        assert "Resuming 1 in-flight request" in stderr
        assert len(_read_items(tmp_path / "items.jsonl")) == 1

    def test_priority_preserved(self, tmp_path: Path, mockserver: MockServer) -> None:
        proc = _run_spider(
            tmp_path,
            mockserver,
            mode="priority",
            concurrent_requests=1,
            kill_stage="callback",
            wait_stage="callback",
        )
        # The first dequeued request is one of the two priority-0 requests.
        _wait_for(tmp_path / "barriers" / "callback_entered_first", proc)
        _kill(proc)

        proc = _run_spider(
            tmp_path,
            mockserver,
            mode="priority",
            concurrent_requests=1,
            pipeline=True,
        )
        _, stderr = _finish(proc)
        assert proc.returncode == 0, stderr
        items = _read_items(tmp_path / "items.jsonl")
        urls = [item["url"] for item in items]
        # The higher-priority request is processed before any lower-priority
        # one, including the recovered request.
        assert urls[0].endswith("name=high")
        assert {url.split("name=")[1] for url in urls} == {
            "first",
            "second",
            "high",
        }

    def test_unserializable_request_rejected(
        self, tmp_path: Path, mockserver: MockServer
    ) -> None:
        proc = _run_spider(tmp_path, mockserver, mode="unserializable")
        _, stderr = _finish(proc)
        assert "Rejected unserializable request" in stderr
        assert proc.returncode == 0

    def test_persistence_failure_ends_crawl(
        self, tmp_path: Path, mockserver: MockServer
    ) -> None:
        # A read-only in-flight database makes writing the recovered records
        # fail; the crawl must stop with an error instead of continuing
        # without a working recovery store.
        proc = _run_spider(tmp_path, mockserver, pipeline=True)
        jobdir = tmp_path / "job"
        deadline = time.monotonic() + WAIT_TIMEOUT
        while time.monotonic() < deadline:
            if _inflight_rows(jobdir):
                break
            time.sleep(0.02)
        else:
            pytest.fail("No in-flight record appeared")
        _kill(proc)
        for target in jobdir.glob("requests.inflight.db*"):
            target.chmod(0o400)
        proc = _run_spider(tmp_path, mockserver, pipeline=True)
        _, stderr = _finish(proc)
        # The persistence failure is logged with full diagnostics instead of
        # being swallowed as a successful crawl.
        assert "readonly database" in stderr
        assert "Unhandled error in Deferred" in stderr

    def test_tracking_disabled_with_existing_records_fails(
        self, tmp_path: Path, mockserver: MockServer
    ) -> None:
        proc = _run_spider(tmp_path, mockserver)
        jobdir = tmp_path / "job"
        deadline = time.monotonic() + WAIT_TIMEOUT
        while time.monotonic() < deadline:
            if _inflight_rows(jobdir):
                break
            time.sleep(0.02)
        else:
            pytest.fail("No in-flight record appeared")
        _kill(proc)
        proc = _run_spider(
            tmp_path,
            mockserver,
            extra_settings={"disable_tracking": True},
        )
        _, stderr = _finish(proc)
        assert "JOBDIR_TRACK_INFLIGHT is disabled" in stderr

    def test_drop_item_still_acknowledged(
        self, tmp_path: Path, mockserver: MockServer
    ) -> None:
        proc = _run_spider(tmp_path, mockserver, drop_items=True, pipeline=True)
        _, stderr = _finish(proc)
        assert proc.returncode == 0, stderr
        assert _inflight_rows(tmp_path / "job") == 0
        assert "Resuming" not in stderr

    def test_terminal_download_failure_is_acknowledged(
        self, tmp_path: Path, mockserver: MockServer
    ) -> None:
        # A final download failure keeps the existing error semantics and ends
        # the request normally instead of triggering infinite recovery.
        proc = _run_spider(tmp_path, mockserver, mode="download_error")
        _, stderr = _finish(proc)
        assert proc.returncode == 0, stderr
        assert _inflight_rows(tmp_path / "job") == 0
        assert "Resuming" not in stderr
        # Restarting the same job directory finds nothing left to recover.
        proc = _run_spider(tmp_path, mockserver, mode="download_error")
        _, stderr = _finish(proc)
        assert proc.returncode == 0, stderr
        assert "Resuming" not in stderr

    def test_graceful_close_acknowledges_finished_request(
        self, tmp_path: Path, mockserver: MockServer
    ) -> None:
        # During a graceful close the in-flight request runs to completion, so
        # it must be acknowledged rather than recovered on the next start.
        proc = _run_spider(
            tmp_path,
            mockserver,
            pipeline=True,
            extra_settings={"graceful_close": True},
        )
        _, stderr = _finish(proc)
        assert proc.returncode == 0, stderr
        assert "Spider closed (shutdown)" in stderr
        assert _inflight_rows(tmp_path / "job") == 0
        assert "Resuming" not in stderr

    def test_fast_close_keeps_inflight_for_recovery(
        self, tmp_path: Path, mockserver: MockServer
    ) -> None:
        # A fast close drops the in-flight download without waiting for the
        # network; the request must stay recoverable on the next start. The
        # long delay endpoint makes sure the download is still running when the
        # fast close reaches the downloader.
        proc = _run_spider(
            tmp_path,
            mockserver,
            extra_settings={"fast_close": True, "long_delay": True},
        )
        _, stderr = _finish(proc)
        assert proc.returncode == 0, stderr
        assert "Fast shutdown dropped" in stderr
        assert _inflight_rows(tmp_path / "job") >= 1

        proc = _run_spider(tmp_path, mockserver, pipeline=True)
        _, stderr = _finish(proc)
        assert proc.returncode == 0, stderr
        assert "Resuming 1 in-flight request" in stderr
        assert _inflight_rows(tmp_path / "job") == 0
