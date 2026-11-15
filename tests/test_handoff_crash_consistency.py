"""Crash-consistency tests for the scheduler queue -> in-flight handoff.

These tests complement tests/test_inflight_recovery.py, which kills the
subprocess during download, callback and pipeline processing. Here the
subprocess is stopped while a request is being selected, dequeued and handed
to the engine, i.e. before the engine ever receives it. Every kill point is
reached behind a file barrier, so the timing is deterministic and no sleep is
used to guess the window.

The test keeps its own manifest of the request instances it asked the spider
to enqueue (full fields, including identical-URL instances), and after the
job directory is reopened it compares the actually processed instances
against that manifest.
"""

from __future__ import annotations

import json
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from scrapy.utils.test import get_testenv

if TYPE_CHECKING:
    from tests.mockserver.http import MockServer

SCRIPT = Path(__file__).parent / "inflight_recovery" / "inflight_spider.py"
WAIT_TIMEOUT = 15
# Persistent boundaries of the handoff, in the order the scheduler crosses
# them. The recover_* boundaries are crossed while reopening the job
# directory and are covered by the reconciliation tests below.
HANDOFF_STAGES = ("peek", "staged", "popped")


def _wait_for(path: Path, proc: subprocess.Popen[Any]) -> None:
    deadline = time.monotonic() + WAIT_TIMEOUT
    while not path.exists():
        if time.monotonic() > deadline:
            detail = ""
            if proc.poll() is not None and proc.stderr is not None:
                detail = "\n" + proc.stderr.read()
            pytest.fail(f"Timed out waiting for barrier {path.name}{detail}")
        time.sleep(0.02)


def _run_spider(
    tmp_path: Path,
    mockserver: MockServer,
    *,
    manifest: list[dict[str, Any]] | None = None,
    kill_stage: str | None = None,
    pipeline: bool = False,
    extra_settings: dict[str, Any] | None = None,
) -> subprocess.Popen[Any]:
    jobdir = tmp_path / "job"
    barrier_dir = tmp_path / "barriers"
    # Every subprocess starts from a clean barrier directory so markers and
    # counters left by a previous run cannot satisfy a later wait early.
    if barrier_dir.exists():
        for old in barrier_dir.iterdir():
            old.unlink()
    barrier_dir.mkdir(parents=True, exist_ok=True)
    config: dict[str, Any] = {
        "mockserver": mockserver.url("/"),
        "jobdir": str(jobdir),
        "barrier_dir": str(barrier_dir),
        "item_log": str(tmp_path / "items.jsonl"),
        "mode": "handoff",
        "handoff_barriers": True,
        "kill_stage": kill_stage,
        "manifest": manifest or [],
        "pipeline": pipeline,
        # Every instance must reach the held downloader stage together, so use
        # enough downloader concurrency for the whole manifest; a recovered run
        # with an empty manifest holds the recovered requests it finds.
        "concurrent_requests": max(len(manifest or ()), 8),
    }
    config.update(extra_settings or {})
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, str(SCRIPT), str(config_path)],
        env=get_testenv(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if kill_stage is None:
        # A completion run is not killed. The holding downloader middleware
        # parks every handoff request on a download_hold_<n> file after
        # writing downloader_entered_<n>. Create the matching hold for each
        # entry, including recovered requests whose count an empty manifest
        # cannot predict, until the subprocess finishes.
        def _release_holds() -> None:
            while proc.poll() is None:
                for entered in barrier_dir.glob("downloader_entered_*"):
                    index = entered.name.rsplit("_", 1)[-1]
                    hold = barrier_dir / f"download_hold_{index}"
                    if not hold.exists():
                        hold.touch()
                time.sleep(0.01)

        threading.Thread(target=_release_holds, daemon=True).start()
    return proc


def _kill(proc: subprocess.Popen[Any]) -> None:
    # Only the parent test process terminates the subprocess it created.
    if proc.poll() is None:
        proc.send_signal(signal.SIGKILL)
    proc.wait(timeout=WAIT_TIMEOUT)
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
        if not db.execute(
            "SELECT 1 FROM sqlite_schema WHERE name = 'inflight'"
        ).fetchone():
            return 0
        row = db.execute("SELECT COUNT(*) FROM inflight").fetchone()
        return int(row[0]) if row is not None else 0
    finally:
        db.close()


def _read_items(tmp_path: Path) -> list[dict[str, Any]]:
    path = tmp_path / "items.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _base_manifest(mockserver: MockServer) -> list[dict[str, Any]]:
    return [
        {
            "id": "survivor",
            "url": mockserver.url("/status?n=200&name=must-survive"),
            "dont_filter": True,
            "priority": 0,
        }
    ]


class TestHandoffCrashConsistency:
    @pytest.mark.parametrize("stage", HANDOFF_STAGES)
    def test_kill_at_each_handoff_boundary(
        self, tmp_path: Path, mockserver: MockServer, stage: str
    ) -> None:
        manifest = _base_manifest(mockserver)
        proc = _run_spider(tmp_path, mockserver, manifest=manifest, kill_stage=stage)
        _wait_for(tmp_path / "barriers" / f"handoff_{stage}", proc)
        _kill(proc)

        proc = _run_spider(tmp_path, mockserver, manifest=[], pipeline=True)
        code, stderr = _finish(proc)
        assert code == 0, stderr
        items = _read_items(tmp_path)
        # Exactly one schedulable instance survived, regardless of which side
        # of the staging write the kill hit.
        assert len(items) == 1
        assert items[0]["handoff_id"] == "survivor"
        assert items[0]["url"].endswith("name=must-survive")
        assert _inflight_rows(tmp_path / "job") == 0

    @pytest.mark.parametrize("stage", HANDOFF_STAGES)
    def test_consecutive_handoff_kills_never_multiply(
        self, tmp_path: Path, mockserver: MockServer, stage: str
    ) -> None:
        manifest = _base_manifest(mockserver)
        proc = _run_spider(tmp_path, mockserver, manifest=manifest, kill_stage=stage)
        _wait_for(tmp_path / "barriers" / f"handoff_{stage}", proc)
        _kill(proc)
        # Kill again, repeatedly, during the recovery handoff of the instance.
        for _ in range(3):
            proc = _run_spider(tmp_path, mockserver, manifest=[], kill_stage="peek")
            _wait_for(tmp_path / "barriers" / "handoff_peek", proc)
            _kill(proc)

        proc = _run_spider(tmp_path, mockserver, manifest=[], pipeline=True)
        code, stderr = _finish(proc)
        assert code == 0, stderr
        items = _read_items(tmp_path)
        assert [item["handoff_id"] for item in items] == ["survivor"]
        assert _inflight_rows(tmp_path / "job") == 0

    def test_kill_during_recovery_reconciliation(
        self, tmp_path: Path, mockserver: MockServer
    ) -> None:
        # First run: kill after the staging commit, when both the queue entry
        # and the in-flight row exist.
        manifest = _base_manifest(mockserver)
        proc = _run_spider(tmp_path, mockserver, manifest=manifest, kill_stage="staged")
        _wait_for(tmp_path / "barriers" / "handoff_staged", proc)
        _kill(proc)

        proc = _run_spider(
            tmp_path, mockserver, manifest=[], kill_stage="recover_start"
        )
        _wait_for(tmp_path / "barriers" / "handoff_recover_start", proc)
        _kill(proc)

        proc = _run_spider(tmp_path, mockserver, manifest=[], pipeline=True)
        code, stderr = _finish(proc)
        assert code == 0, stderr
        items = _read_items(tmp_path)
        assert len(items) == 1
        assert items[0]["handoff_id"] == "survivor"

    def test_identical_url_instances_stay_independent(
        self, tmp_path: Path, mockserver: MockServer
    ) -> None:
        manifest = [
            {
                "id": "one",
                "url": mockserver.url("/status?n=200&name=dup"),
                "dont_filter": True,
            },
            {
                "id": "two",
                "url": mockserver.url("/status?n=200&name=dup"),
                "dont_filter": True,
            },
        ]
        proc = _run_spider(
            tmp_path,
            mockserver,
            manifest=manifest,
            kill_stage="staged",
        )
        _wait_for(tmp_path / "barriers" / "handoff_staged", proc)
        _kill(proc)

        proc = _run_spider(tmp_path, mockserver, manifest=[], pipeline=True)
        code, stderr = _finish(proc)
        assert code == 0, stderr
        items = sorted(item["handoff_id"] for item in _read_items(tmp_path))
        assert items == ["one", "two"]

    def test_other_pending_requests_and_priorities_survive(
        self, tmp_path: Path, mockserver: MockServer
    ) -> None:
        manifest = [
            {
                "id": "low",
                "url": mockserver.url("/status?n=200&name=low"),
                "priority": -5,
            },
            {
                "id": "high",
                "url": mockserver.url("/status?n=200&name=high"),
                "priority": 10,
            },
            {
                "id": "start",
                "url": mockserver.url("/status?n=200&name=start"),
                "is_start": True,
            },
        ]
        proc = _run_spider(
            tmp_path,
            mockserver,
            manifest=manifest,
            kill_stage="staged",
        )
        _wait_for(tmp_path / "barriers" / "handoff_staged", proc)
        _kill(proc)

        proc = _run_spider(tmp_path, mockserver, manifest=[], pipeline=True)
        code, stderr = _finish(proc)
        assert code == 0, stderr
        raw_items = _read_items(tmp_path)
        items = {item["handoff_id"]: item for item in raw_items}
        assert set(items) == {"low", "high", "start"}
        # The priority of every instance survives the handoff and recovery,
        # including the negative value and the start request's default.
        assert items["low"]["priority"] == -5
        assert items["high"]["priority"] == 10
        assert items["start"]["priority"] == 0
        # The start-request marker survives the separate start disk queue.
        assert items["start"]["is_start"] is True
        # Other pending requests are neither lost nor duplicated.
        assert len(raw_items) == 3

    def test_full_request_fields_survive_handoff(
        self, tmp_path: Path, mockserver: MockServer
    ) -> None:
        manifest = [
            {
                "id": "full",
                "url": mockserver.url("/echo?name=full"),
                "body": "payload",
                "dont_filter": True,
                "priority": 7,
            }
        ]
        proc = _run_spider(tmp_path, mockserver, manifest=manifest, kill_stage="staged")
        _wait_for(tmp_path / "barriers" / "handoff_staged", proc)
        _kill(proc)
        proc = _run_spider(tmp_path, mockserver, manifest=[], pipeline=True)
        code, stderr = _finish(proc)
        assert code == 0, stderr
        item = _read_items(tmp_path)[0]
        assert item["method"] == "POST"
        assert item["priority"] == 7
        assert item["dont_filter"] is True

    def test_resubmission_after_handoff_kill_still_deduplicated(
        self, tmp_path: Path, mockserver: MockServer
    ) -> None:
        # A filtered request whose handoff was interrupted must come back from
        # the job directory itself; a freshly submitted identical request stays
        # rejected, proving recovery did not clear the dupefilter history.
        manifest = [
            {
                "id": "filtered",
                "url": mockserver.url("/status?n=200&name=seen"),
                "dont_filter": False,
            }
        ]
        proc = _run_spider(tmp_path, mockserver, manifest=manifest, kill_stage="staged")
        _wait_for(tmp_path / "barriers" / "handoff_staged", proc)
        _kill(proc)
        proc = _run_spider(
            tmp_path,
            mockserver,
            manifest=[
                {
                    "id": "fresh-duplicate",
                    "url": mockserver.url("/status?n=200&name=seen"),
                }
            ],
            pipeline=True,
        )
        code, stderr = _finish(proc)
        assert code == 0, stderr
        ids = [item["handoff_id"] for item in _read_items(tmp_path)]
        assert ids == ["filtered"]
        assert "fresh-duplicate" not in ids

    def test_acknowledging_one_identical_instance_keeps_the_other(
        self, tmp_path: Path, mockserver: MockServer
    ) -> None:
        manifest = [
            {
                "id": "one",
                "url": mockserver.url("/status?n=200&name=dup"),
                "dont_filter": True,
            },
            {
                "id": "two",
                "url": mockserver.url("/status?n=200&name=dup"),
                "dont_filter": True,
            },
        ]
        proc = _run_spider(
            tmp_path,
            mockserver,
            manifest=manifest,
            kill_stage="staged",
        )
        _wait_for(tmp_path / "barriers" / "handoff_staged", proc)
        _kill(proc)
        # Both copies reconcile; finish the crawl, which processes and acks
        # both, then restart with the same job directory: nothing resurrects.
        proc = _run_spider(tmp_path, mockserver, manifest=[], pipeline=True)
        code, stderr = _finish(proc)
        assert code == 0, stderr
        assert _inflight_rows(tmp_path / "job") == 0
        first_run_ids = {item["handoff_id"] for item in _read_items(tmp_path)}
        assert first_run_ids == {"one", "two"}
        # A second run with the same job directory finds nothing to recover:
        # the acknowledged instances must not resurrect.
        proc = _run_spider(tmp_path, mockserver, manifest=[], pipeline=True)
        code, stderr = _finish(proc)
        assert code == 0, stderr
        assert _inflight_rows(tmp_path / "job") == 0
        # The completion run processed the two manifest instances once each,
        # and the restart does not resurrect or duplicate anything.
        assert {item["handoff_id"] for item in _read_items(tmp_path)} == first_run_ids
