from __future__ import annotations

import asyncio
import pickle
import sqlite3
from typing import TYPE_CHECKING, Any, cast

import pytest

from scrapy import Request, Spider
from scrapy.core.scheduler import _HANDOFF_TOKEN_META_KEY, Scheduler
from scrapy.utils.defer import maybe_deferred_to_future
from scrapy.utils.misc import build_from_crawler
from scrapy.utils.request import request_from_dict
from scrapy.utils.test import get_crawler

if TYPE_CHECKING:
    from pathlib import Path

    from scrapy.crawler import Crawler

_TRACK_SETTINGS = {
    "JOBDIR_TRACK_INFLIGHT": True,
    "SCHEDULER_PRIORITY_QUEUE": "scrapy.pqueues.ScrapyPriorityQueue",
    "SCHEDULER_DISK_QUEUE": "scrapy.squeues.PickleLifoSQLiteQueue",
    "SCHEDULER_START_DISK_QUEUE": "scrapy.squeues.PickleFifoSQLiteQueue",
    "DUPEFILTER_CLASS": "scrapy.dupefilters.DiskDupeFilter",
    "JOBDIR_SYNC_EVERY": 1,
}


def _crawler(tmp_path: Path, **overrides: Any) -> Crawler:
    return get_crawler(
        Spider, {"JOBDIR": str(tmp_path), **_TRACK_SETTINGS, **overrides}
    )


def _open_scheduler(crawler: Crawler) -> tuple[Scheduler, Spider]:
    spider = Spider.from_crawler(crawler, name="inflight")
    crawler.spider = spider
    scheduler = build_from_crawler(Scheduler, crawler)
    scheduler.open(spider)
    return scheduler, spider


def _close(scheduler: Scheduler) -> None:
    result = scheduler.close("finished")
    if result is not None:
        asyncio.run(maybe_deferred_to_future(result))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("SCHEDULER_PRIORITY_QUEUE", "scrapy.pqueues.DownloaderAwarePriorityQueue"),
        ("SCHEDULER_DISK_QUEUE", "scrapy.squeues.PickleLifoDiskQueue"),
        (
            "SCHEDULER_START_DISK_QUEUE",
            "scrapy.squeues.PickleFifoDiskQueue",
        ),
        ("DUPEFILTER_CLASS", "scrapy.dupefilters.RFPDupeFilter"),
        ("JOBDIR_SYNC_EVERY", 0),
    ],
)
def test_unsupported_configuration_raises(
    tmp_path: Path, setting: str, value: Any
) -> None:
    crawler = _crawler(tmp_path, **{setting: value})
    scheduler = build_from_crawler(Scheduler, crawler)
    with pytest.raises(RuntimeError, match="JOBDIR_TRACK_INFLIGHT"):
        scheduler.open(Spider.from_crawler(crawler, name="x"))


def test_tracking_requires_jobdir(tmp_path: Path) -> None:
    crawler = get_crawler(Spider, _TRACK_SETTINGS)
    scheduler = build_from_crawler(Scheduler, crawler)
    with pytest.raises(RuntimeError, match="JOBDIR"):
        scheduler.open(Spider.from_crawler(crawler, name="x"))


def test_track_and_ack_roundtrip(tmp_path: Path) -> None:
    crawler = _crawler(tmp_path)
    scheduler, _spider = _open_scheduler(crawler)
    request = Request("https://example.com/a")
    assert scheduler.enqueue_request(request)
    popped = scheduler.next_request()
    assert popped is not None
    db_path = tmp_path / "requests.inflight.db"
    db = sqlite3.connect(db_path)
    assert db.execute("SELECT COUNT(*) FROM inflight").fetchone()[0] == 1
    db.close()

    scheduler.ack_inflight(popped)
    db = sqlite3.connect(db_path)
    assert db.execute("SELECT COUNT(*) FROM inflight").fetchone()[0] == 0
    db.close()
    _close(scheduler)


def test_recovery_reuses_pending_queue(tmp_path: Path) -> None:
    crawler = _crawler(tmp_path)
    scheduler, _ = _open_scheduler(crawler)
    in_flight = Request("https://example.com/in-flight")
    pending = Request("https://example.com/pending")
    scheduler.enqueue_request(in_flight)
    scheduler.enqueue_request(pending)
    popped = scheduler.next_request()
    assert popped is not None
    _close(scheduler)

    crawler2 = _crawler(tmp_path)
    scheduler2, _ = _open_scheduler(crawler2)
    urls = set()
    while scheduler2.has_pending_requests():
        request = scheduler2.next_request()
        assert request is not None
        urls.add(request.url)
        scheduler2.ack_inflight(request)
    # The in-flight request comes from the recovery store and the other one
    # from the regular SQLite request queue, and both are processed once.
    assert urls == {
        "https://example.com/in-flight",
        "https://example.com/pending",
    }
    _close(scheduler2)


def test_recovery_does_not_change_dont_filter(tmp_path: Path) -> None:
    crawler = _crawler(tmp_path)
    scheduler, _ = _open_scheduler(crawler)
    scheduler.enqueue_request(Request("https://example.com/a"))
    popped = scheduler.next_request()
    assert popped is not None
    _close(scheduler)

    crawler2 = _crawler(tmp_path)
    scheduler2, _ = _open_scheduler(crawler2)
    recovered = scheduler2.next_request()
    assert recovered is not None
    assert recovered.dont_filter is False
    # A brand-new identical request must still go through the dupefilter,
    # i.e. be rejected as a duplicate rather than silently accepted.
    duplicate = Request("https://example.com/a")
    assert scheduler2.enqueue_request(duplicate) is False
    scheduler2.ack_inflight(recovered)
    _close(scheduler2)


def test_identical_dont_filter_requests_tracked_separately(tmp_path: Path) -> None:
    crawler = _crawler(tmp_path)
    scheduler, _ = _open_scheduler(crawler)
    first = Request("https://example.com/dup", dont_filter=True)
    second = Request("https://example.com/dup", dont_filter=True)
    assert scheduler.enqueue_request(first)
    assert scheduler.enqueue_request(second)
    popped_first = scheduler.next_request()
    popped_second = scheduler.next_request()
    assert popped_first is not None
    assert popped_second is not None
    db = sqlite3.connect(tmp_path / "requests.inflight.db")
    assert db.execute("SELECT COUNT(*) FROM inflight").fetchone()[0] == 2
    db.close()
    # Acknowledging one must not acknowledge the other.
    scheduler.ack_inflight(popped_first)
    db = sqlite3.connect(tmp_path / "requests.inflight.db")
    assert db.execute("SELECT COUNT(*) FROM inflight").fetchone()[0] == 1
    db.close()
    scheduler.ack_inflight(popped_second)
    _close(scheduler)


def test_recovery_preserves_request_attributes(tmp_path: Path) -> None:
    crawler = _crawler(tmp_path)
    scheduler, _spider = _open_scheduler(crawler)
    request = Request(
        "https://example.com/a",
        method="POST",
        body=b"payload",
        headers={"X-Test": "1"},
        cookies={"session": "abc"},
        meta={"depth": 3, "is_start_request": True},
        cb_kwargs={"item_id": 7},
        callback=_spider.parse,
        priority=42,
        dont_filter=True,
    )
    scheduler.enqueue_request(request)
    popped = scheduler.next_request()
    assert popped is not None
    _close(scheduler)

    crawler2 = _crawler(tmp_path)
    scheduler2, spider2 = _open_scheduler(crawler2)
    recovered = scheduler2.next_request()
    assert recovered is not None
    assert recovered.url == "https://example.com/a"
    assert recovered.method == "POST"
    assert recovered.body == b"payload"
    assert recovered.headers["X-Test"] == b"1"
    assert recovered.cookies == {"session": "abc"}
    assert recovered.meta["depth"] == 3
    assert recovered.cb_kwargs == {"item_id": 7}
    assert recovered.callback == spider2.parse
    assert recovered.errback is None
    assert recovered.priority == 42
    assert recovered.dont_filter is True
    _close(scheduler2)


def test_unserializable_request_rejected_when_tracking(tmp_path: Path) -> None:
    crawler = _crawler(tmp_path)
    scheduler, _ = _open_scheduler(crawler)
    request = Request("https://example.com/a", meta={"callable": lambda: None})
    assert scheduler.enqueue_request(request) is False
    assert crawler.stats.get_value("scheduler/inflight/unserializable") == 1
    _close(scheduler)


def test_disabled_tracking_with_records_raises(tmp_path: Path) -> None:
    crawler = _crawler(tmp_path)
    scheduler, _ = _open_scheduler(crawler)
    scheduler.enqueue_request(Request("https://example.com/a"))
    popped = scheduler.next_request()
    assert popped is not None
    _close(scheduler)

    crawler2 = _crawler(tmp_path, JOBDIR_TRACK_INFLIGHT=False)
    scheduler2 = build_from_crawler(Scheduler, crawler2)
    with pytest.raises(RuntimeError, match="JOBDIR_TRACK_INFLIGHT is disabled"):
        scheduler2.open(Spider.from_crawler(crawler2, name="x"))


def test_empty_jobdir_with_tracking_disabled_is_fine(tmp_path: Path) -> None:
    crawler = get_crawler(
        Spider,
        {
            "JOBDIR": str(tmp_path),
            "SCHEDULER_PRIORITY_QUEUE": "scrapy.pqueues.ScrapyPriorityQueue",
        },
    )
    scheduler = build_from_crawler(Scheduler, crawler)
    result = scheduler.open(Spider.from_crawler(crawler, name="x"))
    assert result is None or not result.called
    _close(scheduler)


def test_completed_history_does_not_grow_store(tmp_path: Path) -> None:
    crawler = _crawler(tmp_path)
    scheduler, _ = _open_scheduler(crawler)
    for index in range(20):
        request = Request(f"https://example.com/{index}")
        scheduler.enqueue_request(request)
        popped = scheduler.next_request()
        assert popped is not None
        scheduler.ack_inflight(popped)
    db = sqlite3.connect(tmp_path / "requests.inflight.db")
    # Every acknowledged row is deleted immediately, so the table stays empty
    # regardless of how many requests completed.
    assert db.execute("SELECT COUNT(*) FROM inflight").fetchone()[0] == 0
    db.close()
    _close(scheduler)


# ---------------------------------------------------------------------------
# Crash-safe queue -> in-flight handoff (in-process and fault-injection tests)
# ---------------------------------------------------------------------------


def _inflight_contents(tmp_path: Path) -> list[tuple[bytes, str | None]]:
    db = sqlite3.connect(tmp_path / "requests.inflight.db")
    rows = db.execute(
        "SELECT request, handoff_token FROM inflight ORDER BY id"
    ).fetchall()
    db.close()
    return rows


def test_handoff_stages_before_queue_removal(tmp_path: Path) -> None:
    crawler = _crawler(tmp_path)
    scheduler, _ = _open_scheduler(crawler)
    request = Request("https://example.com/handoff")
    assert scheduler.enqueue_request(request)

    # After the handoff the request left the queue and is held only by the
    # in-flight store, linked through its handoff token.
    popped = scheduler.next_request()
    assert popped is not None
    assert popped.url == "https://example.com/handoff"
    assert len(scheduler) == 0
    rows = _inflight_contents(tmp_path)
    assert len(rows) == 1
    assert rows[0][1] is not None
    # The token is scheduler-private and never reaches the engine request.
    assert _HANDOFF_TOKEN_META_KEY not in popped.meta
    scheduler.ack_inflight(popped)
    assert _inflight_contents(tmp_path) == []
    _close(scheduler)


def test_kill_after_staging_keeps_single_instance(tmp_path: Path) -> None:
    # Simulate a kill after the staging commit but before the queue removal:
    # both stores hold a copy with the same token, so recovery must discard
    # exactly one and keep a single schedulable instance.
    crawler = _crawler(tmp_path)
    scheduler, _ = _open_scheduler(crawler)
    request = Request("https://example.com/overlap", dont_filter=True)
    assert scheduler.enqueue_request(request)
    assert scheduler.dqs is not None
    assert scheduler._inflight is not None
    # Manually create the overlap state the crash would leave behind.
    entry = scheduler.dqs.peek_entry()
    assert entry is not None
    _entry_id, queued = entry
    token = queued.meta["_scrapy_handoff_token"]

    scheduler._inflight.stage(
        queued,
        pickle.dumps(queued.to_dict(spider=scheduler.spider), protocol=4),
        token,
    )
    _close(scheduler)

    crawler2 = _crawler(tmp_path)
    scheduler2, _ = _open_scheduler(crawler2)
    assert crawler2.stats.get_value("scheduler/inflight/duplicated") == 1
    assert _inflight_contents(tmp_path) == []
    popped = scheduler2.next_request()
    assert popped is not None
    assert popped.url == "https://example.com/overlap"
    scheduler2.ack_inflight(popped)
    _close(scheduler2)


def test_kill_before_staging_keeps_queue_copy(tmp_path: Path) -> None:
    # With no in-flight row at all, the queue entry is the surviving copy and
    # recovery must not duplicate or drop it.
    crawler = _crawler(tmp_path)
    scheduler, _ = _open_scheduler(crawler)
    assert scheduler.enqueue_request(Request("https://example.com/queued-only"))
    _close(scheduler)

    crawler2 = _crawler(tmp_path)
    scheduler2, _ = _open_scheduler(crawler2)
    assert crawler2.stats.get_value("scheduler/inflight/recovered") is None
    popped = scheduler2.next_request()
    assert popped is not None
    assert popped.url == "https://example.com/queued-only"
    scheduler2.ack_inflight(popped)
    _close(scheduler2)


def test_legacy_inflight_row_without_token_is_recovered(tmp_path: Path) -> None:
    # A row written by the first in-flight tracking release has no handoff
    # token; it must always be re-enqueued, never mistaken for a duplicate.
    crawler = _crawler(tmp_path)
    scheduler, spider = _open_scheduler(crawler)
    request = Request("https://example.com/legacy")
    data = pickle.dumps(request.to_dict(spider=spider), protocol=4)
    db = sqlite3.connect(tmp_path / "requests.inflight.db")
    db.execute(
        "INSERT INTO inflight (request, handoff_token) VALUES (?, NULL)",
        (data,),
    )
    db.commit()
    db.close()
    _close(scheduler)

    crawler2 = _crawler(tmp_path)
    scheduler2, _ = _open_scheduler(crawler2)
    assert crawler2.stats.get_value("scheduler/inflight/recovered") == 1
    popped = scheduler2.next_request()
    assert popped is not None
    assert popped.url == "https://example.com/legacy"
    scheduler2.ack_inflight(popped)
    _close(scheduler2)


class _FailingStore:
    """Fault double for _InflightRequestStore controlling whether the failure
    happens before or after the write becomes durable."""

    def __init__(self, *, fail_before_commit: bool) -> None:
        self._fail_before_commit = fail_before_commit
        self.rows: list[tuple[bytes, str]] = []

    def stage(self, request, data: bytes, handoff_token: str) -> int:
        if self._fail_before_commit:
            # The operation never took effect: no row exists afterwards.
            raise sqlite3.OperationalError("simulated write failure")
        # The write is durable first, only then the caller sees an error.
        self.rows.append((data, handoff_token))
        raise sqlite3.OperationalError("failure after durable commit")

    def claim(self, request, handoff_token: str) -> None:
        raise AssertionError("claim must not be reached after a stage failure")

    def ack(self, request) -> None:
        raise AssertionError("ack must not be reached after a stage failure")

    def discard_recovered(self, row_id: int) -> None:
        self.rows.pop(row_id - 1)

    def recover(self):
        for index, (data, token) in enumerate(self.rows, start=1):
            yield index, request_from_dict(pickle.loads(data)), token

    def close(self) -> None:
        pass


def test_stage_failure_before_effect_does_not_hand_out_request(
    tmp_path: Path,
) -> None:
    crawler = _crawler(tmp_path)
    scheduler, _ = _open_scheduler(crawler)
    failures: list[str] = []
    scheduler._fail_inflight = failures.append  # type: ignore[method-assign,assignment]
    scheduler.enqueue_request(Request("https://example.com/a"))
    scheduler._inflight = cast("Any", _FailingStore(fail_before_commit=True))

    # The request is not handed to the engine, and the persistence failure is
    # reported through the error-close path instead of being swallowed.
    assert scheduler.next_request() is None
    assert failures
    # The operation never took effect, so the queue still holds the only copy
    # and reopening the job directory recovers it.
    _close(scheduler)

    crawler2 = _crawler(tmp_path)
    scheduler2, _ = _open_scheduler(crawler2)
    popped = scheduler2.next_request()
    assert popped is not None
    assert popped.url == "https://example.com/a"
    scheduler2.ack_inflight(popped)
    _close(scheduler2)


def test_stage_failure_after_effect_still_recoverable(tmp_path: Path) -> None:
    crawler = _crawler(tmp_path)
    scheduler, _ = _open_scheduler(crawler)
    failures: list[str] = []
    scheduler._fail_inflight = failures.append  # type: ignore[method-assign,assignment]
    scheduler.enqueue_request(Request("https://example.com/b"))
    failing = _FailingStore(fail_before_commit=False)
    scheduler._inflight = cast("Any", failing)

    assert scheduler.next_request() is None
    assert failures
    # The durable row exists even though the caller saw an exception.
    assert len(failing.rows) == 1
    _close(scheduler)


class _PopAfterCommitFailingQueue:
    """Fault double whose pop_entry raises only after deleting the row,
    modelling an error that arrives once the removal is already durable."""

    def __init__(self, real_queue) -> None:
        self._real = real_queue

    def peek_entry(self):
        return self._real.peek_entry()

    def stamp_entry(self, entry_id, request):
        return self._real.stamp_entry(entry_id, request)

    def pop_entry(self, entry_id):
        self._real.pop_entry(entry_id)
        raise sqlite3.OperationalError("failure after durable removal")

    def pending_handoff_tokens(self):
        return set()

    def push(self, request):
        self._real.push(request)

    def close(self):
        return self._real.close()

    def state(self):
        return self._real.state()

    def __len__(self):
        return len(self._real)


def test_pop_failure_after_durable_removal_uses_inflight_copy(
    tmp_path: Path,
) -> None:
    crawler = _crawler(tmp_path)
    scheduler, _ = _open_scheduler(crawler)
    failures: list[str] = []
    scheduler._fail_inflight = failures.append  # type: ignore[method-assign,assignment]
    scheduler.enqueue_request(Request("https://example.com/c"))
    scheduler.dqs = cast("Any", _PopAfterCommitFailingQueue(scheduler.dqs))

    # Nothing is handed out after the error; the staged in-flight row is the
    # surviving copy and must be re-scheduled on the next open.
    assert scheduler.next_request() is None
    assert failures
    _close(scheduler)

    crawler2 = _crawler(tmp_path)
    scheduler2, _ = _open_scheduler(crawler2)
    popped = scheduler2.next_request()
    assert popped is not None
    assert popped.url == "https://example.com/c"
    scheduler2.ack_inflight(popped)
    _close(scheduler2)


def test_two_identical_dont_filter_handoffs_keep_both(tmp_path: Path) -> None:
    crawler = _crawler(tmp_path)
    scheduler, _ = _open_scheduler(crawler)
    first = Request("https://example.com/dup", dont_filter=True)
    second = Request("https://example.com/dup", dont_filter=True)
    assert scheduler.enqueue_request(first)
    assert scheduler.enqueue_request(second)
    popped_first = scheduler.next_request()
    popped_second = scheduler.next_request()
    assert popped_first is not None
    assert popped_second is not None
    assert len(_inflight_contents(tmp_path)) == 2
    scheduler.ack_inflight(popped_first)
    assert len(_inflight_contents(tmp_path)) == 1
    scheduler.ack_inflight(popped_second)
    assert _inflight_contents(tmp_path) == []
    _close(scheduler)


def test_recovery_bypass_does_not_change_dont_filter_or_history(
    tmp_path: Path,
) -> None:
    crawler = _crawler(tmp_path)
    scheduler, _ = _open_scheduler(crawler)
    scheduler.enqueue_request(Request("https://example.com/seen"))
    popped = scheduler.next_request()
    assert popped is not None
    _close(scheduler)

    crawler2 = _crawler(tmp_path)
    scheduler2, _ = _open_scheduler(crawler2)
    recovered = scheduler2.next_request()
    assert recovered is not None
    assert recovered.dont_filter is False
    # The dupefilter history survived: a brand-new identical request is still
    # rejected, proving recovery neither cleared history nor set dont_filter.
    assert scheduler2.enqueue_request(Request("https://example.com/seen")) is False
    scheduler2.ack_inflight(recovered)
    _close(scheduler2)
