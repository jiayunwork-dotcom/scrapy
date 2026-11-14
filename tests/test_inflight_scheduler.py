from __future__ import annotations

import asyncio
import sqlite3
from typing import TYPE_CHECKING, Any

import pytest

from scrapy import Request, Spider
from scrapy.core.scheduler import Scheduler
from scrapy.utils.defer import maybe_deferred_to_future
from scrapy.utils.misc import build_from_crawler
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
