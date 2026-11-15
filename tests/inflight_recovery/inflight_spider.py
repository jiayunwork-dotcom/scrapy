"""Subprocess spider script for tests/test_inflight_recovery.py.

The spider crawls against the test mock server and uses barrier files in the
barrier directory to tell the parent test process which processing stage a
request is in, so the parent can kill the process at a precise point instead
of guessing with sleeps.

The script is intentionally self-contained: every variation is selected through
the JSON config given as the first command-line argument, so the parent process
never has to rewrite it.
"""

from __future__ import annotations

import asyncio
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any

from scrapy import Request, Spider, signals
from scrapy.core import scheduler as scheduler_module
from scrapy.crawler import CrawlerProcess
from scrapy.exceptions import DropItem
from scrapy.utils.defer import _schedule_coro

_CONFIG: dict[str, Any] = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))


class _Barrier:
    """File-based barrier: the spider reaches a stage by creating a file, and
    optionally waits for the parent process to create another one."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory

    def reach(self, name: str) -> None:
        path = self._directory / name
        with path.open("wb") as f:
            f.flush()
            os.fsync(f.fileno())

    def wait_for(self, name: str) -> None:
        path = self._directory / name
        while not path.exists():
            time.sleep(0.02)


class _HoldingDownloaderMiddleware:
    """Parks each handoff-mode request in the downloader, after the scheduler
    handed it over, until the parent releases it or kills the process.

    The request is parked inside an ``await`` loop, so the reactor keeps
    running and the scheduler keeps dequeuing the remaining manifest requests.
    With enough downloader concurrency every instance reaches the same
    handoff boundary and none can complete (and get acknowledged) before the
    parent kills the process.
    """

    def __init__(self) -> None:
        self._seen = 0

    async def process_request(self, request, spider=None):
        if _CONFIG.get("mode") != "handoff":
            return
        from twisted.internet import reactor

        self._seen += 1
        barrier = _Barrier(Path(_CONFIG["barrier_dir"]))
        hold = barrier._directory / f"download_hold_{self._seen}"
        barrier.reach(f"downloader_entered_{self._seen}")

        # Park without blocking the reactor: the scheduler keeps dequeuing the
        # other manifest requests while this one waits for its release file.
        while not hold.exists():
            await asyncio.sleep(0.02)
            if not reactor.running:  # pragma: no cover - safety net only
                break


class _BarrierScheduler(scheduler_module.Scheduler):
    """Scheduler variant that records each persistent boundary of the queue ->
    in-flight handoff behind a file barrier, so the parent test can kill the
    subprocess at a provably correct point."""

    def _handoff_barrier(self, stage: str) -> None:
        barrier = _Barrier(Path(_CONFIG["barrier_dir"]))
        expected = len(_CONFIG.get("manifest", ()))
        # Count how many requests reached this durable boundary and expose the
        # stage marker once every manifest instance is accounted for. None of
        # them can finish before that point: the holding downloader middleware
        # parks them after they leave the scheduler.
        marker_count = barrier._directory / f"handoff_{stage}_count"
        reached = 1
        if marker_count.exists():
            reached = int(marker_count.read_text()) + 1
        marker_count.write_text(str(reached))
        if expected == 0 or reached >= expected:
            barrier.reach(f"handoff_{stage}")

    def _handoff_disk_request(self):
        assert self.dqs is not None
        assert self._inflight is not None
        entry = self.dqs.peek_entry()
        if entry is None:
            return None
        entry_id, queued_request = entry
        self._handoff_barrier("peek")
        token = scheduler_module._handoff_token_of(queued_request)
        if token is None:
            token = scheduler_module._new_handoff_token()
            queued_request.meta[scheduler_module._HANDOFF_TOKEN_META_KEY] = token
            if not self.dqs.stamp_entry(entry_id, queued_request):
                return None
        data = pickle.dumps(queued_request.to_dict(spider=self.spider), protocol=4)
        self._inflight.stage(queued_request, data, token)
        self._handoff_barrier("staged")
        request = self.dqs.pop_entry(entry_id)
        self._sync_dqs_state()
        self._handoff_barrier("popped")
        if request is None:
            return None
        self._inflight.claim(request, token)
        request.meta.pop(scheduler_module._HANDOFF_TOKEN_META_KEY, None)
        self.stats.inc_value("scheduler/inflight/tracked")
        return request

    def _recover_inflight(self) -> None:
        self._handoff_barrier("recover_start")
        super()._recover_inflight()
        self._handoff_barrier("recover_done")


class InflightSpider(Spider):
    name = "inflight"

    custom_settings: dict[str, Any] = {
        "JOBDIR_TRACK_INFLIGHT": True,
        "JOBDIR_SYNC_EVERY": 1,
        "SCHEDULER_PRIORITY_QUEUE": "scrapy.pqueues.ScrapyPriorityQueue",
        "SCHEDULER_DISK_QUEUE": "scrapy.squeues.PickleLifoSQLiteQueue",
        "SCHEDULER_START_DISK_QUEUE": ("scrapy.squeues.PickleFifoSQLiteQueue"),
        "DUPEFILTER_CLASS": "scrapy.dupefilters.DiskDupeFilter",
        "LOG_LEVEL": "INFO",
    }

    @classmethod
    def update_settings(cls, settings):
        super().update_settings(settings)
        settings.set("JOBDIR", _CONFIG["jobdir"], priority="spider")
        settings.set(
            "CONCURRENT_REQUESTS",
            _CONFIG.get("concurrent_requests", 1),
            priority="spider",
        )
        settings.set("RETRY_TIMES", _CONFIG.get("retry_times", 2), priority="spider")
        if _CONFIG.get("handoff_barriers"):
            settings.set(
                "SCHEDULER", f"{__name__}._BarrierScheduler", priority="spider"
            )
            settings.set(
                "DOWNLOADER_MIDDLEWARES",
                {f"{__name__}._HoldingDownloaderMiddleware": 100},
                priority="spider",
            )
        if _CONFIG.get("disable_tracking"):
            settings.set("JOBDIR_TRACK_INFLIGHT", False, priority="spider")
        if _CONFIG.get("pipeline"):
            settings.set(
                "ITEM_PIPELINES",
                {f"{__name__}.BarrierPipeline": 100},
                priority="spider",
            )
        if _CONFIG.get("drop_items"):
            settings.set(
                "ITEM_PIPELINES",
                {f"{__name__}.DropPipeline": 100},
                priority="spider",
            )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mockserver: str = _CONFIG["mockserver"]
        self.barrier_dir: Path = Path(_CONFIG["barrier_dir"])
        self.kill_stage: str | None = _CONFIG.get("kill_stage")
        self.wait_stage: str | None = _CONFIG.get("wait_stage")
        self.mode: str = _CONFIG.get("mode", "basic")
        self.item_log: Path = Path(_CONFIG["item_log"])

    def _barrier(self) -> _Barrier:
        return _Barrier(self.barrier_dir)

    def _handoff_requests(self):
        """Build the requests described by the parent's manifest. The manifest
        is the test's independent record of every instance, including requests
        sharing a URL, so recovery can be compared against it field by field."""
        for entry in _CONFIG.get("manifest", []):
            kwargs: dict[str, Any] = {
                "dont_filter": entry.get("dont_filter", False),
                "priority": entry.get("priority", 0),
                "callback": self.parse_handoff,
                "meta": {"handoff_id": entry["id"]},
            }
            if entry.get("is_start"):
                kwargs["meta"]["is_start_request"] = True
            if entry.get("body") is not None:
                kwargs["body"] = entry["body"].encode()
                kwargs["method"] = "POST"
            yield Request(entry["url"], **kwargs)

    def parse_handoff(self, response, **kwargs):
        yield {
            "kind": "handoff",
            "handoff_id": response.meta["handoff_id"],
            "url": response.url,
            "method": response.request.method,
            "priority": response.request.priority,
            "dont_filter": response.request.dont_filter,
            "is_start": response.meta.get("is_start_request", False),
        }

    def _maybe_kill(self, stage: str) -> None:
        if self.kill_stage == stage:
            sys.stderr.flush()
            os._exit(1)

    async def start(self):
        if self.mode == "handoff":
            # The parent process writes the independently recorded instances,
            # with their full fields, into the manifest; the spider only
            # enqueues them and lets the scheduler barriers control timing.
            for request in self._handoff_requests():
                yield request
            return
        if self.mode == "duplicate":
            for _ in range(2):
                yield Request(
                    self.mockserver + "/status?n=200",
                    dont_filter=True,
                    callback=self.parse,
                )
            return
        if self.mode == "priority":
            # The first two requests share a priority and exercise the
            # no-order-guarantee case; the third one has a higher priority and
            # must always be dequeued before the lower-priority pending one.
            for url, priority in (
                ("/status?n=200&name=first", 0),
                ("/status?n=200&name=second", 0),
                ("/status?n=200&name=high", 10),
            ):
                yield Request(
                    self.mockserver + url,
                    priority=priority,
                    callback=self.parse,
                )
            return
        if self.mode == "retry":
            yield Request(
                self.mockserver + "/status?n=500",
                callback=self.parse,
                errback=self.errback,
            )
            return
        if self.mode == "redirect":
            yield Request(
                self.mockserver + "/redirect",
                callback=self.parse,
            )
            return
        if self.mode == "unserializable":
            yield Request(
                self.mockserver + "/status?n=200",
                meta={"not_picklable": lambda: None},
            )
            return
        if self.mode == "download_error":
            # Retries are disabled, so the 500 response exhausts the retry
            # middleware and reaches the errback as a terminal failure.
            yield Request(
                self.mockserver + "/status?n=500",
                errback=self.errback,
                meta={"dont_retry": True},
            )
            return
        delay = 10 if _CONFIG.get("long_delay") else 2
        yield Request(
            self.mockserver + f"/delay?n={delay}",
            callback=self.parse,
            errback=self.errback,
        )
        if _CONFIG.get("with_pending"):
            # A second request stays queued while the first one is in flight.
            yield Request(
                self.mockserver + "/status?n=200&name=pending",
                callback=self.parse,
            )

    def parse(self, response, **kwargs):
        barrier = self._barrier()
        stage = response.url.split("name=")[-1] if "name=" in response.url else "main"
        barrier.reach(f"callback_entered_{stage}")
        self._maybe_kill("callback")
        if self.wait_stage == "callback":
            barrier.wait_for("release_callback")
        if self.mode == "duplicate":
            yield {"url": response.url, "kind": "duplicate"}
            return
        if self.mode == "priority":
            yield {"url": response.url, "kind": "priority"}
            return
        yield {"url": response.url, "kind": "item"}

    def errback(self, failure):
        barrier = self._barrier()
        barrier.reach("errback_entered")
        self._maybe_kill("errback")
        return {"url": failure.request.url, "kind": "errback"}


class BarrierPipeline:
    async def process_item(self, item, spider: InflightSpider):
        barrier = spider._barrier()
        barrier.reach("pipeline_entered")
        spider._maybe_kill("pipeline")
        if spider.wait_stage == "pipeline":
            barrier.wait_for("release_pipeline")
        with spider.item_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(item) + "\n")
        return item


class DropPipeline:
    async def process_item(self, item, spider):
        raise DropItem("dropped by design")


class _AutoClose:
    """Waits until the first request is being downloaded, then asks the
    engine to close in the configured mode, so the close races with an actual
    in-flight request instead of happening before any request is scheduled."""

    def __init__(self, mode: str, barrier_dir: Path):
        self._mode = mode
        self._barrier_dir = barrier_dir

    @classmethod
    def from_crawler(cls, crawler):
        extension = cls(
            crawler.settings["CLOSESPIDER_INFLIGHT_MODE"],
            Path(crawler.settings["CLOSESPIDER_BARRIER_DIR"]),
        )
        # Fires when the request enters the downloader, while its response is
        # still pending, which is the in-flight window.
        crawler.signals.connect(
            extension.request_reached_downloader,
            signals.request_reached_downloader,
        )
        return extension

    def request_reached_downloader(self, request, spider):
        _Barrier(self._barrier_dir).reach("close_started")
        _schedule_coro(
            spider.crawler.engine.close_spider_async(reason="shutdown", mode=self._mode)
        )


def main() -> None:
    settings: dict[str, Any] = {}
    if _CONFIG.get("graceful_close") or _CONFIG.get("fast_close"):
        settings["EXTENSIONS"] = {f"{__name__}._AutoClose": 0}
        settings["CLOSESPIDER_INFLIGHT_MODE"] = (
            "fast" if _CONFIG.get("fast_close") else "graceful"
        )
        settings["CLOSESPIDER_BARRIER_DIR"] = _CONFIG["barrier_dir"]
    process = CrawlerProcess(settings=settings)
    process.crawl(InflightSpider)
    process.start()


if __name__ == "__main__":
    main()
