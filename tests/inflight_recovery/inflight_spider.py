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

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from scrapy import Request, Spider, signals
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

    def _maybe_kill(self, stage: str) -> None:
        if self.kill_stage == stage:
            sys.stderr.flush()
            os._exit(1)

    async def start(self):
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
