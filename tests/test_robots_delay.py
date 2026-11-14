"""Tests for enforcing robots.txt delay directives in the downloader.

Timing uses intervals of 0.2 to 0.5 seconds and asserts lower bounds rather
than exact moments, so that the tests do not depend on scheduler precision.
"""

from __future__ import annotations

import asyncio
from time import monotonic
from typing import Any

from scrapy import Request, Spider
from scrapy.http import Response
from scrapy.utils.asyncio import sleep
from scrapy.utils.test import get_crawler
from tests.utils.decorators import coroutine_test

TOLERANCE = 0.01


class _RecordingHandler:
    """Download handler that records when each download starts, and returns a
    response immediately.

    Tests configure the class attributes before starting a crawl, so that the
    instance the downloader builds sees them.
    """

    lazy = False

    starts: list[tuple[str, float]] = []
    robots: dict[str, bytes] = {}
    fail_robots = False
    override: Any = None

    @classmethod
    def reset(cls) -> None:
        cls.starts = []
        cls.robots = {}
        cls.fail_robots = False
        cls.override = None

    @classmethod
    def from_crawler(cls, crawler: Any) -> _RecordingHandler:
        return cls()

    async def download_request(self, request: Request) -> Response:
        if type(self).override is not None:
            return await type(self).override(request)
        type(self).starts.append((request.url, monotonic()))
        if request.url.endswith("/robots.txt"):
            if type(self).fail_robots:
                raise RuntimeError("robots.txt download failed")
            return Response(request.url, body=type(self).robots.get(request.url, b""))
        return Response(request.url)

    async def close(self) -> None:
        pass

    @classmethod
    def start_times(cls, host: str | None = None) -> list[float]:
        return [
            started
            for url, started in cls.starts
            if not url.endswith("/robots.txt")
            and (host is None or f"://{host}/" in url)
        ]


_HANDLER = "tests.test_robots_delay._RecordingHandler"


def _crawler(spidercls: type[Spider], **settings: Any) -> Any:
    return get_crawler(
        spidercls,
        {
            "ROBOTSTXT_OBEY": True,
            "ROBOTSTXT_OBEY_DELAY": True,
            "DOWNLOAD_HANDLERS": {"http": _HANDLER},
            "DOWNLOAD_DELAY": 0,
            "DOWNLOAD_DELAY_JITTER": 0,
        }
        | settings,
    )


class _SiteSpider(Spider):
    """Spider that requests the URLs in its ``urls`` argument."""

    name = "site"

    def __init__(
        self,
        *args: Any,
        urls: list[str],
        meta: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.urls = urls
        self.meta = meta or {}

    async def start(self):
        for url in self.urls:
            yield Request(url, meta=dict(self.meta))

    def parse(self, response: Response) -> None:
        pass


def _gaps(times: list[float]) -> list[float]:
    return [later - earlier for earlier, later in zip(times, times[1:])]


def _assert_interval(times: list[float], interval: float) -> None:
    assert len(times) >= 2
    assert all(gap >= interval - TOLERANCE for gap in _gaps(times)), _gaps(times)


@coroutine_test
async def test_crawl_delay_enforced() -> None:
    _RecordingHandler.reset()
    _RecordingHandler.robots["http://a.example/robots.txt"] = (
        b"User-agent: *\nCrawl-delay: 0.3\n"
    )
    crawler = _crawler(_SiteSpider, CONCURRENT_REQUESTS=4)
    await crawler.crawl_async(urls=[f"http://a.example/{i}" for i in range(4)])
    times = _RecordingHandler.start_times()
    assert len(times) == 4
    _assert_interval(times, 0.3)
    assert crawler.stats
    assert crawler.stats.get_value("robotstxt/delayed_count") == 4


@coroutine_test
async def test_request_rate_enforced() -> None:
    _RecordingHandler.reset()
    # 2 requests every 1 second means one every 0.5 seconds.
    _RecordingHandler.robots["http://a.example/robots.txt"] = (
        b"User-agent: *\nRequest-rate: 2/1\n"
    )
    crawler = _crawler(_SiteSpider, CONCURRENT_REQUESTS=4)
    await crawler.crawl_async(urls=[f"http://a.example/{i}" for i in range(4)])
    times = _RecordingHandler.start_times()
    assert len(times) == 4
    # 2 requests every 1 second means one every 0.5 seconds.
    _assert_interval(times, 0.5)


@coroutine_test
async def test_request_rate_is_merged_with_crawl_delay() -> None:
    """The interval is the larger of Crawl-delay and the Request-rate
    period."""
    _RecordingHandler.reset()
    _RecordingHandler.robots["http://a.example/robots.txt"] = (
        b"User-agent: *\nCrawl-delay: 0.15\nRequest-rate: 1/1\n"
    )
    crawler = _crawler(_SiteSpider, CONCURRENT_REQUESTS=3)
    await crawler.crawl_async(urls=[f"http://a.example/{i}" for i in range(3)])
    _assert_interval(_RecordingHandler.start_times(), 1.0)


@coroutine_test
async def test_robots_download_counts_as_start() -> None:
    _RecordingHandler.reset()
    _RecordingHandler.robots["http://a.example/robots.txt"] = (
        b"User-agent: *\nCrawl-delay: 0.3\n"
    )
    crawler = _crawler(_SiteSpider, CONCURRENT_REQUESTS=2)
    await crawler.crawl_async(urls=["http://a.example/0"])
    robots_start = next(
        t for u, t in _RecordingHandler.starts if u.endswith("/robots.txt")
    )
    (page_start,) = _RecordingHandler.start_times()
    assert page_start - robots_start >= 0.3 - TOLERANCE


@coroutine_test
async def test_obey_delay_off() -> None:
    _RecordingHandler.reset()
    _RecordingHandler.robots["http://a.example/robots.txt"] = (
        b"User-agent: *\nCrawl-delay: 1\n"
    )
    crawler = _crawler(
        _SiteSpider, ROBOTSTXT_OBEY_DELAY=False, CONCURRENT_REQUESTS=4
    )
    await crawler.crawl_async(urls=[f"http://a.example/{i}" for i in range(4)])
    times = _RecordingHandler.start_times()
    assert len(times) == 4
    assert max(_gaps(times)) < 0.2
    assert crawler.stats
    assert crawler.stats.get_value("robotstxt/delayed_count") is None


@coroutine_test
async def test_obey_delay_requires_obey() -> None:
    _RecordingHandler.reset()
    _RecordingHandler.robots["http://a.example/robots.txt"] = (
        b"User-agent: *\nCrawl-delay: 1\n"
    )
    crawler = _crawler(_SiteSpider, ROBOTSTXT_OBEY=False, CONCURRENT_REQUESTS=4)
    await crawler.crawl_async(urls=[f"http://a.example/{i}" for i in range(4)])
    times = _RecordingHandler.start_times()
    assert len(times) == 4
    assert max(_gaps(times)) < 0.2


@coroutine_test
async def test_robots_error_means_no_delay() -> None:
    _RecordingHandler.reset()
    _RecordingHandler.robots["http://a.example/robots.txt"] = (
        b"User-agent: *\nCrawl-delay: 0.5\n"
    )
    _RecordingHandler.fail_robots = True
    crawler = _crawler(_SiteSpider, CONCURRENT_REQUESTS=4, RETRY_ENABLED=False)
    await crawler.crawl_async(urls=[f"http://a.example/{i}" for i in range(4)])
    times = _RecordingHandler.start_times()
    assert len(times) == 4
    assert max(_gaps(times)) < 0.2


@coroutine_test
async def test_no_throttling_of_other_site() -> None:
    """Waiting for one site's interval must not hold up requests to another
    site, even with a low CONCURRENT_REQUESTS."""

    class TwoSiteSpider(Spider):
        name = "two_sites"

        async def start(self):
            for i in range(10):
                yield Request(f"http://a.example/{i}")
            for i in range(10):
                yield Request(f"http://b.example/{i}")

        def parse(self, response: Response) -> None:
            pass

    _RecordingHandler.reset()
    _RecordingHandler.robots["http://a.example/robots.txt"] = (
        b"User-agent: *\nCrawl-delay: 0.5\n"
    )
    _RecordingHandler.robots["http://b.example/robots.txt"] = b"User-agent: *\n"
    crawler = _crawler(TwoSiteSpider, CONCURRENT_REQUESTS=2)
    await crawler.crawl_async()

    a_times = _RecordingHandler.start_times("a.example")
    b_times = _RecordingHandler.start_times("b.example")
    assert len(a_times) == 10
    assert len(b_times) == 10
    # All of B finishes before the third A request starts.
    assert max(b_times) < sorted(a_times)[2]


@coroutine_test
async def test_delay_across_download_slots() -> None:
    """The robots.txt interval applies across download slots of the same
    site."""

    class SlottedSpider(Spider):
        name = "slotted"

        async def start(self):
            for i in range(4):
                yield Request(
                    f"http://a.example/{i}", meta={"download_slot": f"slot{i % 2}"}
                )

        def parse(self, response: Response) -> None:
            pass

    _RecordingHandler.reset()
    _RecordingHandler.robots["http://a.example/robots.txt"] = (
        b"User-agent: *\nCrawl-delay: 0.3\n"
    )
    crawler = _crawler(SlottedSpider, CONCURRENT_REQUESTS=8)
    await crawler.crawl_async()
    times = _RecordingHandler.start_times()
    assert len(times) == 4
    _assert_interval(times, 0.3)


@coroutine_test
async def test_mixed_sites_in_one_slot() -> None:
    """A slot shared by several sites applies each site's interval
    separately."""

    class MixedSpider(Spider):
        name = "mixed"

        async def start(self):
            for i in range(4):
                meta = {"download_slot": "shared"}
                yield Request(f"http://a.example/{i}", meta=meta)
                yield Request(f"http://b.example/{i}", meta=meta)

        def parse(self, response: Response) -> None:
            pass

    _RecordingHandler.reset()
    _RecordingHandler.robots["http://a.example/robots.txt"] = (
        b"User-agent: *\nCrawl-delay: 0.3\n"
    )
    _RecordingHandler.robots["http://b.example/robots.txt"] = (
        b"User-agent: *\nCrawl-delay: 0.3\n"
    )
    crawler = _crawler(MixedSpider, CONCURRENT_REQUESTS=8)
    await crawler.crawl_async()
    for host in ("a.example", "b.example"):
        times = _RecordingHandler.start_times(host)
        assert len(times) == 4
        _assert_interval(times, 0.3)


@coroutine_test
async def test_redirect_uses_new_site_interval() -> None:
    """A redirect request is throttled according to the robots.txt file of its
    own URL, not the one of the request that redirected."""
    _RecordingHandler.reset()
    _RecordingHandler.robots["http://a.example/robots.txt"] = (
        b"User-agent: *\nCrawl-delay: 0.3\n"
    )
    _RecordingHandler.robots["http://b.example/robots.txt"] = (
        b"User-agent: *\nCrawl-delay: 0.3\n"
    )

    class RedirectSpider(Spider):
        name = "redirect"

        async def start(self):
            yield Request("http://a.example/start")
            yield Request("http://b.example/0")

        def parse(self, response: Response) -> None:
            pass

    async def download_request(request: Request) -> Response:
        _RecordingHandler.starts.append((request.url, monotonic()))
        if request.url.endswith("/robots.txt"):
            return Response(
                request.url, body=_RecordingHandler.robots.get(request.url, b"")
            )
        if request.url == "http://a.example/start":
            return Response(
                request.url,
                status=302,
                headers={"Location": "http://b.example/target"},
            )
        return Response(request.url)

    _RecordingHandler.override = download_request
    crawler = _crawler(RedirectSpider, CONCURRENT_REQUESTS=4)
    await crawler.crawl_async()
    b_times = _RecordingHandler.start_times("b.example")
    # Both the original b.example request and the redirect target are started
    # at least 0.3 s after the b.example robots.txt download.
    assert len(b_times) == 2
    _assert_interval(b_times, 0.3)


@coroutine_test
async def test_retry_uses_own_interval() -> None:
    """A retry is throttled according to its own URL's robots.txt file."""
    _RecordingHandler.reset()
    _RecordingHandler.robots["http://a.example/robots.txt"] = (
        b"User-agent: *\nCrawl-delay: 0.3\n"
    )

    class RetrySpider(Spider):
        name = "retry"

        async def start(self):
            yield Request("http://a.example/fail")
            yield Request("http://a.example/0")

        def parse(self, response: Response) -> None:
            pass

    attempts = {"n": 0}

    async def download_request(request: Request) -> Response:
        _RecordingHandler.starts.append((request.url, monotonic()))
        if request.url.endswith("/robots.txt"):
            return Response(
                request.url, body=_RecordingHandler.robots.get(request.url, b"")
            )
        if request.url.endswith("/fail") and attempts["n"] == 0:
            attempts["n"] += 1
            return Response(request.url, status=500)
        return Response(request.url)

    _RecordingHandler.override = download_request
    crawler = _crawler(
        RetrySpider,
        CONCURRENT_REQUESTS=4,
        RETRY_TIMES=1,
        RETRY_PRIORITY_ADJUST=0,
    )
    await crawler.crawl_async()
    times = _RecordingHandler.start_times()
    # /fail, its retry, and /0, all governed by the same robots.txt file.
    assert len(times) == 3
    _assert_interval(times, 0.3)


@coroutine_test
async def test_priority_order_among_waiting_requests() -> None:
    """A higher priority request is granted before a lower priority one that
    is waiting at the same time."""

    class PrioritySpider(Spider):
        name = "priority"

        async def start(self):
            for i in range(3):
                yield Request(f"http://a.example/low{i}", priority=0)
            for i in range(3):
                yield Request(f"http://a.example/high{i}", priority=10)

        def parse(self, response: Response) -> None:
            pass

    _RecordingHandler.reset()
    _RecordingHandler.robots["http://a.example/robots.txt"] = (
        b"User-agent: *\nCrawl-delay: 0.3\n"
    )
    crawler = _crawler(PrioritySpider, CONCURRENT_REQUESTS=8)
    await crawler.crawl_async()
    order = [
        url
        for url, _ in sorted(_RecordingHandler.starts, key=lambda x: x[1])
        if not url.endswith("/robots.txt")
    ]
    assert order == [
        "http://a.example/high0",
        "http://a.example/high1",
        "http://a.example/high2",
        "http://a.example/low0",
        "http://a.example/low1",
        "http://a.example/low2",
    ]


@coroutine_test
async def test_idle_crawl_with_waiting_requests() -> None:
    """A crawl whose only pending requests are waiting for their interval does
    not get closed as idle."""
    _RecordingHandler.reset()
    _RecordingHandler.robots["http://a.example/robots.txt"] = (
        b"User-agent: *\nCrawl-delay: 0.3\n"
    )
    crawler = _crawler(_SiteSpider, CONCURRENT_REQUESTS=4)
    await crawler.crawl_async(urls=[f"http://a.example/{i}" for i in range(3)])
    assert len(_RecordingHandler.start_times()) == 3
    assert crawler.stats
    assert crawler.stats.get_value("finish_reason") == "finished"


@coroutine_test
async def test_dont_obey_robotstxt_does_not_wait() -> None:
    _RecordingHandler.reset()
    _RecordingHandler.robots["http://a.example/robots.txt"] = (
        b"User-agent: *\nCrawl-delay: 1\n"
    )
    crawler = _crawler(_SiteSpider, CONCURRENT_REQUESTS=4)
    await crawler.crawl_async(
        urls=[f"http://a.example/{i}" for i in range(4)],
        meta={"dont_obey_robotstxt": True},
    )
    times = _RecordingHandler.start_times()
    assert len(times) == 4
    assert max(_gaps(times)) < 0.2


@coroutine_test
async def test_data_url_does_not_wait() -> None:
    class DataSpider(Spider):
        name = "data"

        async def start(self):
            yield Request("data:,hello")

        def parse(self, response: Response) -> None:
            pass

    crawler = get_crawler(
        DataSpider,
        {
            "ROBOTSTXT_OBEY": True,
            "ROBOTSTXT_OBEY_DELAY": True,
            "DOWNLOAD_DELAY": 0,
            "DOWNLOAD_DELAY_JITTER": 0,
        },
    )
    await crawler.crawl_async()
    assert crawler.stats
    assert crawler.stats.get_value("robotstxt/delayed_count") is None


@coroutine_test
async def test_http_cache_hit_does_not_wait(tmp_path: Any) -> None:
    """A request served from the HTTP cache is not downloaded, so it neither
    waits for nor advances the robots.txt interval."""
    _RecordingHandler.reset()
    _RecordingHandler.robots["http://a.example/robots.txt"] = (
        b"User-agent: *\nCrawl-delay: 0.3\n"
    )

    class CachedSpider(Spider):
        name = "cached"

        async def start(self):
            # The duplicate is filtered out unless filtering is disabled; a
            # cache hit also needs the first response to be cacheable.
            yield Request("http://a.example/x", dont_filter=True)
            yield Request("http://a.example/x", dont_filter=True)
            yield Request("http://a.example/y")

        def parse(self, response: Response) -> None:
            pass

    async def download_request(request: Request) -> Response:
        _RecordingHandler.starts.append((request.url, monotonic()))
        if request.url.endswith("/robots.txt"):
            return Response(
                request.url, body=_RecordingHandler.robots.get(request.url, b"")
            )
        return Response(
            request.url,
            body=b"ok",
            headers={"Cache-Control": "max-age=3600"},
        )

    _RecordingHandler.override = download_request
    crawler = _crawler(
        CachedSpider,
        CONCURRENT_REQUESTS=4,
        HTTPCACHE_ENABLED=True,
        HTTPCACHE_DIR=str(tmp_path),
        HTTPCACHE_EXPIRATION_SECS=3600,
    )
    await crawler.crawl_async()
    assert crawler.stats
    assert crawler.stats.get_value("httpcache/hit") == 1
    # The two downloaded pages are /x and /y, 0.3 s apart; the cache hit
    # neither waited nor pushed back /y.
    times = _RecordingHandler.start_times()
    assert len(times) == 2
    _assert_interval(times, 0.3)


@coroutine_test
async def test_delay_capped() -> None:
    _RecordingHandler.reset()
    _RecordingHandler.robots["http://a.example/robots.txt"] = (
        b"User-agent: *\nCrawl-delay: 5\n"
    )
    crawler = _crawler(_SiteSpider, ROBOTSTXT_MAX_DELAY=0.3, CONCURRENT_REQUESTS=2)
    await crawler.crawl_async(urls=[f"http://a.example/{i}" for i in range(2)])
    assert crawler.stats
    assert crawler.stats.get_value("robotstxt/delay_capped") == 1
    times = _RecordingHandler.start_times()
    assert len(times) == 2
    assert times[1] - times[0] >= 0.3 - TOLERANCE
    assert times[1] - times[0] < 5


@coroutine_test
async def test_close_does_not_wait_for_intervals(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Closing the spider drops waiting requests instead of waiting out their
    interval."""
    _RecordingHandler.reset()
    _RecordingHandler.robots["http://a.example/robots.txt"] = (
        b"User-agent: *\nCrawl-delay: 5\n"
    )
    crawler = _crawler(_SiteSpider, CONCURRENT_REQUESTS=4)

    async def stop_soon() -> None:
        await sleep(0.3)
        await crawler.stop_async()

    stop_task = asyncio.ensure_future(stop_soon())
    try:
        start = monotonic()
        await crawler.crawl_async(urls=[f"http://a.example/{i}" for i in range(5)])
        elapsed = monotonic() - start
    finally:
        await stop_task
    assert elapsed < 1.0
    # Only the robots.txt download and at most the granted request started.
    assert len(_RecordingHandler.start_times()) <= 1
    # Dropping the waiting requests is part of the normal close, not a
    # download error.
    assert not [r for r in caplog.records if r.levelname == "ERROR"]


@coroutine_test
async def test_slot_gc_and_rebuild_keeps_interval() -> None:
    """Removing and rebuilding a download slot does not reset the robots.txt
    interval of its site."""
    _RecordingHandler.reset()
    _RecordingHandler.robots["http://a.example/robots.txt"] = (
        b"User-agent: *\nCrawl-delay: 0.3\n"
    )

    class RebuildSpider(Spider):
        name = "rebuild"

        async def start(self):
            yield Request("http://a.example/0", callback=self.first)

        def first(self, response: Response):
            downloader = self.crawler.engine.downloader
            # Force the slot of the site to be collected and rebuilt.
            key = downloader.get_slot_key(response.request)
            downloader.slots.pop(key).close()
            yield Request("http://a.example/1")

        def parse(self, response: Response) -> None:
            pass

    crawler = _crawler(RebuildSpider, CONCURRENT_REQUESTS=2)
    await crawler.crawl_async()
    times = _RecordingHandler.start_times()
    assert len(times) == 2
    assert times[1] - times[0] >= 0.3 - TOLERANCE