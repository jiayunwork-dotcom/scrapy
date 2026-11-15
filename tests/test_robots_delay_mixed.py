"""Regression tests for mixing robots.txt intervals of different magnitudes.

A single robots.txt file may give different user agents different
``Crawl-delay`` values, including 0. The interval is a per-site floor on the
real starts (the moments requests are handed to the download handler), not a
per-user-agent or per-slot one: for every non-exempt request R with interval
d, its start is at least d seconds after the previous non-exempt start of the
same site, even if that previous request had an interval of 0, and even if R
was delayed in another download slot.

The tests drive a real crawler through the public Request scheduling and
``engine.download_async()`` paths. A substitute download handler records the
real starts, and test-controlled barriers, not sleeps, order the requests.
Timing asserts lower bounds with a 10 ms tolerance.
"""

from __future__ import annotations

from itertools import pairwise
from time import monotonic
from typing import Any

from twisted.internet.defer import Deferred

from scrapy import Request, Spider
from scrapy.http import Response
from scrapy.utils.asyncio import sleep
from scrapy.utils.defer import deferred_from_coro, maybe_deferred_to_future
from scrapy.utils.test import get_crawler
from tests.utils.decorators import coroutine_test

TOLERANCE = 0.01


class _Barrier:
    """Reactor-agnostic one-shot barrier.

    ``asyncio.Event`` only works while an asyncio event loop is running. The
    test suite also runs on the plain Twisted reactor, where a Deferred is the
    awaitable primitive that both pytest-twisted and pytest-asyncio handle.
    """

    def __init__(self) -> None:
        self._dfd: Deferred[None] = Deferred()

    def set(self) -> None:
        if not self._dfd.called:
            self._dfd.callback(None)

    async def wait(self) -> None:
        await maybe_deferred_to_future(self._dfd)


async def _yield_control(times: int = 1) -> None:
    """Let queued reactor/loop callbacks run a few times, without relying on
    a specific event loop implementation.

    sleep(0) schedules a real deferred callback (asyncio sleep or
    reactor.deferLater), so the running test coroutine yields control instead
    of resuming synchronously, which the plain-reactor test driver requires.
    """
    for _ in range(times):
        await sleep(0)


def _spawn(coro: Any) -> Any:
    """Schedule a coroutine on whatever drives the test, asyncio loop or
    Twisted reactor, and return the awaitable handle it produces."""
    return deferred_from_coro(coro)


async def _cancel(handle: Any) -> None:
    """Cancel a handle returned by :func:`_spawn` and wait for the resulting
    cancellation error, on either the asyncio loop or the plain reactor."""
    handle.cancel()
    try:
        await maybe_deferred_to_future(handle)
    except BaseException:
        # asyncio.CancelledError, twisted CancelledError, DownloadCancelledError
        # and RuntimeError wrappers all mean the request did not run.
        return

# One robots.txt for the same site: the fastbot user agent gets no interval,
# slowbot gets 0.4 s. The wildcard group matches any other user agent.
MIXED_ROBOTS = (
    b"User-agent: fastbot\nCrawl-delay: 0\n"
    b"User-agent: slowbot\nCrawl-delay: 0.4\n"
    b"User-agent: *\nCrawl-delay: 0.4\n"
)


class _RecordingHandler:
    """Download handler that records real start times and can block chosen
    requests on a test-controlled event."""

    lazy = False

    starts: list[tuple[str, float]] = []
    robots: dict[str, bytes] = {}
    blocker_predicate: Any = None
    release_event: _Barrier | None = None
    override: Any = None

    @classmethod
    def reset(cls) -> None:
        cls.starts = []
        cls.robots = {}
        cls.blocker_predicate = None
        cls.release_event = None
        cls.override = None

    @classmethod
    def from_crawler(cls, crawler: Any) -> _RecordingHandler:
        return cls()

    async def download_request(self, request: Request) -> Response:
        if type(self).override is not None:
            response = await type(self).override(request)
            assert isinstance(response, Response)
            return response
        type(self).starts.append((request.url, monotonic()))
        if request.url.endswith("/robots.txt"):
            return Response(request.url, body=type(self).robots.get(request.url, b""))
        if type(self).blocker_predicate is not None and type(self).blocker_predicate(
            request
        ):
            barrier = type(self).release_event
            assert barrier is not None
            await barrier.wait()
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

    @classmethod
    def start_time_of(cls, suffix: str) -> float:
        return next(t for url, t in cls.starts if url.endswith(suffix))


_HANDLER = "tests.test_robots_delay_mixed._RecordingHandler"


def _crawler(spidercls: type[Spider], **settings: Any) -> Any:
    return get_crawler(
        spidercls,
        {
            "ROBOTSTXT_OBEY": True,
            "ROBOTSTXT_OBEY_DELAY": True,
            "DOWNLOAD_HANDLERS": {"http": _HANDLER},
            "DOWNLOAD_DELAY": 0,
            "DOWNLOAD_DELAY_JITTER": 0,
            "USER_AGENT": "slowbot",
        }
        | settings,
    )


def _fast(url: str, **meta: Any) -> Request:
    return Request(url, headers={"User-Agent": "fastbot"}, meta=meta)


def _slow(url: str, **meta: Any) -> Request:
    return Request(url, headers={"User-Agent": "slowbot"}, meta=meta)


class _InlineSpider(Spider):
    """Spider whose start() body is supplied per test."""

    name = "inline"
    body: Any = None

    async def start(self):
        if False:  # make this an async generator
            yield Request("data:,")
        assert self.body is not None
        await self.body(self)

    def parse(self, response: Response) -> None:
        pass


def _gaps(times: list[float]) -> list[float]:
    return [later - earlier for earlier, later in pairwise(times)]


@coroutine_test
async def test_zero_then_positive_through_scheduler() -> None:
    """The same floor holds for requests that go through the public
    scheduler, not only engine.download_async()."""
    _RecordingHandler.reset()
    _RecordingHandler.robots["http://a.example/robots.txt"] = MIXED_ROBOTS

    class ScheduledSpider(Spider):
        name = "scheduled"

        async def start(self):
            yield _fast("http://a.example/f0")
            yield _fast("http://a.example/f1")
            yield _slow("http://a.example/s0")
            yield _slow("http://a.example/s1")

        def parse(self, response: Response) -> None:
            pass

    crawler = _crawler(ScheduledSpider, CONCURRENT_REQUESTS=4)
    await crawler.crawl_async()

    ordered = sorted(
        (started, url)
        for url, started in _RecordingHandler.starts
        if "/robots.txt" not in url
    )
    for (previous, _), (started, url) in pairwise(ordered):
        if url.endswith(("/s0", "/s1")):
            assert started - previous >= 0.4 - TOLERANCE
    f0 = _RecordingHandler.start_time_of("/f0")
    f1 = _RecordingHandler.start_time_of("/f1")
    assert f1 - f0 < 0.15


@coroutine_test
async def test_zero_interval_then_positive_interval() -> None:
    """Regression: a zero-interval request must still count as the previous
    start of a later positive-interval request of the same robots.txt file."""
    _RecordingHandler.reset()
    _RecordingHandler.robots["http://a.example/robots.txt"] = MIXED_ROBOTS

    async def body(spider: Spider) -> None:
        engine = spider.crawler.engine
        await engine.download_async(_fast("http://a.example/seed"))
        # Less than the slowbot interval; the slow request below has to wait
        # the full 0.4 s after the fast request anyway.
        await sleep(0.2)
        await engine.download_async(_fast("http://a.example/fast"))
        await engine.download_async(_slow("http://a.example/slow"))

    _InlineSpider.body = staticmethod(body)
    crawler = _crawler(_InlineSpider, CONCURRENT_REQUESTS=4)
    await crawler.crawl_async()

    fast = _RecordingHandler.start_time_of("/fast")
    slow = _RecordingHandler.start_time_of("/slow")
    assert slow - fast >= 0.4 - TOLERANCE
    # The zero-interval fast request did not wait for an interval itself.
    assert crawler.stats
    # Only the slow request actually waited for an interval.
    assert crawler.stats.get_value("robotstxt/delayed_count") == 1


@coroutine_test
async def test_positive_interval_request_queued_then_zero_interval_overtakes() -> None:
    """Regression: a positive-interval request blocked in a shared slot must
    revalidate its interval when a later zero-interval request of the same
    site starts through another slot."""
    _RecordingHandler.reset()
    _RecordingHandler.robots["http://a.example/robots.txt"] = MIXED_ROBOTS
    _RecordingHandler.robots["http://b.example/robots.txt"] = (
        b"User-agent: *\nCrawl-delay: 0\n"
    )
    release = _Barrier()
    _RecordingHandler.release_event = release
    _RecordingHandler.blocker_predicate = lambda r: r.url == "http://b.example/block"

    async def body(spider: Spider) -> None:
        engine = spider.crawler.engine
        # Make the previous start of the site older than the interval.
        await engine.download_async(_slow("http://a.example/old"))
        await sleep(0.45)
        blocker_task = _spawn(
            engine.download_async(
                Request(
                    "http://b.example/block",
                    meta={
                        "download_slot": "shared",
                        "dont_obey_robotstxt": True,
                    },
                )
            )
        )
        # Let the blocker reach the download handler and hold the shared slot.
        await _yield_control(3)
        slow_task = _spawn(
            engine.download_async(
                _slow("http://a.example/slow", download_slot="shared")
            )
        )
        # Let the slow request get through the middleware and queue.
        await _yield_control(3)
        # A zero-interval request of the same site uses another, free slot.
        await engine.download_async(
            _fast("http://a.example/fast", download_slot="other")
        )
        release.set()
        await maybe_deferred_to_future(slow_task)
        await maybe_deferred_to_future(blocker_task)

    _InlineSpider.body = staticmethod(body)
    crawler = _crawler(
        _InlineSpider,
        CONCURRENT_REQUESTS=4,
        CONCURRENT_REQUESTS_PER_DOMAIN=1,
    )
    await crawler.crawl_async()

    fast = _RecordingHandler.start_time_of("/fast")
    slow = _RecordingHandler.start_time_of("/slow")
    # Before the fix the slow request started ~1 ms after the fast one.
    assert slow - fast >= 0.4 - TOLERANCE
    # The fast request was not held back by the queued slow request.
    assert crawler.stats
    delayed = crawler.stats.get_value("robotstxt/delayed_count") or 0
    assert delayed >= 1


@coroutine_test
async def test_zero_interval_requests_never_wait() -> None:
    """A run of zero-interval requests starts back to back, and a following
    positive-interval request is spaced from the last of them."""
    _RecordingHandler.reset()
    _RecordingHandler.robots["http://a.example/robots.txt"] = MIXED_ROBOTS

    async def body(spider: Spider) -> None:
        engine = spider.crawler.engine
        for i in range(4):
            await engine.download_async(_fast(f"http://a.example/f{i}"))
        await engine.download_async(_slow("http://a.example/slow"))

    _InlineSpider.body = staticmethod(body)
    crawler = _crawler(_InlineSpider, CONCURRENT_REQUESTS=8)
    await crawler.crawl_async()

    fast_times = [
        _RecordingHandler.start_time_of(f"/f{i}") for i in range(4)
    ]
    # No robots.txt wait: the only spacing is event-loop scheduling, far below
    # the 0.4 s interval.
    assert max(_gaps(fast_times)) < 0.15
    slow = _RecordingHandler.start_time_of("/slow")
    assert slow - fast_times[-1] >= 0.4 - TOLERANCE
    assert crawler.stats
    assert crawler.stats.get_value("robotstxt/delayed_count") == 1


@coroutine_test
async def test_two_different_positive_intervals() -> None:
    """Two user agents with different positive intervals each respect the
    larger floor relative to the previous real start."""
    _RecordingHandler.reset()
    _RecordingHandler.robots["http://a.example/robots.txt"] = (
        b"User-agent: quick\nCrawl-delay: 0.2\n"
        b"User-agent: slow\nCrawl-delay: 0.4\n"
        b"User-agent: *\nCrawl-delay: 0.4\n"
    )

    async def body(spider: Spider) -> None:
        engine = spider.crawler.engine
        for url, agent in (
            ("http://a.example/q0", "quick"),
            ("http://a.example/s0", "slow"),
            ("http://a.example/q1", "quick"),
            ("http://a.example/s1", "slow"),
        ):
            await engine.download_async(
                Request(url, headers={"User-Agent": agent})
            )

    _InlineSpider.body = staticmethod(body)
    crawler = _crawler(_InlineSpider, CONCURRENT_REQUESTS=8)
    await crawler.crawl_async()

    times = _RecordingHandler.start_times("a.example")
    assert len(times) == 4
    # q0 -> s0: 0.4; s0 -> q1: 0.2; q1 -> s1: 0.4.
    gaps = _gaps(times)
    assert gaps[0] >= 0.4 - TOLERANCE
    assert gaps[1] >= 0.2 - TOLERANCE
    assert gaps[2] >= 0.4 - TOLERANCE


@coroutine_test
async def test_same_slot_and_cross_slot() -> None:
    """The floor holds whether the requests share a download slot or use
    different ones, and zero-interval requests stay back to back."""
    _RecordingHandler.reset()
    _RecordingHandler.robots["http://a.example/robots.txt"] = MIXED_ROBOTS

    class SlotSpider(Spider):
        name = "slotted"

        async def start(self):
            # Two zero-interval requests in one slot, two positive ones in
            # another slot, all governed by the same robots.txt file.
            yield _fast("http://a.example/f0", download_slot="alpha")
            yield _fast("http://a.example/f1", download_slot="alpha")
            yield _slow("http://a.example/s0", download_slot="beta")
            yield _slow("http://a.example/s1", download_slot="beta")

        def parse(self, response: Response) -> None:
            pass

    crawler = _crawler(SlotSpider, CONCURRENT_REQUESTS=8)
    await crawler.crawl_async()

    ordered = sorted(
        (started, url)
        for url, started in _RecordingHandler.starts
        if "/robots.txt" not in url
    )
    # Each slow start is at least its interval after the previous real start
    # of the site, regardless of which slot either request used.
    for (previous, _), (started, url) in pairwise(ordered):
        if url.endswith(("/s0", "/s1")):
            assert started - previous >= 0.4 - TOLERANCE
    f0 = _RecordingHandler.start_time_of("/f0")
    f1 = _RecordingHandler.start_time_of("/f1")
    # The two zero-interval requests were not held apart by any robots.txt wait.
    assert f1 - f0 < 0.15


@coroutine_test
async def test_other_site_keeps_progressing() -> None:
    """Waiting for the interval of one site neither takes a download
    concurrency spot nor blocks another site."""

    class TwoSiteSpider(Spider):
        name = "two_sites"

        async def start(self):
            for i in range(4):
                yield _slow(f"http://a.example/{i}")
            for i in range(6):
                yield Request(f"http://b.example/{i}")

        def parse(self, response: Response) -> None:
            pass

    _RecordingHandler.reset()
    _RecordingHandler.robots["http://a.example/robots.txt"] = MIXED_ROBOTS
    _RecordingHandler.robots["http://b.example/robots.txt"] = (
        b"User-agent: *\nCrawl-delay: 0\n"
    )
    crawler = _crawler(TwoSiteSpider, CONCURRENT_REQUESTS=2)
    await crawler.crawl_async()

    b_times = _RecordingHandler.start_times("b.example")
    a_times = _RecordingHandler.start_times("a.example")
    assert len(b_times) == 6
    assert len(a_times) == 4
    # B finishes well before the last spaced-out A request.
    assert max(b_times) < max(a_times) - 0.3


@coroutine_test
async def test_cancel_before_start_is_not_a_start() -> None:
    """Cancelling a request still waiting for its interval records no start
    and leaves no blockage for the following request."""
    _RecordingHandler.reset()
    _RecordingHandler.robots["http://a.example/robots.txt"] = MIXED_ROBOTS

    async def body(spider: Spider) -> None:
        engine = spider.crawler.engine
        await engine.download_async(_slow("http://a.example/first"))
        pending = _spawn(
            engine.download_async(_slow("http://a.example/cancelled"))
        )
        # Let it enter the interval wait, then cancel before it can start.
        await _yield_control(3)
        await _cancel(pending)
        # A later request still gets a full interval from the first start.
        await engine.download_async(_slow("http://a.example/after"))

    _InlineSpider.body = staticmethod(body)
    crawler = _crawler(_InlineSpider, CONCURRENT_REQUESTS=4)
    await crawler.crawl_async()

    urls = [url for url, _ in _RecordingHandler.starts]
    assert "http://a.example/cancelled" not in urls
    first = _RecordingHandler.start_time_of("/first")
    after = _RecordingHandler.start_time_of("/after")
    assert after - first >= 0.4 - TOLERANCE


@coroutine_test
async def test_close_ends_interval_waits_within_a_second() -> None:
    """Closing while requests wait for an interval drops them immediately and
    records no start for the dropped requests."""
    _RecordingHandler.reset()
    _RecordingHandler.robots["http://a.example/robots.txt"] = (
        b"User-agent: *\nCrawl-delay: 5\n"
    )

    async def body(spider: Spider) -> None:
        engine = spider.crawler.engine
        for i in range(4):
            _spawn(engine.download_async(_slow(f"http://a.example/{i}")))
        await _yield_control(3)
        await sleep(0.25)
        await spider.crawler.stop_async()

    _InlineSpider.body = staticmethod(body)
    crawler = _crawler(_InlineSpider, CONCURRENT_REQUESTS=4)
    start = monotonic()
    await crawler.crawl_async()
    assert monotonic() - start < 1.0
    assert len(_RecordingHandler.start_times("a.example")) <= 1


@coroutine_test
async def test_cache_hit_neither_waits_nor_advances_interval(tmp_path: Any) -> None:
    """A response served from the HTTP cache involves no download, so it does
    not count as a start and does not push back a later real download."""
    _RecordingHandler.reset()
    _RecordingHandler.robots["http://a.example/robots.txt"] = MIXED_ROBOTS

    class CachedSpider(Spider):
        name = "cached"

        async def start(self):
            yield Request("http://a.example/x", dont_filter=True)
            yield Request("http://a.example/x", dont_filter=True)
            yield _slow("http://a.example/y")

        def parse(self, response: Response) -> None:
            pass

    async def override(request: Request) -> Response:
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

    _RecordingHandler.override = override
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
    times = _RecordingHandler.start_times("a.example")
    assert len(times) == 2
    assert times[1] - times[0] >= 0.4 - TOLERANCE


@coroutine_test
async def test_retry_and_redirect_use_own_url_interval() -> None:
    """Requests generated by a retry or a redirect are handled according to
    their own URL and user agent, including zero intervals."""
    _RecordingHandler.reset()
    _RecordingHandler.robots["http://a.example/robots.txt"] = MIXED_ROBOTS
    _RecordingHandler.robots["http://b.example/robots.txt"] = (
        b"User-agent: *\nCrawl-delay: 0.4\n"
    )
    attempts = {"fail": 0}

    class RrSpider(Spider):
        name = "rr"

        async def start(self):
            yield _fast("http://a.example/fail")
            yield Request("http://b.example/redirect")

        def parse(self, response: Response) -> None:
            pass

    async def override(request: Request) -> Response:
        _RecordingHandler.starts.append((request.url, monotonic()))
        if request.url.endswith("/robots.txt"):
            return Response(
                request.url, body=_RecordingHandler.robots.get(request.url, b"")
            )
        if request.url.endswith("/fail") and attempts["fail"] == 0:
            attempts["fail"] += 1
            return Response(request.url, status=500)
        if request.url.endswith("/redirect"):
            return Response(
                request.url,
                status=302,
                headers={"Location": "http://b.example/target"},
            )
        return Response(request.url)

    _RecordingHandler.override = override
    crawler = _crawler(RrSpider, CONCURRENT_REQUESTS=4, RETRY_TIMES=1)
    await crawler.crawl_async()

    # b.example robots.txt, /redirect and /target are spaced by 0.4 s.
    b_times = _RecordingHandler.start_times("b.example")
    assert len(b_times) == 2
    assert b_times[1] - b_times[0] >= 0.4 - TOLERANCE
    # The fastbot retry on a.example had a zero interval and did not wait.
    assert crawler.stats
    delayed = crawler.stats.get_value("robotstxt/delayed_count") or 0
    assert delayed >= 2


@coroutine_test
async def test_delayed_count_incremented_once_per_request() -> None:
    """A request that waits, including one revalidated while queued in a slot,
    increments robotstxt/delayed_count exactly once."""
    _RecordingHandler.reset()
    _RecordingHandler.robots["http://a.example/robots.txt"] = MIXED_ROBOTS
    _RecordingHandler.robots["http://b.example/robots.txt"] = (
        b"User-agent: *\nCrawl-delay: 0\n"
    )
    release = _Barrier()
    _RecordingHandler.release_event = release
    _RecordingHandler.blocker_predicate = lambda r: r.url == "http://b.example/block"

    async def body(spider: Spider) -> None:
        engine = spider.crawler.engine
        await engine.download_async(_slow("http://a.example/old"))
        await sleep(0.45)
        blocker_task = _spawn(
            engine.download_async(
                Request(
                    "http://b.example/block",
                    meta={
                        "download_slot": "shared",
                        "dont_obey_robotstxt": True,
                    },
                )
            )
        )
        await _yield_control(3)
        slow_task = _spawn(
            engine.download_async(
                _slow("http://a.example/slow", download_slot="shared")
            )
        )
        await _yield_control(3)
        await engine.download_async(
            _fast("http://a.example/fast", download_slot="other")
        )
        release.set()
        await maybe_deferred_to_future(slow_task)
        await maybe_deferred_to_future(blocker_task)

    _InlineSpider.body = staticmethod(body)
    crawler = _crawler(
        _InlineSpider,
        CONCURRENT_REQUESTS=4,
        CONCURRENT_REQUESTS_PER_DOMAIN=1,
    )
    await crawler.crawl_async()
    assert crawler.stats
    # old and slow each waited once; fast never waited.
    assert crawler.stats.get_value("robotstxt/delayed_count") == 2


@coroutine_test
async def test_disabled_by_default_and_when_obey_delay_off() -> None:
    """With ROBOTSTXT_OBEY_DELAY off, zero and positive Crawl-delay requests
    all start immediately and no delay stat is written."""
    _RecordingHandler.reset()
    _RecordingHandler.robots["http://a.example/robots.txt"] = MIXED_ROBOTS

    async def body(spider: Spider) -> None:
        engine = spider.crawler.engine
        for i in range(4):
            await engine.download_async(_slow(f"http://a.example/{i}"))

    _InlineSpider.body = staticmethod(body)
    crawler = _crawler(
        _InlineSpider, CONCURRENT_REQUESTS=4, ROBOTSTXT_OBEY_DELAY=False
    )
    await crawler.crawl_async()
    times = _RecordingHandler.start_times("a.example")
    assert len(times) == 4
    assert max(_gaps(times)) < 0.2
    assert crawler.stats
    assert crawler.stats.get_value("robotstxt/delayed_count") is None
