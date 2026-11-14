"""
This is a middleware to respect robots.txt policies. To activate it you must
enable this middleware and enable the ROBOTSTXT_OBEY setting.

"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, Self

from twisted.internet.defer import Deferred

from scrapy import signals
from scrapy.core.downloader._robotsdelay import (
    _ROBOTS_DELAY_META_KEY,
    _ROBOTS_START_META_KEY,
)
from scrapy.exceptions import CloseSpider, IgnoreRequest, NotConfigured
from scrapy.http import Request, Response
from scrapy.http.request import NO_CALLBACK
from scrapy.utils.decorators import _warn_spider_arg
from scrapy.utils.defer import maybe_deferred_to_future
from scrapy.utils.httpobj import urlparse_cached
from scrapy.utils.misc import build_from_crawler, load_object

if TYPE_CHECKING:
    from scrapy import Spider
    from scrapy.crawler import Crawler
    from scrapy.robotstxt import RobotParser
    from scrapy.statscollectors import StatsCollector


logger = logging.getLogger(__name__)


class RobotsTxtMiddleware:
    """This middleware filters out requests forbidden by the robots.txt
    exclusion standard.

    To make sure Scrapy respects robots.txt make sure the middleware is enabled
    and the :setting:`ROBOTSTXT_OBEY` setting is enabled.

    The :setting:`ROBOTSTXT_USER_AGENT` setting can be used to specify the
    user agent string to use for matching in the robots.txt_ file. If it
    is ``None``, the User-Agent header you are sending with the request or the
    :setting:`USER_AGENT` setting (in that order) will be used for determining
    the user agent to use in the robots.txt_ file.

    When :setting:`ROBOTSTXT_OBEY_DELAY` is enabled as well, this middleware
    also enforces the ``Crawl-delay`` and ``Request-rate`` directives of each
    robots.txt_ file. Requests are grouped by the robots.txt_ file that governs
    them, independently of download slots. For each group, the start of a
    request, i.e. the moment it is handed to the download handler, is at least
    as far from the start of the previous request of the group as the interval
    that the directives ask for: the larger of the ``Crawl-delay`` value and
    the ``Request-rate`` period divided by its request count. The robots.txt_
    download itself counts as the first start. Requests waiting for their
    interval hold no download slot and do not count toward
    :setting:`CONCURRENT_REQUESTS`.

    This is a floor that other throttling components can only raise. The
    :setting:`DOWNLOAD_DELAY` and :setting:`DOWNLOAD_SLOTS` delays, their
    jitter and :ref:`AutoThrottle <topics-autothrottle>` keep applying on top
    of it, but lowering their values, e.g. through jitter or AutoThrottle
    adjustments, never makes requests start closer together than the robots.txt_
    interval. Intervals are capped at :setting:`ROBOTSTXT_MAX_DELAY`.

    Requests that do not wait for an interval, and do not count toward one,
    are those with :reqmeta:`dont_obey_robotstxt` set, requests with ``data:``
    or ``file:`` URLs, and requests answered from the HTTP cache
    (:class:`~scrapy.downloadermiddlewares.httpcache.HttpCacheMiddleware`).
    Requests dropped by robots.txt_ rules do not count either.

    This middleware has to be combined with a robots.txt_ parser.

    Scrapy ships with support for the following robots.txt_ parsers:

    * :ref:`Protego <protego-parser>` (default)
    * :ref:`RobotFileParser <python-robotfileparser>`
    * :ref:`Robotexclusionrulesparser <rerp-parser>`

    You can change the robots.txt_ parser with the :setting:`ROBOTSTXT_PARSER`
    setting. Or you can also :ref:`implement support for a new parser
    <support-for-new-robots-parser>`.

    If no :ref:`start request <start-requests>` can be crawled, and robots.txt
    rules denied at least one of them, the crawl stops with the
    ``robotstxt_denied`` :stat:`finish_reason`, as long as
    :class:`~scrapy.spidermiddlewares.start.StartSpiderMiddleware` is enabled.
    """

    DOWNLOAD_PRIORITY: int = 1000

    def __init__(self, crawler: Crawler):
        if not crawler.settings.getbool("ROBOTSTXT_OBEY"):
            raise NotConfigured
        self._start_request_crawled = False
        self._start_request_denied = False
        self._default_useragent: str = crawler.settings["USER_AGENT"]
        self._robotstxt_useragent: str | None = crawler.settings["ROBOTSTXT_USER_AGENT"]
        self.crawler: Crawler = crawler
        self._stats: StatsCollector = crawler.stats
        self._parsers: dict[str, RobotParser | Deferred[RobotParser | None] | None] = {}
        self._parserimpl: type[RobotParser] = load_object(
            crawler.settings.get("ROBOTSTXT_PARSER")
        )
        self._obey_delay: bool = crawler.settings.getbool("ROBOTSTXT_OBEY_DELAY")
        self._max_delay: float = self._get_max_delay(crawler)
        # Network locations whose interval was capped at ROBOTSTXT_MAX_DELAY.
        self._delay_capped: set[str] = set()

        # check if parser dependencies are met, this should throw an error otherwise.
        build_from_crawler(self._parserimpl, self.crawler, b"")

        crawler.signals.connect(
            self._response_received, signal=signals.response_received
        )
        crawler.signals.connect(self._spider_idle, signal=signals.spider_idle)

    @staticmethod
    def _get_max_delay(crawler: Crawler) -> float:
        value = crawler.settings["ROBOTSTXT_MAX_DELAY"]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(
                f"ROBOTSTXT_MAX_DELAY must be a non-negative finite number, "
                f"got {value!r}."
            )
        return float(value)

    @classmethod
    def from_crawler(cls, crawler: Crawler) -> Self:
        return cls(crawler)

    def _response_received(self, request: Request) -> None:
        if request.meta.get("is_start_request"):
            self._start_request_crawled = True

    def _spider_idle(self) -> None:
        if self._start_request_crawled or not self._start_request_denied:
            return
        logger.error(
            "Stopping the crawl: no start request could be crawled, and at "
            "least one of them was rejected based on robots.txt rules. See "
            "https://docs.scrapy.org/en/latest/topics/downloader-middleware.html#topics-dlmw-robots"
        )
        raise CloseSpider("robotstxt_denied")

    @_warn_spider_arg
    async def process_request(
        self, request: Request, spider: Spider | None = None
    ) -> None:
        if request.meta.get("dont_obey_robotstxt"):
            return
        if request.url.startswith("data:") or request.url.startswith("file:"):
            return
        netloc = urlparse_cached(request).netloc
        if netloc not in self._parsers and self._is_robotstxt_request(request):
            # Let this request download normally and use its own response as
            # this host's robots.txt, instead of triggering a second,
            # separate download of the same URL.
            if self._obey_delay:
                request.meta[_ROBOTS_START_META_KEY] = netloc
            self._parsers[netloc] = Deferred()
            self._stats.inc_value("robotstxt/request_count")
            return
        rp = await self.robot_parser(request)
        self.process_request_2(rp, request)

    @_warn_spider_arg
    async def process_response(
        self, request: Request, response: Response, spider: Spider | None = None
    ) -> Response:
        netloc = urlparse_cached(request).netloc
        parser = self._parsers.get(netloc)
        if (
            isinstance(parser, Deferred)
            and not parser.called
            and self._is_robotstxt_request(request)
        ):
            await self._parse_robots(response, netloc, request)
        return response

    @_warn_spider_arg
    async def process_exception(
        self, request: Request, exception: Exception, spider: Spider | None = None
    ) -> None:
        netloc = urlparse_cached(request).netloc
        parser = self._parsers.get(netloc)
        if (
            isinstance(parser, Deferred)
            and not parser.called
            and self._is_robotstxt_request(request)
        ):
            self._robots_error(exception, netloc, request)

    @staticmethod
    def _is_robotstxt_request(request: Request) -> bool:
        # robot_parser() handles its own internal robots.txt request directly,
        # without going through process_response()/process_exception().
        if request.meta.get("dont_obey_robotstxt"):
            return False
        url = urlparse_cached(request)
        return request.url == f"{url.scheme}://{url.netloc}/robots.txt"

    def process_request_2(self, rp: RobotParser | None, request: Request) -> None:
        useragent: str | bytes | None = self._robotstxt_useragent
        if not useragent:
            useragent = request.headers.get(b"User-Agent", self._default_useragent)
            assert useragent is not None
        if rp is not None and not rp.allowed(request.url, useragent):
            logger.debug(
                "Forbidden by robots.txt: %(request)s",
                {"request": request},
                extra={"spider": self.crawler.spider},
            )
            self._stats.inc_value("robotstxt/forbidden")
            if request.meta.get("is_start_request"):
                self._start_request_denied = True
            raise IgnoreRequest("Forbidden by robots.txt")
        if self._obey_delay:
            netloc = urlparse_cached(request).netloc
            min_delay = self._min_delay(rp, useragent, netloc)
            # Derived requests, e.g. redirects, inherit this key: recompute it
            # for the URL of this request.
            request.meta.pop(_ROBOTS_DELAY_META_KEY, None)
            if min_delay > 0:
                request.meta[_ROBOTS_DELAY_META_KEY] = (netloc, min_delay)

    def _min_delay(
        self, rp: RobotParser | None, useragent: str | bytes, netloc: str
    ) -> float:
        """Return the minimum interval, in seconds, that robots.txt asks for
        between the starts of the requests governed by the same file."""
        if rp is None:
            return 0.0
        delay = max(
            self._crawl_delay(rp, useragent), self._request_rate_delay(rp, useragent)
        )
        if delay > self._max_delay:
            if netloc not in self._delay_capped:
                self._delay_capped.add(netloc)
                self._stats.inc_value("robotstxt/delay_capped")
                logger.warning(
                    f"The robots.txt delay for {netloc} ({delay:g} s) exceeds "
                    f"ROBOTSTXT_MAX_DELAY ({self._max_delay:g} s), capping it. "
                    f"See "
                    f"https://docs.scrapy.org/en/latest/topics/settings.html#robotstxt-max-delay",
                    extra={"spider": self.crawler.spider},
                )
            return self._max_delay
        return delay

    @staticmethod
    def _crawl_delay(rp: RobotParser, useragent: str | bytes) -> float:
        delay = rp.crawl_delay(useragent)
        if delay is None or not math.isfinite(delay) or delay < 0:
            return 0.0
        return float(delay)

    @staticmethod
    def _request_rate_delay(rp: RobotParser, useragent: str | bytes) -> float:
        rate = rp.request_rate(useragent)
        if rate is None:
            return 0.0
        requests, seconds = rate
        if (
            requests <= 0
            or seconds <= 0
            or not math.isfinite(requests)
            or not math.isfinite(seconds)
        ):
            return 0.0
        return float(seconds) / float(requests)

    async def robot_parser(self, request: Request) -> RobotParser | None:
        url = urlparse_cached(request)
        netloc = url.netloc

        if netloc not in self._parsers:
            self._parsers[netloc] = Deferred()
            robotsurl = f"{url.scheme}://{url.netloc}/robots.txt"
            meta: dict[str, Any] = {
                "dont_obey_robotstxt": True,
                "is_robotstxt_request": True,
            }
            if "download_handler" in request.meta:
                meta["download_handler"] = request.meta["download_handler"]
            if self._obey_delay:
                # The robots.txt download itself counts as a start for the
                # interval, even though it is not throttled.
                meta[_ROBOTS_START_META_KEY] = netloc
            robotsreq = Request(
                robotsurl,
                priority=self.DOWNLOAD_PRIORITY,
                meta=meta,
                callback=NO_CALLBACK,
            )
            try:
                resp = await self.crawler.engine.download_async(robotsreq)
                await self._parse_robots(resp, netloc, request)
            except Exception as e:
                self._robots_error(e, netloc, request)
            self._stats.inc_value("robotstxt/request_count")

        parser = self._parsers[netloc]
        if isinstance(parser, Deferred):
            # Every caller needs to get its own Deferred, so that awaiting or
            # cancelling one doesn't affect others.
            waiter: Deferred[RobotParser | None] = Deferred()

            def cb(result: RobotParser | None) -> RobotParser | None:
                waiter.callback(result)
                return result

            parser.addCallback(cb)
            return await maybe_deferred_to_future(waiter)
        return parser

    async def _parse_robots(
        self, response: Response, netloc: str, request: Request
    ) -> None:
        self._stats.inc_value("robotstxt/response_count")
        self._stats.inc_value(f"robotstxt/response_status_count/{response.status}")
        rp = build_from_crawler(self._parserimpl, self.crawler, response.body)
        await self.crawler.signals.send_catch_log_async(
            signal=signals.robots_parsed,
            robotparser=rp,
            request=request,
            response=response,
        )
        rp_dfd = self._parsers[netloc]
        assert isinstance(rp_dfd, Deferred)
        self._parsers[netloc] = rp
        rp_dfd.callback(rp)

    def _robots_error(self, exc: Exception, netloc: str, request: Request) -> None:
        if not isinstance(exc, IgnoreRequest):
            logger.error(
                f"Error downloading {request}",
                exc_info=exc,
                extra={"spider": self.crawler.spider},
            )
            key = f"robotstxt/exception_count/{type(exc)}"
            self._stats.inc_value(key)
        rp_dfd = self._parsers[netloc]
        assert isinstance(rp_dfd, Deferred)
        self._parsers[netloc] = None
        rp_dfd.callback(None)
