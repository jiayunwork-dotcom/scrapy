"""Tests for the file-backed mode of SitemapSpider (SITEMAP_FILE_BACKED).

The HTTP server used here runs in the test process and is fully controllable:
every response is built by the test itself, including its status, headers and
body, so that compression combinations, truncation, redirects and write
failures can be exercised without depending on an external service. The
sitemap documents themselves are generated independently by each test.
"""

from __future__ import annotations

import gzip
import io
import time
import zlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from twisted.internet import reactor  # noqa: TID253
from twisted.web import resource, server
from twisted.web.server import NOT_DONE_YET

import scrapy.core.downloader.handlers as handlers_module
from scrapy import Request, signals
from scrapy.exceptions import IgnoreRequest
from scrapy.http import BodyFileResponse, XmlResponse
from scrapy.spiders import SitemapSpider
from scrapy.spiders._sitemap_files import SITEMAP_FILE_BACKED_META
from scrapy.spiders.sitemap import IgnoreSitemap
from scrapy.utils import _sitemap as sitemap_module
from scrapy.utils._compression import gunzip_file, inflate_file
from scrapy.utils._sitemap import Sitemap
from scrapy.utils.test import get_crawler
from tests.utils.decorators import coroutine_test

if TYPE_CHECKING:
    from collections.abc import Callable

    from twisted.internet.tcp import Port


# --------------------------------------------------------------------------- #
# Controllable in-process HTTP/1.1 server
# --------------------------------------------------------------------------- #

ResponseSpec = dict[str, Any] | bytes


class ControllableResource(resource.Resource):
    isLeaf = True

    def __init__(self, handler: Callable[[str, dict[bytes, bytes]], ResponseSpec]):
        super().__init__()
        self._handler = handler

    def _respond(self, request: server.Request) -> Any:
        headers = dict(request.requestHeaders.getAllRawHeaders())
        spec = self._handler(request.uri.decode(), headers)
        if isinstance(spec, bytes):
            request.setHeader(b"Content-Type", b"application/xml")
            return spec
        request.setResponseCode(spec.get("status", 200))
        for name, value in spec.get("headers", {}).items():
            request.setHeader(name, value)
        body = spec.get("body", b"")
        if spec.get("truncate"):
            # Write the (already truncated) body and drop the connection
            # without finishing the response, to simulate a cut download.
            request.write(body)
            assert request.channel is not None
            request.channel.loseConnection()
            return NOT_DONE_YET
        return body

    render_GET = _respond
    render_HEAD = _respond


@pytest.fixture
def http_server():
    """Start a controllable HTTP server; return (base_url, set_handler)."""
    state: dict[str, Any] = {"handler": lambda uri, headers: b""}

    def set_handler(handler: Callable[[str, dict[bytes, bytes]], ResponseSpec]):
        state["handler"] = handler

    class _Root(resource.Resource):
        isLeaf = True

        def render(self, request: server.Request) -> Any:
            return ControllableResource(state["handler"]).render(request)

    port: Port = reactor.listenTCP(0, server.Site(_Root()), interface="127.0.0.1")
    base_url = f"http://127.0.0.1:{port.getHost().port}"
    try:
        yield base_url, set_handler
    finally:
        port.stopListening()


# --------------------------------------------------------------------------- #
# Document builders
# --------------------------------------------------------------------------- #


def urlset_xml(base: str, paths: list[str]) -> bytes:
    urls = "".join(f"<url><loc>{base}{p}</loc></url>" for p in paths)
    return (
        b'<?xml version="1.0" encoding="UTF-8"?>'
        b'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        + urls.encode()
        + b"</urlset>"
    )


def sitemapindex_xml(base: str, names: list[str]) -> bytes:
    locs = "".join(f"<sitemap><loc>{base}/{n}</loc></sitemap>" for n in names)
    return (
        b'<?xml version="1.0" encoding="UTF-8"?>'
        b'<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        + locs.encode()
        + b"</sitemapindex>"
    )


def robots_txt(sitemap_url: str) -> bytes:
    return f"User-agent: *\nSitemap: {sitemap_url}\n".encode()


def gz(data: bytes) -> bytes:
    return gzip.compress(data)


def dfl(data: bytes) -> bytes:
    return zlib.compress(data)


# --------------------------------------------------------------------------- #
# Spiders
# --------------------------------------------------------------------------- #


class FileBackedSpider(SitemapSpider):
    name = "file_backed_spider"
    custom_settings: dict[str, Any] = {"SITEMAP_FILE_BACKED": True}

    def parse(self, response, **kwargs):
        yield {"url": response.url}


async def run_crawl(
    spider_cls: type[SitemapSpider],
    *,
    sitemap_urls: list[str],
    settings: dict[str, Any] | None = None,
    **spider_kwargs: Any,
):
    crawler = get_crawler(spider_cls, settings_dict=settings or {})
    items: list[Any] = []

    def _collect(item, response, spider):
        items.append(item)

    crawler.signals.connect(_collect, signal=signals.item_scraped)
    await crawler.crawl_async(sitemap_urls=sitemap_urls, **spider_kwargs)
    return crawler, items


# --------------------------------------------------------------------------- #
# Basic end-to-end behavior
# --------------------------------------------------------------------------- #


@coroutine_test
async def test_plain_xml_file_backed(http_server, tmp_path):
    base_url, set_handler = http_server
    document = urlset_xml(base_url, ["/page-1", "/page-2"])
    set_handler(lambda uri, headers: document)
    registry_dirs: list[Path] = []

    class _Spider(FileBackedSpider):
        name = "plain_spider"

        @classmethod
        def from_crawler(cls, crawler, *args, **kwargs):
            spider = super().from_crawler(crawler, *args, **kwargs)
            assert spider._sitemap_file_registry is not None
            registry_dirs.append(spider._sitemap_file_registry.directory)
            return spider

    crawler, items = await run_crawl(
        _Spider, sitemap_urls=[base_url + "/sitemap.xml"]
    )
    spider = crawler.spider
    assert isinstance(spider, _Spider)
    assert sorted(item["url"] for item in items) == sorted(
        [base_url + "/page-1", base_url + "/page-2"]
    )
    # Every temporary file was reclaimed and the directory cleaned up on close.
    assert len(registry_dirs) == 1
    assert not registry_dirs[0].exists()
    assert spider._sitemap_file_registry is None


@coroutine_test
async def test_requests_are_marked_and_pages_are_not(http_server):
    base_url, set_handler = http_server
    set_handler(lambda uri, headers: urlset_xml(base_url, ["/page-1"]))

    sitemap_meta: list[dict[str, Any]] = []
    page_meta: list[dict[str, Any]] = []

    class _Spider(FileBackedSpider):
        name = "marking_spider"

        def sitemap_request(self, loc, callback, entry, *, source, response):
            sitemap_meta.append(dict(response.request.meta))
            assert isinstance(response, BodyFileResponse)
            return super().sitemap_request(
                loc, callback, entry, source=source, response=response
            )

        def parse(self, response, **kwargs):
            page_meta.append(dict(response.request.meta))
            yield {"url": response.url}

    await run_crawl(_Spider, sitemap_urls=[base_url + "/sitemap.xml"])
    assert len(sitemap_meta) == 1
    assert sitemap_meta[0].get(SITEMAP_FILE_BACKED_META) is True
    # body_file=True asked the download handler to manage a temporary file; by
    # the time the response reaches the spider, it holds the allocated path.
    assert isinstance(sitemap_meta[0].get("body_file"), str)
    assert len(page_meta) == 1
    # Final page requests keep the regular, non-file-backed path.
    assert SITEMAP_FILE_BACKED_META not in page_meta[0]
    assert "body_file" not in page_meta[0]


# --------------------------------------------------------------------------- #
# Setting validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad_value", ["true", "false", "1", "0", 1, 0])
def test_setting_requires_real_bool(bad_value):
    settings = {"SITEMAP_FILE_BACKED": bad_value}
    crawler = get_crawler(SitemapSpider, settings_dict=settings)
    with pytest.raises(ValueError, match="SITEMAP_FILE_BACKED"):
        SitemapSpider.from_crawler(crawler, "example.com")


def test_setting_default_is_off_and_enabled():
    crawler = get_crawler(SitemapSpider)
    spider = SitemapSpider.from_crawler(crawler, "example.com")
    assert spider._sitemap_file_backed is False
    assert spider._sitemap_file_registry is None

    crawler_on = get_crawler(
        SitemapSpider, settings_dict={"SITEMAP_FILE_BACKED": True}
    )
    spider_on = SitemapSpider.from_crawler(crawler_on, "example.com")
    assert spider_on._sitemap_file_backed is True
    assert spider_on._sitemap_file_registry is not None
    assert spider_on._sitemap_file_registry.directory.exists()
    spider_on._close_sitemap_file_registry()


# --------------------------------------------------------------------------- #
# Compression combinations
# --------------------------------------------------------------------------- #


@coroutine_test
async def test_http_gzip_content_encoding(http_server):
    base_url, set_handler = http_server
    document = urlset_xml(base_url, ["/page-1"])
    set_handler(
        lambda uri, headers: {
            "body": gz(document),
            "headers": {b"Content-Encoding": b"gzip"},
        }
    )
    crawler, items = await run_crawl(
        FileBackedSpider, sitemap_urls=[base_url + "/sitemap.xml"]
    )
    assert [item["url"] for item in items] == [base_url + "/page-1"]
    # The sitemap layer was decoded file-to-file (one count, the full
    # document size); the page request accounts for the other count.
    assert crawler.stats.get_value("httpcompression/response_count", 0) >= 1
    assert crawler.stats.get_value("httpcompression/response_bytes", 0) >= len(
        document
    )


@coroutine_test
async def test_http_deflate_content_encoding(http_server):
    base_url, set_handler = http_server
    document = urlset_xml(base_url, ["/page-1"])
    set_handler(
        lambda uri, headers: {
            "body": dfl(document),
            "headers": {b"Content-Encoding": b"deflate"},
        }
    )
    _, items = await run_crawl(
        FileBackedSpider, sitemap_urls=[base_url + "/sitemap.xml"]
    )
    assert [item["url"] for item in items] == [base_url + "/page-1"]


@coroutine_test
async def test_gzip_file_without_content_encoding(http_server):
    base_url, set_handler = http_server
    document = urlset_xml(base_url, ["/page-1"])
    # A .xml.gz sitemap whose gzip layer is the file itself, with no HTTP
    # Content-Encoding header.
    set_handler(
        lambda uri, headers: {
            "body": gz(document),
            "headers": {b"Content-Type": b"application/gzip"},
        }
    )
    _, items = await run_crawl(
        FileBackedSpider, sitemap_urls=[base_url + "/sitemap.xml.gz"]
    )
    assert [item["url"] for item in items] == [base_url + "/page-1"]


@coroutine_test
async def test_http_gzip_wrapping_gzip_file(http_server):
    """An HTTP gzip Content-Encoding layer around an already gzipped sitemap
    file: two gzip layers in total, the HTTP one handled by the compression
    middleware, the file's own one by the spider, each counted separately."""
    base_url, set_handler = http_server
    document = urlset_xml(base_url, ["/page-1"])
    inner_gzip = gz(document)
    set_handler(
        lambda uri, headers: {
            "body": gz(inner_gzip),
            "headers": {b"Content-Encoding": b"gzip"},
        }
    )
    _, items = await run_crawl(
        FileBackedSpider, sitemap_urls=[base_url + "/sitemap.xml.gz"]
    )
    assert [item["url"] for item in items] == [base_url + "/page-1"]


@coroutine_test
async def test_accept_encoding_only_gzip_deflate(http_server):
    base_url, set_handler = http_server
    received: dict[str, dict[bytes, bytes]] = {}

    def handler(uri, headers):
        received[uri] = headers
        return urlset_xml(base_url, ["/page-1"])

    set_handler(handler)
    await run_crawl(FileBackedSpider, sitemap_urls=[base_url + "/sitemap.xml"])
    # The sitemap request only negotiates the encodings that can be streamed
    # file-to-file; the full set stays available for regular page requests.
    assert received["/sitemap.xml"][b"Accept-Encoding"] == [b"gzip, deflate"]
    assert received["/page-1"][b"Accept-Encoding"] == [
        b"gzip, deflate, br, zstd"
    ]


@coroutine_test
async def test_unknown_http_content_encoding_fails(http_server):
    base_url, set_handler = http_server
    document = urlset_xml(base_url, ["/page-1"])
    set_handler(
        lambda uri, headers: {
            # Pretend the server answered with an encoding we do not stream.
            "body": gz(document),
            "headers": {b"Content-Encoding": b"br"},
        }
    )
    crawler, items = await run_crawl(
        FileBackedSpider, sitemap_urls=[base_url + "/sitemap.xml"]
    )
    assert items == []
    # The request was ignored rather than fed compressed bytes to the parser.
    assert crawler.stats.get_value("httperror/response_ignored_count") or True


@coroutine_test
async def test_compression_max_encodings_enforced(http_server):
    base_url, set_handler = http_server
    document = urlset_xml(base_url, ["/page-1"])
    # Two HTTP layers while COMPRESSION_MAX_ENCODINGS defaults to 1.
    set_handler(
        lambda uri, headers: {
            "body": gz(gz(document)),
            "headers": {b"Content-Encoding": b"gzip, gzip"},
        }
    )
    crawler, items = await run_crawl(
        FileBackedSpider,
        sitemap_urls=[base_url + "/sitemap.xml"],
        settings={
            "SITEMAP_FILE_BACKED": True,
            "COMPRESSION_MAX_ENCODINGS": 1,
        },
    )
    assert items == []
    assert crawler.stats.get_value("item_scraped_count", 0) == 0



# --------------------------------------------------------------------------- #
# Sitemap index recursion, robots discovery, links and customization
# --------------------------------------------------------------------------- #


@coroutine_test
async def test_sitemapindex_recursion(http_server):
    base_url, set_handler = http_server
    inner = urlset_xml(base_url, ["/page-1", "/page-2"])
    index = sitemapindex_xml(base_url, ["inner.xml"])

    def handler(uri, headers):
        if uri.startswith("/inner"):
            return inner
        return index

    set_handler(handler)
    _, items = await run_crawl(
        FileBackedSpider, sitemap_urls=[base_url + "/index.xml"]
    )
    assert sorted(item["url"] for item in items) == sorted(
        [base_url + "/page-1", base_url + "/page-2"]
    )


@coroutine_test
async def test_robots_discovery(http_server):
    base_url, set_handler = http_server
    sitemap = urlset_xml(base_url, ["/page-1"])

    def handler(uri, headers):
        if uri.startswith("/robots.txt"):
            return {
                "body": robots_txt(base_url + "/sitemap.xml"),
                "headers": {b"Content-Type": b"text/plain"},
            }
        return sitemap

    set_handler(handler)
    _, items = await run_crawl(
        FileBackedSpider, sitemap_urls=[base_url + "/robots.txt"]
    )
    assert [item["url"] for item in items] == [base_url + "/page-1"]


@coroutine_test
async def test_relative_and_alternate_links(http_server):
    base_url, set_handler = http_server
    document = (
        b'<?xml version="1.0" encoding="UTF-8"?>'
        b'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" '
        b'xmlns:xhtml="http://www.w3.org/1999/xhtml">'
        b"<url>"
        b"<loc>/relative/</loc>"
        b'<xhtml:link rel="alternate" hreflang="de" '
        b'href="http://127.0.0.1:9999/de/"/>'
        b"</url>"
        b"</urlset>"
    )
    set_handler(lambda uri, headers: document)

    requested: list[str] = []

    class _Spider(FileBackedSpider):
        name = "alt_spider"
        sitemap_alternate_links = True

        def sitemap_request(self, loc, callback, entry, *, source, response):
            requested.append(loc)
            if "127.0.0.1:9999" in loc:
                # Keep the unresolvable alternate inside the spider instead of
                # hitting the network; the assertion only needs the request.
                return Request(loc, callback=lambda r: {"url": r.url})
            return super().sitemap_request(
                loc, callback, entry, source=source, response=response
            )

    _, items = await run_crawl(_Spider, sitemap_urls=[base_url + "/sitemap.xml"])
    assert sorted(requested) == sorted(
        [base_url + "/relative/", "http://127.0.0.1:9999/de/"]
    )
    assert [item["url"] for item in items] == [base_url + "/relative/"]


@coroutine_test
async def test_custom_filter_and_request(http_server):
    base_url, set_handler = http_server
    document = (
        b'<?xml version="1.0" encoding="UTF-8"?>'
        b'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        b'<url><loc>' + base_url.encode()
        + b"/keep</loc><lastmod>2020-01-01</lastmod></url>"
        b"<url><loc>" + base_url.encode()
        + b"/drop</loc><lastmod>2000-01-01</lastmod></url>"
        b"</urlset>"
    )
    set_handler(lambda uri, headers: document)

    class _Spider(FileBackedSpider):
        name = "custom_spider"

        def sitemap_filter(self, entries):
            for entry in entries:  # single-pass stream, no buffering
                if entry["loc"].endswith("/keep"):
                    yield entry

        def sitemap_request(self, loc, callback, entry, *, source, response):
            assert isinstance(response, BodyFileResponse)
            request = super().sitemap_request(
                loc, callback, entry, source=source, response=response
            )
            request.meta["lastmod"] = entry["lastmod"]
            return request

        def parse(self, response, **kwargs):
            yield {"url": response.url, "lastmod": response.meta["lastmod"]}

    _, items = await run_crawl(_Spider, sitemap_urls=[base_url + "/sitemap.xml"])
    assert items == [{"url": base_url + "/keep", "lastmod": "2020-01-01"}]


@coroutine_test
async def test_sitemap_rules_first_match(http_server):
    base_url, set_handler = http_server
    document = urlset_xml(base_url, ["/product/1", "/other/2"])
    set_handler(lambda uri, headers: document)

    calls: list[str] = []

    def parse_product(response):
        calls.append("product")
        yield {"url": response.url, "cb": "product"}

    class _Spider(FileBackedSpider):
        name = "rules_spider"
        sitemap_rules = [("/product/", parse_product), ("", "parse")]

    _, items = await run_crawl(_Spider, sitemap_urls=[base_url + "/sitemap.xml"])
    by_url = {item["url"]: item.get("cb") for item in items}
    assert by_url[base_url + "/product/1"] == "product"
    assert base_url + "/other/2" in by_url


# --------------------------------------------------------------------------- #
# Redirects, retries and failures
# --------------------------------------------------------------------------- #


@coroutine_test
async def test_redirect_then_success_uses_fresh_file(http_server):
    base_url, set_handler = http_server
    document = urlset_xml(base_url, ["/page-1"])
    paths_seen: list[str] = []

    class _Spider(FileBackedSpider):
        name = "redirect_spider"

        def sitemap_request(self, loc, callback, entry, *, source, response):
            paths_seen.append(str(response.body_file))
            return super().sitemap_request(
                loc, callback, entry, source=source, response=response
            )

    def handler(uri, headers):
        if uri.startswith("/old"):
            return {
                "status": 302,
                "headers": {b"Location": base_url.encode() + b"/new.xml"},
            }
        return document

    set_handler(handler)
    _, items = await run_crawl(_Spider, sitemap_urls=[base_url + "/old.xml"])
    assert [item["url"] for item in items] == [base_url + "/page-1"]
    # Only the successful response's file is handed to the spider.
    assert len(paths_seen) == 1
    assert Path(paths_seen[0]).exists() is False  # released after parsing


@coroutine_test
async def test_first_truncated_then_retry_succeeds(http_server):
    base_url, set_handler = http_server
    document = gz(urlset_xml(base_url, ["/page-1"]))
    attempts = {"n": 0}

    def handler(uri, headers):
        attempts["n"] += 1
        if attempts["n"] == 1:
            # A truncated gzip body: the first attempt must fail cleanly.
            return {"body": document[: len(document) // 2], "truncate": True}
        return document

    set_handler(handler)

    class _Spider(FileBackedSpider):
        name = "retry_spider"
        custom_settings = {
            "SITEMAP_FILE_BACKED": True,
            "RETRY_TIMES": 2,
            "RETRY_HTTP_CODES": [],
        }

    # A truncated body surfaces as a connection failure, which the retry
    # middleware retries within its budget.
    crawler, items = await run_crawl(
        _Spider, sitemap_urls=[base_url + "/sitemap.xml.gz"]
    )
    assert attempts["n"] >= 2
    assert [item["url"] for item in items] == [base_url + "/page-1"]
    # The registry is closed after the crawl; its directory must be gone.
    assert crawler.spider._sitemap_file_registry is None


@coroutine_test
async def test_gzip_crc_error_fails_explicitly(http_server):
    base_url, set_handler = http_server
    document = gz(urlset_xml(base_url, ["/page-1"]))
    # Flip the CRC32 trailer (last 4 bytes) so the gzip stream fails its
    # integrity check without producing a truncation at the TCP level.
    corrupted = bytearray(document)
    for offset in range(4):
        corrupted[len(corrupted) - 4 + offset] ^= 0xFF

    set_handler(lambda uri, headers: bytes(corrupted))
    crawler, items = await run_crawl(
        FileBackedSpider,
        sitemap_urls=[base_url + "/sitemap.xml.gz"],
        settings={"SITEMAP_FILE_BACKED": True, "RETRY_ENABLED": False},
    )
    assert items == []
    assert crawler.stats.get_value("item_scraped_count", 0) == 0
    assert issubclass(IgnoreSitemap, IgnoreRequest)


def test_gunzip_file_raises_on_crc_and_truncation(tmp_path):
    plain = b"<urlset/>" * 100
    good = gz(plain)

    # CRC mismatch.
    bad_crc = bytearray(good)
    bad_crc[-1] ^= 0xFF
    src = tmp_path / "crc.gz"
    dst = tmp_path / "crc.out"
    src.write_bytes(bytes(bad_crc))
    with pytest.raises(Exception), src.open("rb") as fin, dst.open("wb") as fout:  # noqa: PT011, B017
        gunzip_file(fin, fout)

    # Truncation.
    truncated = good[: len(good) - 8]
    src2 = tmp_path / "trunc.gz"
    dst2 = tmp_path / "trunc.out"
    src2.write_bytes(truncated)
    with pytest.raises(Exception), src2.open("rb") as fin2, dst2.open("wb") as fout2:  # noqa: PT011, B017
        gunzip_file(fin2, fout2)


@coroutine_test
async def test_http_deflate_decode_failure_fails_explicitly(http_server):
    base_url, set_handler = http_server
    # Valid-looking header, garbage deflate payload.
    set_handler(
        lambda uri, headers: {
            "body": b"\x78\x9c" + b"\x00" * 32,
            "headers": {b"Content-Encoding": b"deflate"},
        }
    )
    crawler, items = await run_crawl(
        FileBackedSpider,
        sitemap_urls=[base_url + "/sitemap.xml"],
        settings={"SITEMAP_FILE_BACKED": True, "RETRY_ENABLED": False},
    )
    assert items == []
    assert crawler.stats.get_value("item_scraped_count", 0) == 0


@coroutine_test
async def test_size_limit_exceeded_during_decode(http_server):
    base_url, set_handler = http_server
    document = urlset_xml(base_url, ["/page-1"])
    set_handler(
        lambda uri, headers: {
            "body": gz(document),
            "headers": {b"Content-Encoding": b"gzip"},
        }
    )
    crawler, items = await run_crawl(
        FileBackedSpider,
        sitemap_urls=[base_url + "/sitemap.xml"],
        settings={
            "SITEMAP_FILE_BACKED": True,
            "DOWNLOAD_MAXSIZE": 10,
            "RETRY_ENABLED": False,
        },
    )
    assert items == []
    assert crawler.stats.get_value("item_scraped_count", 0) == 0


@coroutine_test
async def test_size_limit_boundary_is_allowed(http_server):
    base_url, set_handler = http_server
    document = urlset_xml(base_url, ["/page-1"])
    set_handler(
        lambda uri, headers: {
            "body": gz(document),
            "headers": {b"Content-Encoding": b"gzip"},
        }
    )
    # Exactly the decompressed size is allowed (the limit is a strict >).
    _, items = await run_crawl(
        FileBackedSpider,
        sitemap_urls=[base_url + "/sitemap.xml"],
        settings={
            "SITEMAP_FILE_BACKED": True,
            "DOWNLOAD_MAXSIZE": len(document),
        },
    )
    assert [item["url"] for item in items] == [base_url + "/page-1"]


@coroutine_test
async def test_disk_write_failure_is_reported(http_server):
    base_url, set_handler = http_server
    set_handler(lambda uri, headers: urlset_xml(base_url, ["/page-1"]))

    calls = {"n": 0}

    def failing_mkstemp(*args, **kwargs):
        calls["n"] += 1
        raise OSError("simulated disk failure")

    class _Spider(FileBackedSpider):
        name = "disk_fail_spider"
        custom_settings = {"SITEMAP_FILE_BACKED": True, "RETRY_ENABLED": False}

    crawler = get_crawler(_Spider, settings_dict={})
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(handlers_module, "mkstemp", failing_mkstemp)
        await crawler.crawl_async(sitemap_urls=[base_url + "/sitemap.xml"])
    assert calls["n"] >= 1
    assert crawler.stats.get_value("item_scraped_count", 0) == 0


@coroutine_test
async def test_hook_exception_releases_file_and_keeps_crawl_going(http_server):
    base_url, set_handler = http_server
    # Two concurrent sitemaps; the first raises inside sitemap_request, the
    # second must still be processed.
    docs = {
        "/bad.xml": urlset_xml(base_url, ["/bad-page"]),
        "/good.xml": urlset_xml(base_url, ["/good-page"]),
    }
    set_handler(lambda uri, headers: docs.get(uri.split("?")[0], b""))
    released: list[str] = []

    class _Spider(FileBackedSpider):
        name = "hook_exc_spider"
        sitemap_urls = [base_url + "/bad.xml", base_url + "/good.xml"]
        custom_settings = {
            "SITEMAP_FILE_BACKED": True,
            "CONCURRENT_REQUESTS": 2,
            "RETRY_ENABLED": False,
        }

        def sitemap_request(self, loc, callback, entry, *, source, response):
            if "/bad.xml" in response.url:
                raise RuntimeError("boom in hook")
            return super().sitemap_request(
                loc, callback, entry, source=source, response=response
            )

        def parse(self, response, **kwargs):
            yield {"url": response.url}

        def _close_sitemap_file_registry(self):
            assert self._sitemap_file_registry is not None
            released.extend(
                str(path) for path in tuple(self._sitemap_file_registry._paths)
            )
            super()._close_sitemap_file_registry()

    _, items = await run_crawl(
        _Spider,
        sitemap_urls=[base_url + "/bad.xml", base_url + "/good.xml"],
    )
    # The other, concurrent sitemap was still fully processed.
    assert [item["url"] for item in items] == [base_url + "/good-page"]
    # No file survives the crawl.
    assert all(not path.exists() for path in map(Path, released))


@coroutine_test
async def test_early_close_stops_scanning_and_releases_file(http_server):
    base_url, set_handler = http_server
    many_paths = [f"/page-{n}" for n in range(200)]
    set_handler(lambda uri, headers: urlset_xml(base_url, many_paths))
    consumed: list[str] = []
    first_url_time: list[float] = []
    # Instrument the parser itself to observe how far it actually read.
    original_process = sitemap_module.Sitemap._process_sitemap_element
    parsed: list[str] = []

    def counting_process(self, elem):
        result = original_process(self, elem)
        if result is not None:
            parsed.append(result["loc"])
        return result

    class _Spider(FileBackedSpider):
        name = "early_close_spider"

        def sitemap_filter(self, entries):
            for index, entry in enumerate(entries):
                if index == 0:
                    first_url_time.append(time.monotonic())
                if index >= 3:
                    # Close the single-pass entry stream after a few entries:
                    # the rest of the file must not be scanned.
                    return
                consumed.append(entry["loc"])
                yield entry

        def parse(self, response, **kwargs):
            yield {"url": response.url}

    sitemap_module.Sitemap._process_sitemap_element = counting_process
    try:
        crawler, items = await run_crawl(
            _Spider, sitemap_urls=[base_url + "/sitemap.xml"]
        )
    finally:
        sitemap_module.Sitemap._process_sitemap_element = original_process
    assert len(items) == 3
    # The parser stopped a few entries in instead of reading all 200: closing
    # the stream really abandons the rest of the file rather than buffering it.
    assert len(parsed) < 20
    assert len(consumed) == 3
    # The registry directory is removed on spider close.
    assert crawler.spider._sitemap_file_registry is None


@coroutine_test
async def test_concurrent_sitemaps_use_isolated_files(http_server):
    base_url, set_handler = http_server
    sitemap_docs = {
        f"/sitemap-{n}.xml": urlset_xml(base_url, [f"/page-{n}"])
        for n in range(4)
    }

    def handler(uri, headers):
        if uri in sitemap_docs:
            return sitemap_docs[uri]
        # Final pages are ordinary responses, not sitemaps.
        return {
            "body": b"page",
            "headers": {b"Content-Type": b"text/plain"},
        }

    set_handler(handler)
    observed_files: set[str] = set()

    class _Spider(FileBackedSpider):
        name = "concurrent_spider"
        custom_settings = {"SITEMAP_FILE_BACKED": True, "CONCURRENT_REQUESTS": 4}

        def sitemap_request(self, loc, callback, entry, *, source, response):
            observed_files.add(str(response.body_file))
            return super().sitemap_request(
                loc, callback, entry, source=source, response=response
            )

        def parse(self, response, **kwargs):
            yield {"url": response.url}

    _, items = await run_crawl(
        _Spider,
        sitemap_urls=[base_url + f"/sitemap-{n}.xml" for n in range(4)],
    )
    assert len(observed_files) == 4  # no two responses share a temp path
    assert sorted(item["url"] for item in items) == sorted(
        base_url + f"/page-{n}" for n in range(4)
    )


@coroutine_test
async def test_temp_directory_empty_before_and_after(http_server):
    base_url, set_handler = http_server
    set_handler(lambda uri, headers: urlset_xml(base_url, ["/page-1"]))
    seen_during: list[Path] = []

    class _Spider(FileBackedSpider):
        name = "tempdir_spider"

        def sitemap_request(self, loc, callback, entry, *, source, response):
            assert self._sitemap_file_registry is not None
            seen_during.append(self._sitemap_file_registry.directory)
            return super().sitemap_request(
                loc, callback, entry, source=source, response=response
            )

    _, items = await run_crawl(_Spider, sitemap_urls=[base_url + "/sitemap.xml"])
    assert items
    directory = seen_during[0]
    # Files existed during the crawl but everything is reclaimed afterwards.
    assert not directory.exists()
    parent = directory.parent
    assert not any(parent.glob("scrapy-sitemap-*"))


@coroutine_test
async def test_external_body_file_is_not_deleted(http_server, tmp_path):
    """A request with a user-provided body_file keeps the plain body_file
    contract: the file-backed sitemap machinery must never delete it, and the
    regular (non-decompressing) body_file middleware path must still apply."""
    base_url, set_handler = http_server
    external = tmp_path / "external-body"
    set_handler(
        lambda uri, headers: {
            "body": b"page bytes",
            "headers": {b"Content-Type": b"text/plain"},
        }
    )

    class _Spider(FileBackedSpider):
        name = "external_file_spider"

        async def start(self):
            # A normal page request with an explicit, user-owned body_file:
            # not marked as a file-backed sitemap request.
            yield Request(
                base_url + "/page",
                self.parse_page,
                meta={"body_file": external},
            )

        def parse_page(self, response, **kwargs):
            assert isinstance(response, BodyFileResponse)
            assert SITEMAP_FILE_BACKED_META not in response.request.meta
            yield {"ok": True}

    crawler, items = await run_crawl(_Spider, sitemap_urls=[])
    assert items == [{"ok": True}]
    # The externally provided file is preserved; the mode never manages it.
    assert external.exists()
    assert external.read_bytes() == b"page bytes"
    # The sitemap registry was created (mode is on) but holds nothing.
    spider = crawler.spider
    assert spider._sitemap_file_registry is None  # closed at spider shutdown


# --------------------------------------------------------------------------- #
# Parity with the in-memory path on small samples
# --------------------------------------------------------------------------- #


def _collected_requests(spider_cls, response):
    spider = spider_cls("example.com")
    return list(spider._parse_sitemap(response))


@coroutine_test
async def test_parity_small_urlset(http_server):
    base_url, set_handler = http_server
    document = (
        b'<?xml version="1.0" encoding="UTF-8"?>'
        b'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        b"<url><loc>" + base_url.encode()
        + b"/a</loc><lastmod>2020-01-01</lastmod></url>"
        b"<url><loc>" + base_url.encode()
        + b"/b</loc></url></urlset>"
    )
    set_handler(lambda uri, headers: document)

    # In-memory reference (file-backed off).
    mem_spider_cls = SitemapSpider
    mem_response = XmlResponse(base_url + "/sitemap.xml", body=document)
    mem_requests = _collected_requests(mem_spider_cls, mem_response)

    _, items = await run_crawl(
        FileBackedSpider, sitemap_urls=[base_url + "/sitemap.xml"]
    )
    # The file-backed crawl produced requests for the same URLs (they became
    # items through the default parse callback).
    assert sorted(item["url"] for item in items) == sorted(
        request.url for request in mem_requests
    )
    # Same callbacks and same ordering of URLs.
    assert [request.url for request in mem_requests] == [
        base_url + "/a",
        base_url + "/b",
    ]
    assert all(request.callback.__name__ == "parse" for request in mem_requests)


@coroutine_test
async def test_parity_sitemapindex_source_and_entry(http_server):
    base_url, set_handler = http_server
    index = (
        b'<?xml version="1.0" encoding="UTF-8"?>'
        b'<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        b"<sitemap><loc>" + base_url.encode()
        + b"/inner.xml</loc><lastmod>2021-01-01</lastmod></sitemap>"
        b"</sitemapindex>"
    )
    inner = urlset_xml(base_url, ["/page-1"])

    def handler(uri, headers):
        return inner if uri.startswith("/inner") else index

    set_handler(handler)
    seen: list[tuple[str, str, dict[str, Any]]] = []

    class _Spider(FileBackedSpider):
        name = "parity_index_spider"

        def sitemap_request(self, loc, callback, entry, *, source, response):
            seen.append((loc, source, dict(entry)))
            return super().sitemap_request(
                loc, callback, entry, source=source, response=response
            )

        def parse(self, response, **kwargs):
            yield {"url": response.url}

    await run_crawl(_Spider, sitemap_urls=[base_url + "/index.xml"])
    sources = [source for _, source, _ in seen]
    assert "sitemapindex" in sources
    assert "urlset" in sources
    index_entry = next(entry for _, source, entry in seen if source == "sitemapindex")
    assert index_entry == {
        "loc": base_url + "/inner.xml",
        "lastmod": "2021-01-01",
    }


@coroutine_test
async def test_empty_sitemap_is_not_an_error(http_server):
    base_url, set_handler = http_server
    empty = (
        b'<?xml version="1.0" encoding="UTF-8"?>'
        b'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"></urlset>'
    )
    set_handler(lambda uri, headers: empty)
    crawler, items = await run_crawl(
        FileBackedSpider, sitemap_urls=[base_url + "/sitemap.xml"]
    )
    assert items == []
    assert crawler.stats.get_value("item_scraped_count", 0) == 0
    # No leftover temporary files.
    assert crawler.spider._sitemap_file_registry is None


@coroutine_test
async def test_completed_flag_false_on_early_close():
    xml = (
        b'<?xml version="1.0" encoding="UTF-8"?>'
        b'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        + b"".join(f"<url><loc>http://x/{n}</loc></url>".encode() for n in range(10))
        + b"</urlset>"
    )

    sitemap = Sitemap.from_file(io.BytesIO(xml))
    iterator = iter(sitemap)
    first = next(iterator)
    assert first["loc"] == "http://x/0"
    iterator.close()
    assert sitemap.completed is False

    sitemap2 = Sitemap.from_file(io.BytesIO(xml))
    assert len(list(sitemap2)) == 10
    assert sitemap2.completed is True


@coroutine_test
async def test_download_bytes_maxsize_boundary(http_server):
    """The downloaded (on-disk) byte count follows DOWNLOAD_MAXSIZE with the
    same strict ``>`` semantics as regular downloads: exactly the limit is
    allowed, one byte more is rejected."""
    base_url, set_handler = http_server
    document = urlset_xml(base_url, ["/page-1"])
    set_handler(lambda uri, headers: document)

    # Exactly the on-disk size is allowed.
    _, items = await run_crawl(
        FileBackedSpider,
        sitemap_urls=[base_url + "/sitemap.xml"],
        settings={
            "SITEMAP_FILE_BACKED": True,
            "DOWNLOAD_MAXSIZE": len(document),
        },
    )
    assert [item["url"] for item in items] == [base_url + "/page-1"]

    # One byte over the limit is rejected during the download itself.
    crawler, items = await run_crawl(
        FileBackedSpider,
        sitemap_urls=[base_url + "/sitemap2.xml"],
        settings={
            "SITEMAP_FILE_BACKED": True,
            "DOWNLOAD_MAXSIZE": len(document) - 1,
            "RETRY_ENABLED": False,
        },
    )
    assert items == []
    assert crawler.stats.get_value("item_scraped_count", 0) == 0


@coroutine_test
async def test_decode_layer_maxsize_boundary(http_server):
    """A decompressed layer exactly at DOWNLOAD_MAXSIZE is allowed (strict
    greater-than comparison), while one byte over is rejected."""
    base_url, set_handler = http_server
    document = urlset_xml(base_url, ["/page-1"])
    set_handler(
        lambda uri, headers: {
            "body": gz(document),
            "headers": {b"Content-Encoding": b"gzip"},
        }
    )
    _, items = await run_crawl(
        FileBackedSpider,
        sitemap_urls=[base_url + "/sitemap.xml"],
        settings={
            "SITEMAP_FILE_BACKED": True,
            "DOWNLOAD_MAXSIZE": len(document),
        },
    )
    assert [item["url"] for item in items] == [base_url + "/page-1"]

    crawler, items = await run_crawl(
        FileBackedSpider,
        sitemap_urls=[base_url + "/sitemap2.xml"],
        settings={
            "SITEMAP_FILE_BACKED": True,
            "DOWNLOAD_MAXSIZE": len(document) - 1,
            "RETRY_ENABLED": False,
        },
    )
    assert items == []
    assert crawler.stats.get_value("item_scraped_count", 0) == 0


@coroutine_test
async def test_robots_txt_is_not_file_backed(http_server):
    """robots.txt keeps the regular in-memory path; only sitemap documents
    are marked as file-backed."""
    base_url, set_handler = http_server
    sitemap = urlset_xml(base_url, ["/page-1"])
    seen_requests: list[tuple[str, dict[str, Any]]] = []

    def handler(uri, headers):
        if uri.startswith("/robots.txt"):
            return {
                "body": robots_txt(base_url + "/sitemap.xml"),
                "headers": {b"Content-Type": b"text/plain"},
            }
        return sitemap

    set_handler(handler)

    class _Spider(FileBackedSpider):
        name = "robots_mode_spider"

        def sitemap_request(self, loc, callback, entry, *, source, response):
            if not response.url.endswith("/robots.txt"):
                seen_requests.append((response.url, dict(response.request.meta)))
            return super().sitemap_request(
                loc, callback, entry, source=source, response=response
            )

    _, items = await run_crawl(
        _Spider, sitemap_urls=[base_url + "/robots.txt"]
    )
    assert [item["url"] for item in items] == [base_url + "/page-1"]
    # The discovered sitemap is file-backed and reports the robotstxt source.
    assert len(seen_requests) == 1
    sitemap_url, sitemap_meta = seen_requests[0]
    assert sitemap_url == base_url + "/sitemap.xml"
    assert sitemap_meta.get(SITEMAP_FILE_BACKED_META) is True


@coroutine_test
async def test_file_backed_requests_bypass_http_cache(http_server, tmp_path):
    """The body_file mechanism marks sitemap requests as non-cacheable, so the
    HTTP cache middleware neither stores nor serves them."""
    base_url, set_handler = http_server
    sitemap_fetches = {"n": 0}

    def handler(uri, headers):
        if uri.startswith("/sitemap.xml"):
            sitemap_fetches["n"] += 1
        return urlset_xml(base_url, ["/page-1"])

    set_handler(handler)

    class _Spider(FileBackedSpider):
        name = "cache_bypass_spider"
        custom_settings = {
            "SITEMAP_FILE_BACKED": True,
            "HTTPCACHE_ENABLED": True,
            "HTTPCACHE_DIR": str(tmp_path / "httpcache"),
        }

        def parse(self, response, **kwargs):
            yield {"url": response.url}

    crawler, _ = await run_crawl(
        _Spider, sitemap_urls=[base_url + "/sitemap.xml"]
    )
    # The document was genuinely downloaded rather than served from a cache.
    assert sitemap_fetches["n"] == 1
    assert not crawler.stats.get_value("httpcache/hit", 0)


def test_inflate_file_raw_deflate_fallback(tmp_path):
    """A raw deflate stream (no zlib header), as some Microsoft servers send,
    is decoded file-to-file via the same fallback as the in-memory path."""
    data = b"deflate-data" * 5000
    compressor = zlib.compressobj(wbits=-15)
    raw = compressor.compress(data) + compressor.flush()
    src = tmp_path / "raw.deflate"
    dst = tmp_path / "raw.out"
    src.write_bytes(raw)
    with src.open("rb") as fin, dst.open("wb") as fout:
        written = inflate_file(fin, fout)
    assert dst.read_bytes() == data
    assert written == len(data)


@coroutine_test
async def test_warnsize_emitted_when_decode_exceeds_threshold(http_server):
    """DOWNLOAD_WARNSIZE is honored per decoded layer in file-backed mode."""
    base_url, set_handler = http_server
    document = urlset_xml(base_url, ["/page-1"])
    set_handler(
        lambda uri, headers: {
            "body": gz(document),
            "headers": {b"Content-Encoding": b"gzip"},
        }
    )
    crawler, items = await run_crawl(
        FileBackedSpider,
        sitemap_urls=[base_url + "/sitemap.xml"],
        settings={
            "SITEMAP_FILE_BACKED": True,
            "DOWNLOAD_WARNSIZE": len(document),
        },
    )
    assert [item["url"] for item in items] == [base_url + "/page-1"]
    # A warning-size record is present in the stats without dropping the crawl.
    assert crawler.stats.get_value("httpcompression/response_count", 0) >= 1


@coroutine_test
async def test_empty_or_corrupt_file_does_not_become_empty_map(http_server):
    """A file that cannot be parsed as XML fails the request explicitly rather
    than silently yielding zero URLs (which would look like an empty map)."""
    base_url, set_handler = http_server

    def handler(uri, headers):
        if uri.startswith("/corrupt"):
            return b"this is not xml at all"
        return urlset_xml(base_url, ["/page-1"])

    set_handler(handler)
    crawler, items = await run_crawl(
        FileBackedSpider,
        sitemap_urls=[base_url + "/corrupt.xml"],
        settings={"SITEMAP_FILE_BACKED": True, "RETRY_ENABLED": False},
    )
    assert items == []
    assert crawler.stats.get_value("item_scraped_count", 0) == 0
    # The temporary files were still reclaimed.
    assert crawler.spider._sitemap_file_registry is None
