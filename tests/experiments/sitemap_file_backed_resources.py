"""Resource experiment for file-backed SitemapSpider downloads.

Run in its own subprocess so that the peak allocation of one crawl size and
one compression mode is measured in isolation. Tracing starts only after the
crawler is built and initialized, so import and crawler initialization do not
count toward the peak.

The test side never keeps the XML or the results: entries are counted and
folded into a running checksum as they are produced.

Usage:
    python -I tests/experiments/sitemap_file_backed_resources.py \
        --size-mib 4 --mode plain|gzip|gzip-gzip
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import gzip
import hashlib
import json
import os
import tempfile
import threading
import time
import tracemalloc
from pathlib import Path
from socket import socket as _socket
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

from scrapy.utils.reactor import install_reactor

install_reactor("twisted.internet.asyncioreactor.AsyncioSelectorReactor")

from twisted.internet import reactor  # noqa: E402,TID253
from twisted.internet.endpoints import TCP4ServerEndpoint  # noqa: E402
from twisted.web import resource, server, static  # noqa: E402

from scrapy import signals  # noqa: E402
from scrapy.http import BodyFileResponse  # noqa: E402
from scrapy.spiders import SitemapSpider  # noqa: E402
from scrapy.utils.test import get_crawler  # noqa: E402

PADDING = 7000  # keeps every entry well under 8 KiB
PAGE_COUNT = 32  # fixed pool of final page URLs
MAX_ENTRIES = 50_000


def build_sitemap(base_url: str, target_mib: int, padding: int = PADDING) -> bytes:
    target = target_mib * 1024 * 1024
    parts: list[bytes] = [
        (
            b'<?xml version="1.0" encoding="UTF-8"?>'
            b'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        )
    ]
    n = 0
    base = base_url.encode()
    while sum(len(p) for p in parts) < target and n < MAX_ENTRIES:
        page = n % PAGE_COUNT
        # Highly compressible padding keeps the gzip experiments meaningful.
        parts.append(
            b"<url><loc>"
            + base
            + b"/page-"
            + str(page).encode()
            + b"</loc><data>"
            + b"A" * padding
            + b"</data></url>"
        )
        n += 1
    parts.append(b"</urlset>")
    return b"".join(parts)


def _write_temp_file(data: bytes) -> str:
    fd, path = tempfile.mkstemp(prefix="sitemap-served-")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path


class SitemapResource(resource.Resource):
    """Serves the (possibly encoded) sitemap by streaming it from a file.

    Returning a multi-megabyte ``bytes`` body would make Twisted buffer the
    whole document in this process and pollute the client-side memory
    measurement. The document is therefore written to a temporary file and
    a :class:`twisted.web.static.File` streams it without loading it.
    """

    isLeaf = True

    def __init__(self, body: bytes, mode: str):
        super().__init__()
        if mode == "gzip":
            encoded = gzip.compress(body)
            self.content_encoding: bytes | None = b"gzip"
        elif mode == "gzip-gzip":
            # HTTP gzip layer around a sitemap file that is itself gzipped.
            encoded = gzip.compress(gzip.compress(body))
            self.content_encoding = b"gzip"
        else:
            encoded = body
            self.content_encoding = None
        self._file_path = _write_temp_file(encoded)
        self._static = static.File(self._file_path, defaultType="application/xml")

    def render(self, request: server.Request) -> Any:
        request.setHeader(b"Content-Type", b"application/xml")
        if self.content_encoding:
            request.setHeader(b"Content-Encoding", self.content_encoding)
        return self._static.render(request)

    def cleanup(self) -> None:
        with contextlib.suppress(OSError):
            Path(self._file_path).unlink()


class PageResource(resource.Resource):
    isLeaf = True

    def render(self, request: server.Request) -> bytes:
        request.setHeader(b"Content-Type", b"text/plain")
        return b"page"


class _Root(resource.Resource):
    """Routes /sitemap.xml and /sitemap.xml.gz and every /page-* path."""

    def __init__(self, sitemap_resource: resource.Resource):
        super().__init__()
        self.putChild(b"sitemap.xml", sitemap_resource)
        self.putChild(b"sitemap.xml.gz", sitemap_resource)
        self._page = PageResource()

    def getChildWithDefault(self, path: bytes, request: server.Request) -> Any:
        if path.startswith(b"page"):
            return self._page
        return super().getChildWithDefault(path, request)

    def render(self, request: server.Request) -> bytes:
        request.setResponseCode(404)
        return b"not found"


class _RssSampler(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.peak = 0
        self._stop_event = threading.Event()
        self._page_size = os.sysconf("SC_PAGE_SIZE")

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                with Path("/proc/self/statm").open("rb") as f:
                    resident_pages = int(f.read().split()[1])
            except OSError:
                return
            self.peak = max(self.peak, resident_pages * self._page_size)
            self._stop_event.wait(0.02)

    def stop(self) -> None:
        self._stop_event.set()


class _DiskSampler(threading.Thread):
    """Records the peak total size of the files in the registry directory."""

    def __init__(self, directory: str) -> None:
        super().__init__(daemon=True)
        self._directory = Path(directory)
        self.peak = 0
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.is_set():
            total = 0
            try:
                for path in self._directory.iterdir():
                    with contextlib.suppress(OSError):
                        total += path.stat().st_size
            except FileNotFoundError:
                return
            self.peak = max(self.peak, total)
            self._stop_event.wait(0.01)

    def stop(self) -> None:
        self._stop_event.set()


def run(target_mib: int, mode: str) -> dict[str, Any]:
    # Reserve a free port before the reactor runs, so the generated sitemap
    # points at the final URLs and no reactor restart is ever needed.
    sock = _socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    base_url = f"http://127.0.0.1:{port}"

    outcome: dict[str, Any] = {}
    xml = build_sitemap(base_url, target_mib)
    decoded_xml_size = len(xml)
    sitemap_resource = SitemapResource(xml, mode)
    root = _Root(sitemap_resource)
    # The document lives on disk from here on; drop the in-process bytes so
    # they cannot contribute to the measured peak.
    del xml

    class ExperimentSpider(SitemapSpider):
        name = "resource_experiment"
        custom_settings = {
            "SITEMAP_FILE_BACKED": True,
            "CONCURRENT_REQUESTS": 1,
            "RETRY_ENABLED": False,
            "TELNETCONSOLE_ENABLED": False,
        }
        entry_count = 0
        checksum = 0
        first_url_at: float | None = None

        def sitemap_request(self, loc, callback, entry, *, source, response):
            assert isinstance(response, BodyFileResponse)
            now = time.monotonic()
            if type(self).first_url_at is None:
                type(self).first_url_at = now
            type(self).entry_count += 1
            type(self).checksum ^= int.from_bytes(
                hashlib.blake2b(entry["loc"].encode(), digest_size=8).digest(),
                "little",
            )
            return super().sitemap_request(
                loc, callback, entry, source=source, response=response
            )

        def parse(self, response, **kwargs):
            # Consume the page one at a time, storing nothing.
            yield {"seen": True}

    # Crawler initialization happens before measurement starts. The spider is
    # created explicitly and assigned to the crawler, so the same instance
    # (and its registry) is used by the whole crawl.
    crawler = get_crawler(ExperimentSpider)
    items: list[Any] = []

    async def _on_item_scraped(item: Any) -> None:
        # item_scraped is sent through the async signal path; an async
        # receiver avoids the synchronous-dispatch mismatch that silently
        # drops a plain function receiver in this standalone reactor setup.
        items.append(item)

    crawler.signals.connect(_on_item_scraped, signal=signals.item_scraped)
    sitemap_url = f"{base_url}/sitemap.xml"
    if mode == "gzip-gzip":
        sitemap_url += ".gz"
    spider = ExperimentSpider.from_crawler(
        crawler, name=ExperimentSpider.name, sitemap_urls=[sitemap_url]
    )
    # crawl_async() would build a second spider via _create_spider(); keep the
    # one whose registry is being measured.
    crawler._create_spider = lambda *args, **kwargs: spider  # type: ignore[method-assign]
    assert spider._sitemap_file_registry is not None
    registry_dir = str(spider._sitemap_file_registry.directory)

    (
        traced_peak,
        rss_peak,
        disk_peak,
        first_url_seconds,
    ) = _measure_crawl(
        crawler, root, port, rss_sampler_factory=_RssSampler,
        disk_sampler_factory=lambda: _DiskSampler(registry_dir),
        first_url_attr=lambda: ExperimentSpider.first_url_at,
        outcome=outcome,
    )
    sitemap_resource.cleanup()

    return {
        "size_mib": target_mib,
        "mode": mode,
        "xml_bytes": decoded_xml_size,
        "entries": ExperimentSpider.entry_count,
        "checksum": ExperimentSpider.checksum,
        "items_scraped": len(items),
        "traced_python_peak_bytes": traced_peak,
        "rss_peak_bytes": rss_peak,
        "temp_disk_peak_bytes": disk_peak,
        "first_url_seconds": first_url_seconds,
        "error": outcome.get("error"),
    }


def _measure_crawl(
    crawler: Any,
    root: resource.Resource,
    port: int,
    rss_sampler_factory: Callable[[], _RssSampler],
    disk_sampler_factory: Callable[[], _DiskSampler],
    first_url_attr: Callable[[], float | None],
    outcome: dict[str, Any],
) -> tuple[int, int, int, float | None]:
    """Run one crawl under tracemalloc and the sampling threads, returning the
    Python allocation peak, the RSS peak, the temporary-disk peak and the time
    of the first produced URL."""
    rss_sampler = rss_sampler_factory()
    disk_sampler = disk_sampler_factory()
    tracemalloc.start()
    rss_sampler.start()
    disk_sampler.start()
    start = time.monotonic()

    async def _crawl() -> None:
        try:
            await crawler.crawl_async()
        except BaseException as exc:  # keep the failure visible in the output
            outcome["error"] = repr(exc)
        finally:
            rss_sampler.stop()
            disk_sampler.stop()
            reactor.stop()

    def _start() -> None:
        endpoint = TCP4ServerEndpoint(reactor, port, interface="127.0.0.1")
        endpoint.listen(server.Site(root))
        # Let the endpoint start listening before the crawl begins.
        reactor.callLater(0.05, lambda: asyncio.ensure_future(_crawl()))

    reactor.callWhenRunning(_start)
    reactor.run()

    _, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_sampler.join(timeout=2)
    disk_sampler.join(timeout=2)
    first_url = first_url_attr()
    first_url_seconds = first_url - start if first_url is not None else None
    return traced_peak, rss_sampler.peak, disk_sampler.peak, first_url_seconds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size-mib", type=int, required=True)
    parser.add_argument(
        "--mode", choices=["plain", "gzip", "gzip-gzip"], required=True
    )
    args = parser.parse_args()
    result = run(args.size_mib, args.mode)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
