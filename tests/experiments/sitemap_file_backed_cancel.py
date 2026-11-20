"""Resource experiment for cancelling a file-backed sitemap decompression.

Measures, for an artificial sitemap whose *decoded* size is ``--size-mib``
MiB, served behind an HTTP gzip ``Content-Encoding``:

* cancel latency: from cancelling the download coroutine — parked in the
  compression middleware's decode wait by a threading barrier — to the
  caller observing :class:`asyncio.CancelledError`;
* cleanup completion: from releasing the worker thread to the decode slot
  being settled and every temporary file being reclaimed. Two barrier
  placements are measured: parked *before* the decode starts (cleanup then
  waits for the whole remaining decode — the honest cost of not being able
  to stop a thread) and parked *after* it (cleanup is then immediate);
* extra Python memory (tracemalloc peak) and RSS peak during the cancelled
  decode and its cleanup — the compressed and decoded bodies must never be
  pulled into Python to transfer ownership;
* peak temporary-disk usage of the registry directory;
* a baseline successful crawl of the document, consumed entry by entry
  (never collected), proving the pipeline still streams at this size.

Run in its own subprocess per size, like
:mod:`tests.experiments.sitemap_file_backed_resources`. Emits one JSON line.

Usage:
    python -I tests/experiments/sitemap_file_backed_cancel.py --size-mib 8
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

from scrapy.utils.reactor import install_reactor

install_reactor("twisted.internet.asyncioreactor.AsyncioSelectorReactor")

from twisted.internet import reactor  # noqa: E402,TID253
from twisted.internet.endpoints import TCP4ServerEndpoint  # noqa: E402
from twisted.web import resource, server, static  # noqa: E402

from scrapy import Request  # noqa: E402
from scrapy.core.engine import ExecutionEngine  # noqa: E402
from scrapy.downloadermiddlewares import httpcompression as hcm  # noqa: E402
from scrapy.spiders import SitemapSpider  # noqa: E402
from scrapy.utils.test import get_crawler  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Callable

PADDING = 7000  # keeps every entry well under 8 KiB
PAGE_COUNT = 32
MAX_ENTRIES = 500_000


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


class SitemapResource(resource.Resource):
    """Serves the gzipped sitemap from a file, with Content-Encoding: gzip."""

    isLeaf = True

    def __init__(self, body: bytes):
        super().__init__()
        fd, self._file_path = tempfile.mkstemp(prefix="sitemap-served-")
        with os.fdopen(fd, "wb") as f:
            f.write(gzip.compress(body))
        self._static = static.File(self._file_path, defaultType="application/xml")

    def render(self, request: server.Request) -> Any:
        request.setHeader(b"Content-Type", b"application/xml")
        request.setHeader(b"Content-Encoding", b"gzip")
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
    def __init__(self, sitemap_resource: resource.Resource):
        super().__init__()
        self.putChild(b"sitemap.xml", sitemap_resource)
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
    """Records the peak total size of the files in a directory."""

    def __init__(self, directory: str):
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
                pass
            self.peak = max(self.peak, total)
            self._stop_event.wait(0.005)

    def stop(self) -> None:
        self._stop_event.set()


class _DecodeBarrier:
    """Parks the middleware's gunzip_file call at a deterministic point.

    With ``park="before"`` the worker is stopped before the decode starts;
    with ``park="after"`` it is stopped once the decoded file is fully
    written, before the decode call returns.
    """

    def __init__(self, park: str) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.armed = True
        self._park = park
        self._original = hcm.gunzip_file
        barrier = self

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not barrier.armed:
                return barrier._original(*args, **kwargs)
            barrier.armed = False
            if barrier._park == "before":
                barrier.entered.set()
                barrier.release.wait(60)
                return barrier._original(*args, **kwargs)
            result = barrier._original(*args, **kwargs)
            barrier.entered.set()
            barrier.release.wait(60)
            return result

        hcm.gunzip_file = wrapper  # type: ignore[attr-defined]

    def open(self) -> None:
        self.release.set()

    def restore(self) -> None:
        hcm.gunzip_file = self._original  # type: ignore[attr-defined]


async def _wait_until(predicate: Callable[[], bool], max_seconds: float) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + max_seconds
    while not predicate():
        if loop.time() >= deadline:
            raise TimeoutError("condition not met")
        await asyncio.sleep(0.01)


@contextlib.contextmanager
def _measure(registry_dir: str, outcome: dict[str, Any]) -> Any:
    """Run the tracemalloc, RSS and disk measurements around a phase."""
    disk_sampler = _DiskSampler(registry_dir)
    rss_sampler = _RssSampler()
    tracemalloc.start()
    disk_sampler.start()
    rss_sampler.start()
    try:
        yield disk_sampler
    finally:
        disk_sampler.stop()
        rss_sampler.stop()
        _, traced_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        disk_sampler.join(timeout=2)
        rss_sampler.join(timeout=2)
        outcome["traced_python_peak_bytes"] = traced_peak
        outcome["rss_peak_bytes"] = rss_sampler.peak
        outcome["temp_disk_peak_bytes"] = disk_sampler.peak


async def _cancel_phase(
    sitemap_url: str, park: str, decoded_size: int
) -> dict[str, Any]:
    """Download the sitemap through the real middleware stack, cancel the
    coroutine while the decode is parked, then release the worker thread and
    measure the cleanup."""
    barrier = _DecodeBarrier(park)
    outcome: dict[str, Any] = {"park": park}

    class CancelSpider(SitemapSpider):
        name = "cancel_experiment_cancel"
        custom_settings = {
            "SITEMAP_FILE_BACKED": True,
            "RETRY_ENABLED": False,
            "TELNETCONSOLE_ENABLED": False,
        }

    crawler = get_crawler(CancelSpider)
    spider = CancelSpider.from_crawler(crawler, name=CancelSpider.name)
    crawler.spider = spider
    crawler.engine = ExecutionEngine(crawler)
    registry = spider._sitemap_file_registry
    assert registry is not None
    registry_dir = str(registry.directory)
    started = time.monotonic()
    try:
        with _measure(registry_dir, outcome) as disk_sampler:
            await crawler.engine.open_spider_async()
            request = spider._mark_sitemap_request(Request(sitemap_url))
            task = asyncio.ensure_future(crawler.engine.download_async(request))
            await asyncio.wait_for(asyncio.to_thread(barrier.entered.wait, 60), 65)
            if park == "after":
                # The decoded file is fully written and parked on disk: give
                # the disk sampler a deterministic chance to record the peak
                # before cancelling.
                await _wait_until(lambda: disk_sampler.peak >= decoded_size * 0.95, 60)
            # Cancel the coroutine awaiting the decode; the caller must
            # observe the cancellation immediately, without waiting for the
            # thread.
            t_cancel = time.monotonic()
            task.cancel()
            try:
                await task
                outcome["error"] = "download_async returned despite cancellation"
            except asyncio.CancelledError:
                pass
            t_cancelled = time.monotonic()
            # Let the worker thread run. Cleanup finishes when the decode
            # slot is settled and the registry directory is empty.
            barrier.open()
            await _wait_until(lambda: registry.pending_decodes == 0, 300)
            await _wait_until(lambda: not any(Path(registry_dir).iterdir()), 300)
            t_cleaned = time.monotonic()
            outcome.update(
                {
                    "cancel_latency_seconds": t_cancelled - t_cancel,
                    "cleanup_seconds": t_cleaned - t_cancelled,
                    "files_left": sorted(p.name for p in Path(registry_dir).iterdir()),
                    "pending_decodes": registry.pending_decodes,
                    "httpcompression_count": crawler.stats.get_value(
                        "httpcompression/response_count", 0
                    ),
                }
            )
    finally:
        barrier.open()
        barrier.restore()
        close_started = time.monotonic()
        await crawler.engine.close_async(reason="experiment_done")
        outcome["engine_close_seconds"] = time.monotonic() - close_started
        outcome["total_seconds"] = time.monotonic() - started
        outcome["registry_dir_exists"] = Path(registry_dir).exists()
    return outcome


async def _baseline_crawl(sitemap_url: str) -> dict[str, Any]:
    """A regular crawl of the document; entries are consumed one at a time."""

    class BaselineSpider(SitemapSpider):
        name = "cancel_experiment_baseline"
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
            yield {"seen": True}

    crawler = get_crawler(BaselineSpider)
    start = time.monotonic()
    await crawler.crawl_async(sitemap_urls=[sitemap_url])
    first_url_at = BaselineSpider.first_url_at
    return {
        "entries": BaselineSpider.entry_count,
        "checksum": BaselineSpider.checksum,
        "first_url_seconds": None if first_url_at is None else first_url_at - start,
    }


def run(target_mib: int) -> dict[str, Any]:
    sock = _socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    base_url = f"http://127.0.0.1:{port}"

    xml = build_sitemap(base_url, target_mib)
    decoded_xml_size = len(xml)
    sitemap_resource = SitemapResource(xml)
    del xml
    root = _Root(sitemap_resource)
    sitemap_url = f"{base_url}/sitemap.xml"

    outcome: dict[str, Any] = {}

    async def _experiment() -> None:
        try:
            # Cancel phases first, so their RSS measurements are not
            # polluted by the baseline crawl's allocator arenas.
            outcome["cancel_before_decode"] = await _cancel_phase(
                sitemap_url, "before", decoded_xml_size
            )
            outcome["cancel_after_decode"] = await _cancel_phase(
                sitemap_url, "after", decoded_xml_size
            )
            outcome["baseline"] = await _baseline_crawl(sitemap_url)
        except BaseException as exc:
            outcome["error"] = repr(exc)
        finally:
            reactor.stop()

    def _start() -> None:
        endpoint = TCP4ServerEndpoint(reactor, port, interface="127.0.0.1")
        endpoint.listen(server.Site(root))
        reactor.callLater(0.05, lambda: asyncio.ensure_future(_experiment()))

    reactor.callWhenRunning(_start)
    reactor.run()
    sitemap_resource.cleanup()
    outcome["size_mib"] = target_mib
    outcome["xml_bytes"] = decoded_xml_size
    return outcome


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size-mib", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.size_mib)))


if __name__ == "__main__":
    main()
