"""Tests for the cancellation and close coordination of file-backed sitemap
decompression (``SITEMAP_FILE_BACKED`` + HTTP ``Content-Encoding``).

The file-to-file decode of a sitemap response runs in a worker thread, and
cancelling the middleware coroutine that awaits it cannot stop that thread.
These tests pin down the ownership contract for that race:

* a cancelled wait still surfaces :class:`asyncio.CancelledError` to the
  caller, and the decode output is reclaimed once the worker thread
  finishes, without a success response, success stats or a parse;
* a decode failure keeps surfacing the actual decompression exception;
* the registry close waits for in-flight decodes, rejects late
  registrations and stays idempotent, so no file is ever left without an
  owner.

All synchronization is deterministic: worker threads are parked on
:class:`threading.Event` barriers (with a backstop timeout so a failing
test can never wedge an executor thread), every wait has a watchdog, and
barriers are released in ``finally`` blocks.
"""

from __future__ import annotations

import asyncio
import contextlib
import gc
import gzip
import os
import tempfile
import threading
import zlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from twisted.web import resource, server

from scrapy import signals
from scrapy.downloadermiddlewares import httpcompression as hcm
from scrapy.downloadermiddlewares.httpcompression import HttpCompressionMiddleware
from scrapy.exceptions import DecompressionError, IgnoreRequest
from scrapy.http import BodyFileResponse, Request
from scrapy.spiders import SitemapSpider
from scrapy.spiders._sitemap_files import (
    SITEMAP_FILE_BACKED_META,
    RegistryClosedError,
    SitemapFileRegistry,
)
from scrapy.utils.test import get_crawler
from tests.utils.decorators import coroutine_test

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from twisted.internet.tcp import Port


# The cancellation machinery these tests pin down (the shielded decode wait
# and its disposal callback) only exists when an asyncio loop is available.
pytestmark = pytest.mark.only_asyncio

WATCHDOG = 2.0  # seconds; the bound within which cleanup must finish

XML = (
    b'<?xml version="1.0" encoding="UTF-8"?>'
    b'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
    b"<url><loc>http://example.com/page-1</loc></url>"
    b"</urlset>"
)
XML2 = (
    b'<?xml version="1.0" encoding="UTF-8"?>'
    b'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
    b"<url><loc>http://example.com/page-2</loc></url>"
    b"</urlset>"
)


def gz(data: bytes) -> bytes:
    return gzip.compress(data)


def dfl(data: bytes) -> bytes:
    return zlib.compress(data)


def urlset_xml(base: str, paths: list[str]) -> bytes:
    urls = "".join(f"<url><loc>{base}{p}</loc></url>" for p in paths)
    return (
        b'<?xml version="1.0" encoding="UTF-8"?>'
        b'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        + urls.encode()
        + b"</urlset>"
    )


# --------------------------------------------------------------------------- #
# Deterministic decode barrier
# --------------------------------------------------------------------------- #


class DecodeBarrier:
    """Park the worker-thread decode at a deterministic point.

    Patches a decode function of the HTTP compression middleware so that the
    next (and only the next) call runs the real decode — with its input and
    output files open — and then blocks until :attr:`release` is set. A
    backstop timeout guarantees that a failing test never wedges the
    executor thread.
    """

    BACKSTOP = 30.0  # seconds

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        func_name: str = "gunzip_file",
        *,
        fail_with: Exception | None = None,
    ) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.returned = threading.Event()
        self.armed = True
        self._fail_with = fail_with
        original = getattr(hcm, func_name)

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not self.armed:
                return original(*args, **kwargs)
            self.armed = False
            try:
                result = original(*args, **kwargs)
                self.entered.set()
                self.release.wait(self.BACKSTOP)
                if self._fail_with is not None:
                    raise self._fail_with
                return result
            finally:
                self.returned.set()

        monkeypatch.setattr(hcm, func_name, wrapper)

    def open(self) -> None:
        """Release the parked worker thread."""
        self.release.set()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


async def wait_until(
    predicate: Callable[[], bool],
    *,
    max_seconds: float = WATCHDOG,
    interval: float = 0.005,
    what: str = "condition",
) -> None:
    """Poll *predicate* until it holds; raise TimeoutError after *max_seconds*."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + max_seconds
    while not predicate():
        if loop.time() >= deadline:
            raise TimeoutError(f"{what} not met within {max_seconds} s")
        await asyncio.sleep(interval)


async def loop_ticks(n: int = 5) -> None:
    """Yield to the event loop a fixed number of times (no timing involved)."""
    for _ in range(n):
        await asyncio.sleep(0)


@contextlib.contextmanager
def captured_loop_errors() -> Iterator[list[dict[str, Any]]]:
    """Collect reports sent to the asyncio exception handler (e.g. task
    exceptions that were never retrieved)."""
    contexts: list[dict[str, Any]] = []
    loop = asyncio.get_running_loop()
    previous = loop.get_exception_handler()
    loop.set_exception_handler(lambda loop, ctx: contexts.append(ctx))
    try:
        yield contexts
    finally:
        loop.set_exception_handler(previous)


class CancelSpider(SitemapSpider):
    name = "cancel_spider"
    custom_settings: dict[str, Any] = {"SITEMAP_FILE_BACKED": True}


def build_middleware(
    settings: dict[str, Any] | None = None,
) -> tuple[Any, SitemapSpider, HttpCompressionMiddleware, SitemapFileRegistry]:
    """Build a crawler, spider and middleware wired like in a real crawl."""
    crawler = get_crawler(CancelSpider, settings_dict=settings or {})
    spider = CancelSpider.from_crawler(crawler)
    crawler.spider = spider
    mw = HttpCompressionMiddleware.from_crawler(crawler)
    crawler.stats.open_spider()
    registry = spider._sitemap_file_registry
    assert registry is not None
    return crawler, spider, mw, registry


def make_body_response(
    registry: SitemapFileRegistry,
    data: bytes,
    encoding: bytes = b"gzip",
    url: str = "http://example.com/sitemap.xml",
) -> tuple[Request, BodyFileResponse]:
    """Create the request/response pair the download handler would have
    produced for a file-backed sitemap download."""
    fd, path = tempfile.mkstemp(dir=registry.directory, prefix="body-")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    request = Request(url, meta={SITEMAP_FILE_BACKED_META: True, "body_file": path})
    response = BodyFileResponse(
        url,
        headers={"Content-Encoding": encoding},
        body_file=path,
        request=request,
    )
    return request, response


@pytest.fixture
def registries() -> Iterator[list[SitemapFileRegistry]]:
    """Track created registries so they are closed even if a test fails."""
    created: list[SitemapFileRegistry] = []
    yield created
    for registry in created:
        registry.close()


@pytest.fixture
def tracked_build(registries):
    def _build(settings=None):
        crawler, spider, mw, registry = build_middleware(settings)
        registries.append(registry)
        return crawler, spider, mw, registry

    return _build


def directory_files(registry: SitemapFileRegistry) -> list[str]:
    directory = registry.directory
    if not directory.exists():
        return []
    return sorted(p.name for p in directory.iterdir())


# --------------------------------------------------------------------------- #
# Registry close semantics
# --------------------------------------------------------------------------- #


def test_late_track_after_close_is_rejected(registries):
    registry = SitemapFileRegistry()
    registries.append(registry)
    directory = registry.directory
    registry.close()
    assert registry.closed
    # A late registration is not taken over: the caller learns it through
    # RegistryClosedError instead of the file silently losing its owner.
    with pytest.raises(RegistryClosedError):
        registry.track(directory / "late-file")
    # A repeated close is a no-op and leaves no pending ownership behind.
    registry.close()
    assert not registry._paths
    assert not directory.exists()


@coroutine_test
async def test_close_async_waits_for_pending_decode(registries):
    registry = SitemapFileRegistry()
    registries.append(registry)
    slot = registry.begin_decode()
    assert registry.pending_decodes == 1
    close_task = asyncio.ensure_future(registry.close_async())
    await loop_ticks()
    # The decode slot is still open, so reclamation cannot have finished.
    assert not close_task.done()
    assert not registry.closed
    slot.close()
    await asyncio.wait_for(close_task, WATCHDOG)
    assert registry.closed
    assert not registry.directory.exists()


@coroutine_test
async def test_begin_decode_rejected_once_closing(registries):
    registry = SitemapFileRegistry()
    registries.append(registry)
    slot = registry.begin_decode()
    close_task = asyncio.ensure_future(registry.close_async())
    await loop_ticks()
    with pytest.raises(RegistryClosedError):
        registry.begin_decode()
    slot.close()
    await asyncio.wait_for(close_task, WATCHDOG)
    with pytest.raises(RegistryClosedError):
        registry.begin_decode()


@coroutine_test
async def test_concurrent_and_repeated_close_async(registries):
    registry = SitemapFileRegistry()
    registries.append(registry)
    slot = registry.begin_decode()
    first = asyncio.ensure_future(registry.close_async())
    second = asyncio.ensure_future(registry.close_async())
    await loop_ticks()
    assert not first.done()
    assert not second.done()
    slot.close()
    await asyncio.wait_for(asyncio.gather(first, second), WATCHDOG)
    assert registry.closed
    # Closing again completes immediately and changes nothing.
    await asyncio.wait_for(registry.close_async(), WATCHDOG)
    assert registry.closed


@coroutine_test
async def test_close_async_cancelled_while_waiting_still_sweeps(registries):
    """A close_async() cancelled while waiting for a decode must not leave
    the registry closing-but-never-swept: it degrades to the immediate
    sweep of the synchronous close."""
    registry = SitemapFileRegistry()
    registries.append(registry)
    slot = registry.begin_decode()
    close_task = asyncio.ensure_future(registry.close_async())
    await loop_ticks()
    assert not close_task.done()
    close_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await close_task
    assert registry.closed
    assert not registry.directory.exists()
    # Settling the slot afterwards is harmless, and a later close completes
    # instead of hanging on a close that never finishes.
    slot.close()
    assert registry.pending_decodes == 0
    await asyncio.wait_for(registry.close_async(), WATCHDOG)


def test_sync_close_does_not_wait_but_rejects_late_ownership(registries):
    registry = SitemapFileRegistry()
    registries.append(registry)
    slot = registry.begin_decode()
    registry.close()
    # The synchronous close sweeps immediately, without waiting for the
    # in-flight decode; the directory is gone and late files are rejected.
    assert registry.closed
    assert registry.pending_decodes == 1
    with pytest.raises(RegistryClosedError):
        registry.track(registry.directory / "late")
    # Settling the slot afterwards is harmless.
    slot.close()
    assert registry.pending_decodes == 0
    registry.close()


def test_release_after_close_is_a_safe_noop(registries):
    registry = SitemapFileRegistry()
    registries.append(registry)
    path = registry.directory / "file"
    path.write_bytes(b"x")
    registry.track(path)
    registry.close()
    # Releasing an already reclaimed file after close must not fail.
    registry.release(path)
    registry.release(registry.directory / "never-existed")


# --------------------------------------------------------------------------- #
# Middleware: cancellation of the decode wait
# --------------------------------------------------------------------------- #


@coroutine_test
async def test_cancel_during_decode_reclaims_files(monkeypatch, tracked_build):
    """The scenario from the bug report: cancel the middleware coroutine
    while the decoder thread is parked with its input and output files open,
    then let the thread finish. The decoded file must be reclaimed."""
    crawler, _spider, mw, registry = tracked_build()
    barrier = DecodeBarrier(monkeypatch)
    request, response = make_body_response(registry, gz(XML))
    with captured_loop_errors() as loop_errors:
        task = asyncio.ensure_future(mw.process_response(request, response))
        assert await asyncio.to_thread(barrier.entered.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        try:
            barrier.open()
            # Within the watchdog: the thread finished, its output file was
            # released, the decode slot was settled.
            await wait_until(
                lambda: registry.pending_decodes == 0,
                what="decode slot settled",
            )
        finally:
            barrier.open()
        assert directory_files(registry) == []
        assert not registry._paths
        # No success response, no success stats, no parse happened.
        assert crawler.stats.get_value("httpcompression/response_count", 0) == 0
        assert crawler.stats.get_value("httpcompression/response_bytes", 0) == 0
        gc.collect()
        await loop_ticks()
        assert not [
            ctx for ctx in loop_errors if "never retrieved" in ctx.get("message", "")
        ]


@coroutine_test
async def test_cancel_during_deflate_decode(monkeypatch, tracked_build):
    """Same as above for the deflate layer (inflate_file)."""
    crawler, _spider, mw, registry = tracked_build()
    barrier = DecodeBarrier(monkeypatch, "inflate_file")
    request, response = make_body_response(registry, dfl(XML), b"deflate")
    task = asyncio.ensure_future(mw.process_response(request, response))
    try:
        assert await asyncio.to_thread(barrier.entered.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        barrier.open()
        await wait_until(lambda: registry.pending_decodes == 0)
    finally:
        barrier.open()
    assert directory_files(registry) == []
    assert crawler.stats.get_value("httpcompression/response_count", 0) == 0


@coroutine_test
async def test_cancel_racing_decode_completion(monkeypatch, tracked_build):
    """The worker thread finishes first, but the cancellation is delivered
    before the middleware coroutine resumes: the result must still be
    discarded and reclaimed, never delivered."""
    crawler, _spider, mw, registry = tracked_build()
    barrier = DecodeBarrier(monkeypatch)
    request, response = make_body_response(registry, gz(XML))
    task = asyncio.ensure_future(mw.process_response(request, response))
    try:
        assert await asyncio.to_thread(barrier.entered.wait, 5)
        barrier.open()
        # Block the loop until the worker thread has returned. The decode
        # task is now complete (or about to be), while the middleware
        # coroutine is still suspended, so the cancellation lands on the
        # shielded wait.
        assert barrier.returned.wait(5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await wait_until(lambda: registry.pending_decodes == 0)
    finally:
        barrier.open()
    assert directory_files(registry) == []
    assert not registry._paths
    assert crawler.stats.get_value("httpcompression/response_count", 0) == 0


@coroutine_test
async def test_repeated_cancel(monkeypatch, tracked_build):
    crawler, _spider, mw, registry = tracked_build()
    barrier = DecodeBarrier(monkeypatch)
    request, response = make_body_response(registry, gz(XML))
    task = asyncio.ensure_future(mw.process_response(request, response))
    try:
        assert await asyncio.to_thread(barrier.entered.wait, 5)
        task.cancel()
        task.cancel()  # cancelling again while unwinding changes nothing
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not task.cancel()  # already cancelled: no-op
        barrier.open()
        await wait_until(lambda: registry.pending_decodes == 0)
    finally:
        barrier.open()
    assert directory_files(registry) == []
    assert crawler.stats.get_value("httpcompression/response_count", 0) == 0


@coroutine_test
async def test_decode_error_keeps_actual_exception(tracked_build):
    """A failing decode (no cancellation) surfaces the actual decompression
    exception and still reclaims every file."""
    crawler, _spider, mw, registry = tracked_build()
    truncated = gz(XML)[:-8]  # cut the gzip trailer
    request, response = make_body_response(registry, truncated)
    with pytest.raises(DecompressionError) as excinfo:
        await mw.process_response(request, response)
    # The actual decompression exception is preserved as the cause.
    assert isinstance(excinfo.value.__cause__, EOFError)
    assert registry.pending_decodes == 0
    assert directory_files(registry) == []
    assert crawler.stats.get_value("httpcompression/response_count", 0) == 0


@coroutine_test
async def test_cancel_then_decode_error(monkeypatch, tracked_build, caplog):
    """Cancelled first, then the worker thread fails: the caller keeps the
    CancelledError, the thread exception is retrieved and logged, and the
    files are reclaimed."""
    crawler, _spider, mw, registry = tracked_build()
    barrier = DecodeBarrier(monkeypatch, fail_with=EOFError("boom"))
    request, response = make_body_response(registry, gz(XML))
    with captured_loop_errors() as loop_errors:
        task = asyncio.ensure_future(mw.process_response(request, response))
        try:
            assert await asyncio.to_thread(barrier.entered.wait, 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            barrier.open()
            await wait_until(lambda: registry.pending_decodes == 0)
        finally:
            barrier.open()
        assert directory_files(registry) == []
        assert crawler.stats.get_value("httpcompression/response_count", 0) == 0
        # The late decode failure was logged by the disposal, not lost.
        assert "cancelled file-backed sitemap" in caplog.text
        gc.collect()
        await loop_ticks()
        assert not [
            ctx for ctx in loop_errors if "never retrieved" in ctx.get("message", "")
        ]


@coroutine_test
async def test_cancel_between_encoding_layers(monkeypatch, tracked_build):
    """A two-layer (gzip, deflate) response cancelled while the second layer
    decodes: the intermediate and the final file are both reclaimed."""
    crawler, _spider, mw, registry = tracked_build({"COMPRESSION_MAX_ENCODINGS": 2})
    # Decoding order is deflate first, then gzip: park the gzip layer.
    barrier = DecodeBarrier(monkeypatch, "gunzip_file")
    request, response = make_body_response(registry, dfl(gz(XML)), b"gzip, deflate")
    task = asyncio.ensure_future(mw.process_response(request, response))
    try:
        assert await asyncio.to_thread(barrier.entered.wait, 5)
        # The first layer's intermediate file exists while the second is parked.
        assert directory_files(registry)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        barrier.open()
        await wait_until(lambda: registry.pending_decodes == 0)
    finally:
        barrier.open()
    assert directory_files(registry) == []
    assert not registry._paths
    assert crawler.stats.get_value("httpcompression/response_count", 0) == 0


@coroutine_test
async def test_success_after_barrier_release(monkeypatch, tracked_build):
    """No cancellation: once the barrier opens, the decode delivers a
    readable file with the usual stats and headers."""
    crawler, _spider, mw, registry = tracked_build()
    barrier = DecodeBarrier(monkeypatch)
    request, response = make_body_response(registry, gz(XML))
    task = asyncio.ensure_future(mw.process_response(request, response))
    try:
        assert await asyncio.to_thread(barrier.entered.wait, 5)
        barrier.open()
        result = await asyncio.wait_for(task, WATCHDOG)
    finally:
        barrier.open()
    assert isinstance(result, BodyFileResponse)
    assert result is not response
    assert Path(result.body_file).read_bytes() == XML
    assert "Content-Encoding" not in result.headers
    assert crawler.stats.get_value("httpcompression/response_count") == 1
    assert crawler.stats.get_value("httpcompression/response_bytes") == len(XML)
    assert registry.pending_decodes == 0
    # The file is owned by the response now; releasing it (as the spider
    # would after parsing) leaves nothing behind.
    registry.release(Path(result.body_file))
    assert directory_files(registry) == []


@coroutine_test
async def test_parallel_decode_unaffected_by_stuck_cancellation(
    monkeypatch, tracked_build
):
    """While one decode is parked and then cancelled, another sitemap of the
    same spider still decodes and delivers normally (no global lock)."""
    crawler, _spider, mw, registry = tracked_build()
    barrier = DecodeBarrier(monkeypatch)  # only the first decode is parked
    req1, resp1 = make_body_response(registry, gz(XML), url="http://example.com/s1.xml")
    task1 = asyncio.ensure_future(mw.process_response(req1, resp1))
    try:
        assert await asyncio.to_thread(barrier.entered.wait, 5)
        req2, resp2 = make_body_response(
            registry, gz(XML2), url="http://example.com/s2.xml"
        )
        result2 = await asyncio.wait_for(mw.process_response(req2, resp2), WATCHDOG)
        assert Path(result2.body_file).read_bytes() == XML2
        assert crawler.stats.get_value("httpcompression/response_count") == 1
        task1.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task1
        barrier.open()
        await wait_until(lambda: registry.pending_decodes == 0)
    finally:
        barrier.open()
    # The successful response file is still owned by its response.
    assert Path(result2.body_file).read_bytes() == XML2
    assert crawler.stats.get_value("httpcompression/response_count") == 1
    registry.release(Path(result2.body_file))
    assert directory_files(registry) == []


@coroutine_test
async def test_decode_failure_then_next_sitemap_succeeds(tracked_build):
    """A failure does not poison the registry: the next sitemap decodes."""
    _crawler, _spider, mw, registry = tracked_build()
    bad_req, bad_resp = make_body_response(
        registry, gz(XML)[:-8], url="http://example.com/bad.xml"
    )
    with pytest.raises(DecompressionError):
        await mw.process_response(bad_req, bad_resp)
    good_req, good_resp = make_body_response(
        registry, gz(XML), url="http://example.com/good.xml"
    )
    result = await mw.process_response(good_req, good_resp)
    assert Path(result.body_file).read_bytes() == XML
    registry.release(Path(result.body_file))
    assert directory_files(registry) == []


# --------------------------------------------------------------------------- #
# Middleware: close interleaving
# --------------------------------------------------------------------------- #


@coroutine_test
async def test_close_during_decode_drops_result(monkeypatch, tracked_build):
    """close_async() waits for the in-flight decode; its late result is
    released instead of being delivered as a normal response."""
    crawler, _spider, mw, registry = tracked_build()
    barrier = DecodeBarrier(monkeypatch)
    request, response = make_body_response(registry, gz(XML))
    task = asyncio.ensure_future(mw.process_response(request, response))
    try:
        assert await asyncio.to_thread(barrier.entered.wait, 5)
        close_task = asyncio.ensure_future(registry.close_async())
        await loop_ticks()
        # The registry cannot finish closing while the decode slot is open.
        assert not close_task.done()
        assert registry.closing
        barrier.open()
        # The middleware does not hand the late result to the parser.
        with pytest.raises(IgnoreRequest):
            await asyncio.wait_for(task, WATCHDOG)
        await asyncio.wait_for(close_task, WATCHDOG)
    finally:
        barrier.open()
    assert registry.closed
    assert not registry.directory.exists()
    assert crawler.stats.get_value("httpcompression/response_count", 0) == 0


@coroutine_test
async def test_close_waits_for_multiple_decodes(monkeypatch, tracked_build):
    """close_async() accounts for every in-flight decode, not just one: a
    completed-but-undelivered file and a still-running decode are both
    reclaimed before the close finishes."""
    crawler, _spider, mw, registry = tracked_build()
    barrier = DecodeBarrier(monkeypatch)  # parks the first decode only
    req1, resp1 = make_body_response(
        registry, gz(XML), url="http://example.com/s1.xml"
    )
    req2, resp2 = make_body_response(
        registry, gz(XML2), url="http://example.com/s2.xml"
    )
    task1 = asyncio.ensure_future(mw.process_response(req1, resp1))
    try:
        assert await asyncio.to_thread(barrier.entered.wait, 5)
        # A second decode runs to completion while the first is parked.
        result2 = await asyncio.wait_for(mw.process_response(req2, resp2), WATCHDOG)
        assert isinstance(result2, BodyFileResponse)
        close_task = asyncio.ensure_future(registry.close_async())
        await loop_ticks()
        # The parked decode still holds its slot, so the close cannot finish.
        assert not close_task.done()
        barrier.open()
        # The parked decode's late result is dropped, not delivered.
        with pytest.raises(IgnoreRequest):
            await asyncio.wait_for(task1, WATCHDOG)
        await asyncio.wait_for(close_task, WATCHDOG)
    finally:
        barrier.open()
    assert registry.closed
    # The delivered-but-unparsed file of the second response was swept too.
    assert not Path(result2.body_file).exists()
    assert not registry.directory.exists()
    assert crawler.stats.get_value("httpcompression/response_count") == 1


@coroutine_test
async def test_close_then_decode_is_rejected(tracked_build, tmp_path):
    """A decode offered to an already closing registry is refused and its
    body file reclaimed instead of decoded."""
    crawler, _spider, mw, registry = tracked_build()
    # The body file of the late response still exists (the download handler
    # had written it before the close swept the registry directory).
    body_path = tmp_path / "late-body"
    body_path.write_bytes(gz(XML))
    request = Request(
        "http://example.com/sitemap.xml",
        meta={SITEMAP_FILE_BACKED_META: True, "body_file": str(body_path)},
    )
    response = BodyFileResponse(
        request.url,
        headers={"Content-Encoding": b"gzip"},
        body_file=str(body_path),
        request=request,
    )
    await registry.close_async()
    with pytest.raises(IgnoreRequest):
        await mw.process_response(request, response)
    assert not body_path.exists()
    assert crawler.stats.get_value("httpcompression/response_count", 0) == 0


@coroutine_test
async def test_spiders_do_not_affect_each_other(monkeypatch, registries):
    """Closing one spider's registry while another spider decodes leaves the
    other spider fully functional."""
    _crawler_a, _a, _mw_a, registry_a = build_middleware()
    registries.append(registry_a)
    _crawler_b, _b, mw_b, registry_b = build_middleware()
    registries.append(registry_b)
    barrier = DecodeBarrier(monkeypatch)
    request, response = make_body_response(registry_b, gz(XML))
    task = asyncio.ensure_future(mw_b.process_response(request, response))
    try:
        assert await asyncio.to_thread(barrier.entered.wait, 5)
        await asyncio.wait_for(registry_a.close_async(), WATCHDOG)
        assert registry_a.closed
        barrier.open()
        result = await asyncio.wait_for(task, WATCHDOG)
    finally:
        barrier.open()
    assert isinstance(result, BodyFileResponse)
    assert Path(result.body_file).read_bytes() == XML
    registry_b.release(Path(result.body_file))
    await asyncio.wait_for(registry_b.close_async(), WATCHDOG)


@coroutine_test
async def test_user_body_file_response_is_never_managed(tracked_build, tmp_path):
    """A plain body_file response with a user-provided path stays the user's
    responsibility, even around registry close."""
    _crawler, _spider, mw, registry = tracked_build()
    external = tmp_path / "user-body"
    external.write_bytes(b"user bytes")
    request = Request("http://example.com/page", meta={"body_file": str(external)})
    response = BodyFileResponse(
        request.url,
        headers={"Content-Encoding": b"gzip"},
        body_file=str(external),
        request=request,
    )
    assert await mw.process_response(request, response) is response
    assert external.read_bytes() == b"user bytes"
    await registry.close_async()
    assert external.read_bytes() == b"user bytes"


# --------------------------------------------------------------------------- #
# Crawl-level: real download, middleware entry and spider close
# --------------------------------------------------------------------------- #


class _ControllableResource(resource.Resource):
    isLeaf = True

    def __init__(self, handler):
        super().__init__()
        self._handler = handler

    def render(self, request: server.Request) -> Any:
        spec = self._handler(request.uri.decode())
        if isinstance(spec, bytes):
            request.setHeader(b"Content-Type", b"application/xml")
            return spec
        for name, value in spec.get("headers", {}).items():
            request.setHeader(name, value)
        return spec.get("body", b"")


@pytest.fixture
def http_server():
    """Start a controllable HTTP server; return (base_url, set_handler)."""
    from twisted.internet import reactor  # noqa: TID253  # requires a reactor

    state: dict[str, Any] = {"handler": lambda uri: b""}

    def set_handler(handler):
        state["handler"] = handler

    class _Root(resource.Resource):
        isLeaf = True

        def render(self, request: server.Request) -> Any:
            return _ControllableResource(state["handler"]).render(request)

    port: Port = reactor.listenTCP(0, server.Site(_Root()), interface="127.0.0.1")
    base_url = f"http://127.0.0.1:{port.getHost().port}"
    try:
        yield base_url, set_handler
    finally:
        port.stopListening()


@pytest.mark.requires_reactor
@coroutine_test
async def test_fast_stop_during_decode_reclaims_everything(http_server, monkeypatch):
    """A real crawl, stopped fast while the sitemap decode is parked.

    The engine's close waits for the in-flight middleware coroutine (it only
    drops queued and still-downloading requests), so once the barrier opens,
    the decode delivers its response — before the registry starts closing —
    and the spider close then reclaims every file. The page request parsed
    out of the sitemap is rejected because the engine is already closing.
    """
    base_url, set_handler = http_server
    document = urlset_xml(base_url, ["/page-1"])
    set_handler(
        lambda uri: {
            "body": gz(document),
            "headers": {b"Content-Encoding": b"gzip"},
        }
    )
    barrier = DecodeBarrier(monkeypatch)
    registries: list[SitemapFileRegistry] = []

    class _Spider(CancelSpider):
        name = "fast_stop_spider"

        @classmethod
        def from_crawler(cls, crawler, *args, **kwargs):
            spider = super().from_crawler(crawler, *args, **kwargs)
            assert spider._sitemap_file_registry is not None
            registries.append(spider._sitemap_file_registry)
            return spider

        def parse(self, response, **kwargs):
            yield {"url": response.url}

    crawler = get_crawler(_Spider)
    items: list[Any] = []

    def _collect(item, **kwargs):
        # Kept as a local strong reference: pydispatch holds receivers
        # weakly, so a temporary would be garbage-collected mid-crawl.
        items.append(item)

    crawler.signals.connect(_collect, signal=signals.item_scraped)
    crawl_task = asyncio.ensure_future(
        crawler.crawl_async(sitemap_urls=[base_url + "/sitemap.xml"])
    )
    try:
        assert await asyncio.to_thread(barrier.entered.wait, 5)
        assert len(registries) == 1
        registry = registries[0]
        registry_dir = registry.directory
        stop_task = asyncio.ensure_future(
            crawler.engine.close_spider_async(reason="test", mode="fast")
        )
        await loop_ticks()
        # The engine close waits for the in-flight request, so it cannot
        # finish while the decode thread is parked.
        assert not stop_task.done()
        assert not registry.closing
        barrier.open()
        await asyncio.wait_for(stop_task, 5)
        await asyncio.wait_for(crawl_task, 5)
    finally:
        barrier.open()
        if not crawl_task.done():
            crawl_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await crawl_task
    # The decode delivered before the registry started closing (one counted
    # response); the engine rejected the parsed-out page request as it was
    # already closing, so nothing was scraped.
    assert crawler.stats.get_value("httpcompression/response_count", 0) == 1
    assert items == []
    assert crawler.stats.get_value("item_scraped_count", 0) == 0
    # Everything was reclaimed deterministically at the spider close.
    assert registry.pending_decodes == 0
    assert registry.closed
    assert not registry_dir.exists()
    spider = crawler.spider
    assert isinstance(spider, _Spider)
    assert spider._sitemap_file_registry is None


@pytest.mark.requires_reactor
@coroutine_test
async def test_crawl_after_cancel_scenario_succeeds(http_server, monkeypatch):
    """A crawl where one sitemap decode is cancelled (via fast stop) does
    not affect a subsequent, independent crawl on a fresh crawler."""
    base_url, set_handler = http_server
    document = urlset_xml(base_url, ["/page-1"])
    set_handler(lambda uri: document)

    class _Spider(CancelSpider):
        name = "reuse_spider"

        def parse(self, response, **kwargs):
            yield {"url": response.url}

    crawler = get_crawler(_Spider)
    items: list[Any] = []

    def _collect(item, **kwargs):
        items.append(item)

    crawler.signals.connect(_collect, signal=signals.item_scraped)
    await crawler.crawl_async(sitemap_urls=[base_url + "/sitemap.xml"])
    assert [item["url"] for item in items] == [base_url + "/page-1"]
