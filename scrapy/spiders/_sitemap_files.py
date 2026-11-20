"""Support machinery for the file-backed mode of
:class:`~scrapy.spiders.sitemap.SitemapSpider`.

When :setting:`SITEMAP_FILE_BACKED` is enabled, the body of every sitemap
download is written to a Scrapy-managed temporary file instead of being kept
in memory. :class:`SitemapFileRegistry` owns the temporary directory and
tracks the files backing sitemap downloads, so that they are reclaimed
deterministically when their sitemap response is done with them, when a
download or a decompression fails, or when the spider closes, without relying
on the garbage collector or on process exit.

The downloader middleware allocates the files (a fresh one for every download
attempt, including redirects and retries) through :meth:`track` and removes
the ones it no longer needs through :meth:`release`. The spider releases the
file of each sitemap once it has finished parsing it. Files still around when
the spider closes are removed by :meth:`close_async`, which also removes the
temporary directory.

Ownership contract
==================

Every file in the registry directory has exactly one owner at any moment:

*   the download handler owns the body file until the response reaches the
    HTTP compression middleware,
*   the middleware owns it (and every decoded layer file it creates) until it
    either delivers the response to the spider or drops it,
*   the spider owns the decoded file of a delivered response until it has
    finished parsing it.

A file whose owner drops it (a failed download, a failed decode, a cancelled
or undeliverable response) is released by that owner; the registry reclaims
anything still held when the spider closes.

Decompression runs in a worker thread, and cancelling the coroutine that
waits for it cannot stop the thread. To keep the ownership chain unbroken
across that race, the middleware registers every decode with
:meth:`begin_decode` before starting its thread and settles the returned slot
exactly once: when the decode result has been accounted for (delivered,
released after a failure, or reclaimed by the cancellation disposal
callback). :meth:`close_async` does not consider reclamation finished while
decode slots are open, and once closing has started :meth:`begin_decode`
refuses new decodes. Once the registry has been swept, :meth:`track` refuses
late registrations with :exc:`RegistryClosedError` instead of accepting files
nobody would ever reclaim, so a caller holding a late file learns that the
registry did not take it over.

Files requested explicitly through the ``body_file`` request meta key with a
user-provided path never go through this registry and are never deleted here.
"""

from __future__ import annotations

import contextlib
import logging
import tempfile
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from scrapy.utils.asyncio import run_in_thread

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)

# Request meta key marking a request as a file-backed sitemap request. It is
# private: the spider sets it on the requests it creates for sitemaps, and the
# downloader middleware stack reads it. It must never be copied onto the
# requests built for the final pages discovered in the sitemaps.
SITEMAP_FILE_BACKED_META = "_sitemap_file_backed"

# Private request meta key naming the directory in which the download handler
# must create the Scrapy-managed temporary file of a file-backed sitemap
# request. Only set together with body_file=True and the marker above.
SITEMAP_FILE_DIR_META = "_sitemap_file_dir"


class RegistryClosedError(RuntimeError):
    """A file or a decode operation was offered to a
    :class:`SitemapFileRegistry` that is closing or has already been swept.

    The registry did not take over the resource: the caller keeps (and must
    dispose of) whatever it offered.
    """


@contextlib.contextmanager
def _suppress_oserror() -> Iterator[None]:
    try:
        yield
    except OSError as exc:
        logger.debug("Ignoring error while removing a temporary file: %s", exc)


class _DecodeSlot:
    """Token for one in-flight file-to-file decode operation.

    Settled exactly once through :meth:`close`; closing twice is a no-op, so
    the middleware and its cancellation disposal callback never have to race
    for it. The registry is notified on the first close.
    """

    __slots__ = ("_open", "_registry")

    def __init__(self, registry: SitemapFileRegistry) -> None:
        self._registry = registry
        self._open = True

    def close(self) -> None:
        if not self._open:
            return
        self._open = False
        self._registry._end_decode()


class SitemapFileRegistry:
    """Owns the temporary directory and tracks the files created in it.

    The methods of this class are called both from the event-loop thread
    (:meth:`release`, :meth:`begin_decode`, :meth:`close_async`) and from
    decompression worker threads (:meth:`track`, :meth:`release`); its mutable
    state is guarded by one condition lock that is only ever held for
    bookkeeping, never across file I/O or a whole decode.
    """

    def __init__(self, prefix: str = "scrapy-sitemap-") -> None:
        self._tempdir = tempfile.TemporaryDirectory(prefix=prefix)
        self._condition = threading.Condition()
        self._paths: set[Path] = set()
        self._pending_decodes = 0
        # Closing has started: no new decode operations are accepted, but
        # in-flight ones may still register their output (it is reclaimed by
        # their owner or by the final sweep).
        self._closing = False
        # The directory has been swept: no new files are accepted at all.
        self._swept = False
        self._swept_event = threading.Event()

    @property
    def directory(self) -> Path:
        """Path of the directory holding the temporary files."""
        return Path(self._tempdir.name)

    @property
    def closing(self) -> bool:
        """Whether closing has started (no new decode operations accepted)."""
        return self._closing

    @property
    def closed(self) -> bool:
        """Whether the registry has been swept and its directory removed."""
        return self._swept

    @property
    def pending_decodes(self) -> int:
        """Number of decode operations that may still create files here."""
        return self._pending_decodes

    def track(self, path: str | Path) -> Path:
        """Start tracking *path*.

        The file is expected to be created by the caller, inside
        :attr:`directory`. Tracking it guarantees its reclamation even if the
        caller crashes before releasing it.

        Raise :exc:`RegistryClosedError` if the registry has already been
        swept: the file is *not* taken over and stays the caller's
        responsibility.
        """
        path = Path(path)
        with self._condition:
            if self._swept:
                raise RegistryClosedError(
                    f"The sitemap file registry is closed and does not take "
                    f"over {path}; the caller keeps owning it."
                )
            self._paths.add(path)
        return path

    def release(self, path: str | Path) -> None:
        """Delete *path* and stop tracking it.

        A missing file is not an error: a previous cleanup may already have
        removed it. Safe to call after the registry was closed.
        """
        path = Path(path)
        with self._condition:
            self._paths.discard(path)
        with _suppress_oserror():
            path.unlink()

    def begin_decode(self) -> _DecodeSlot:
        """Register a decode operation that may create files in this
        registry, and return its slot.

        The caller must settle the returned slot exactly once, when the
        decode's outcome has been accounted for (its files delivered or
        removed). :meth:`close_async` waits for all open slots before
        finishing, so a decode in progress is never treated as already
        reclaimed.

        Raise :exc:`RegistryClosedError` if the registry is closing or
        closed.
        """
        with self._condition:
            if self._closing:
                raise RegistryClosedError(
                    "The sitemap file registry is closing and accepts no new "
                    "decode operations."
                )
            self._pending_decodes += 1
        return _DecodeSlot(self)

    def _end_decode(self) -> None:
        with self._condition:
            self._pending_decodes -= 1
            if not self._pending_decodes:
                self._condition.notify_all()

    def _wait_for_decodes(self) -> None:
        """Block until no decode operations are in flight.

        Runs in a worker thread; never call it on the event-loop thread.
        """
        with self._condition:
            while self._pending_decodes:
                self._condition.wait()

    def _begin_close(self) -> bool:
        """Mark the registry as closing; return False if it already was."""
        with self._condition:
            if self._closing:
                return False
            self._closing = True
            return True

    def _sweep(self) -> None:
        """Delete every file still tracked and remove the directory."""
        with self._condition:
            paths = tuple(self._paths)
            self._paths.clear()
        for path in paths:
            with _suppress_oserror():
                path.unlink()
        self._tempdir.cleanup()
        with self._condition:
            self._swept = True
        self._swept_event.set()

    def close(self) -> None:
        """Best-effort synchronous close: reject new decodes and files, then
        delete every file still tracked and remove the temporary directory.

        Safe to call more than once. Unlike :meth:`close_async`, this does
        not wait for in-flight decode operations: their late output is
        rejected by :meth:`track` and removed by the decode itself. Use it
        only where no event loop is available; the spider closes through
        :meth:`close_async`.
        """
        if self._begin_close():
            self._sweep()

    async def close_async(self) -> None:
        """Close the registry, waiting for in-flight decode operations first.

        No new decode operations or files are accepted once this starts.
        Decode operations already in progress finish (their output is
        reclaimed by their owner), then every file still tracked is deleted
        and the temporary directory is removed. Safe to call more than once
        and concurrently: later calls simply wait for the first one to
        finish.

        If this coroutine is itself cancelled while waiting, it degrades to
        the synchronous :meth:`close` behavior (sweep immediately) rather
        than leaving the registry closing-but-never-swept.
        """
        if self._begin_close():
            try:
                with self._condition:
                    pending = self._pending_decodes
                if pending:
                    # The wait parks a worker thread instead of the event
                    # loop; slot holders settle their decodes on the loop
                    # meanwhile.
                    await run_in_thread(self._wait_for_decodes)
            finally:
                self._sweep()
        elif not self._swept_event.is_set():
            await run_in_thread(self._swept_event.wait)
