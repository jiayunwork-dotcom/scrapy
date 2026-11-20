from __future__ import annotations

import asyncio
import contextlib
import warnings
from itertools import chain
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

from scrapy import Request, Spider, signals
from scrapy.exceptions import (
    DecompressionError,
    IgnoreRequest,
    NotConfigured,
    ScrapyDeprecationWarning,
)
from scrapy.http import BodyFileResponse, Response, TextResponse
from scrapy.responsetypes import responsetypes
from scrapy.spiders._sitemap_files import SITEMAP_FILE_BACKED_META, RegistryClosedError
from scrapy.utils._compression import (
    _DECOMPRESSION_ERRORS,
    _DecompressionMaxSizeExceeded,
    _inflate,
    _unbrotli,
    _unzstd,
    gunzip,
    gunzip_file,
    inflate_file,
)
from scrapy.utils._download_handlers import get_warnsize
from scrapy.utils.asyncio import is_asyncio_available, run_in_thread
from scrapy.utils.decorators import _warn_spider_arg
from scrapy.utils.deprecate import warn_on_deprecated_spider_attribute

if TYPE_CHECKING:
    from collections.abc import Callable

    from scrapy.crawler import Crawler
    from scrapy.spiders._sitemap_files import SitemapFileRegistry, _DecodeSlot
    from scrapy.statscollectors import StatsCollector


logger = getLogger(__name__)

ACCEPTED_ENCODINGS: list[bytes] = [b"gzip", b"deflate", b"br", b"zstd"]

# Only gzip and deflate may be negotiated for file-backed sitemap downloads:
# their decoders support streaming from one file into another, while br and
# zstd decoding is only available in-memory.
FILE_BACKED_ACCEPTED_ENCODINGS: list[bytes] = [b"gzip", b"deflate"]

# Dispatching to a thread costs a few hundred microseconds. That is
# negligible next to network latency, but wasted when there is no other
# in-flight request for the reactor to get back to; above this compressed
# size, decoding concurrently with other requests outweighs that cost.
# Tuned by tests/benchmarks/test_httpcompression.py.
_THREAD_DECODE_MIN_SIZE = 8192


class _MaxEncodingCountExceeded(ValueError):
    def __init__(self, count: int, max_count: int) -> None:
        self.count = count
        self.max_count = max_count


class HttpCompressionMiddleware:
    """This middleware allows compressed (gzip, deflate etc.) traffic to be
    sent/received from websites"""

    def __init__(
        self,
        stats: StatsCollector | None = None,
        *,
        crawler: Crawler | None = None,
    ):
        if not crawler:
            warnings.warn(
                "Instantiating HttpCompressionMiddleware without a 'crawler' "
                "argument is deprecated.",
                category=ScrapyDeprecationWarning,
                stacklevel=2,
            )
            self.stats = stats
            self._crawler = None
            self._max_size = 1073741824
            self._warn_size = 33554432
            self._max_encodings = 1
            return
        self.stats = crawler.stats
        self._crawler = crawler
        self._max_size = crawler.settings.getint("DOWNLOAD_MAXSIZE")
        self._warn_size = crawler.settings.getint("DOWNLOAD_WARNSIZE")
        self._max_encodings = crawler.settings.getint("COMPRESSION_MAX_ENCODINGS")
        crawler.signals.connect(self.open_spider, signals.spider_opened)

    @classmethod
    def from_crawler(cls, crawler: Crawler) -> Self:
        if not crawler.settings.getbool("COMPRESSION_ENABLED"):
            raise NotConfigured
        return cls(crawler=crawler)

    def open_spider(self, spider: Spider) -> None:
        if hasattr(spider, "download_maxsize"):
            warn_on_deprecated_spider_attribute("download_maxsize", "DOWNLOAD_MAXSIZE")
            self._max_size = spider.download_maxsize
        if hasattr(spider, "download_warnsize"):
            warn_on_deprecated_spider_attribute(
                "download_warnsize", "DOWNLOAD_WARNSIZE"
            )
            self._warn_size = spider.download_warnsize

    @_warn_spider_arg
    def process_request(
        self, request: Request, spider: Spider | None = None
    ) -> Request | Response | None:
        if request.meta.get(SITEMAP_FILE_BACKED_META):
            # File-backed sitemap requests are written to a file; their
            # Content-Encoding is handled file-to-file in process_response().
            # Only gzip and deflate can be streamed to a file.
            request.headers.setdefault(
                "Accept-Encoding", b", ".join(FILE_BACKED_ACCEPTED_ENCODINGS)
            )
            return None
        if request.meta.get("body_file"):
            # Their body is written to a file, never decompressed.
            return None
        request.headers.setdefault("Accept-Encoding", b", ".join(ACCEPTED_ENCODINGS))
        return None

    @_warn_spider_arg
    async def process_response(
        self, request: Request, response: Response, spider: Spider | None = None
    ) -> Request | Response:
        if request.method == "HEAD":
            return response
        if request.meta.get(SITEMAP_FILE_BACKED_META):
            return await self._process_file_backed(request, response, spider)
        if isinstance(response, BodyFileResponse):
            return response
        content_encoding = response.headers.getlist("Content-Encoding")
        if not content_encoding:
            return response
        max_size = request.meta.get("download_maxsize", self._max_size)
        warn_size = get_warnsize(request.meta, self._warn_size)
        try:
            decoded_body, content_encoding = await self._handle_encoding(
                response.body, content_encoding, max_size
            )
        except _DecompressionMaxSizeExceeded as e:
            msg = (
                f"Ignored response {response} because its body "
                f"({len(response.body)} B compressed, "
                f"{e.decompressed_size} B decompressed so far) exceeded "
                f"DOWNLOAD_MAXSIZE ({max_size} B) during decompression."
            )
            logger.warning(msg)
            raise IgnoreRequest(msg) from e
        except _MaxEncodingCountExceeded as e:
            msg = (
                f"Ignored response {response} because its Content-Encoding "
                f"header lists {e.count} supported encodings, which exceeds "
                f"COMPRESSION_MAX_ENCODINGS ({e.max_count})."
            )
            logger.warning(msg)
            raise IgnoreRequest(msg) from e
        except _DECOMPRESSION_ERRORS as e:
            encodings = b", ".join(content_encoding).decode()
            raise DecompressionError(
                f"Could not decompress {response} ({encodings}): {e}"
            ) from e
        if len(response.body) < warn_size <= len(decoded_body):
            logger.warning(
                f"{response} body size after decompression "
                f"({len(decoded_body)} B) is larger than the "
                f"download warning size ({warn_size} B)."
            )
        if content_encoding:
            self._warn_unknown_encoding(response, content_encoding)
        response.headers["Content-Encoding"] = content_encoding
        if self.stats:
            self.stats.inc_value(
                "httpcompression/response_bytes",
                len(decoded_body),
            )
            self.stats.inc_value("httpcompression/response_count")
        respcls = responsetypes.from_args(
            headers=response.headers, url=response.url, body=decoded_body
        )
        kwargs: dict[str, Any] = {"body": decoded_body}
        if issubclass(respcls, TextResponse):
            # force recalculating the encoding until we make sure the
            # responsetypes guessing is reliable
            kwargs["encoding"] = None
        response = response.replace(cls=respcls, **kwargs)
        if not content_encoding:
            del response.headers["Content-Encoding"]
        return response

    async def _process_file_backed(
        self, request: Request, response: Response, spider: Spider | None
    ) -> BodyFileResponse:
        if not isinstance(response, BodyFileResponse):
            # A download handler that does not support body_file was used: do
            # not silently fall back to an in-memory response.
            raise NotConfigured(
                f"The download handler used for {request} does not support "
                f"file-backed sitemap downloads."
            )
        # The spider argument of process_response() is deprecated and passed as
        # None; the crawler always knows the spider driving the request.
        current_spider = spider or (
            self._crawler.spider if self._crawler is not None else None
        )
        assert current_spider is not None
        return await self._handle_file_backed_response(
            request, response, current_spider
        )

    async def _handle_file_backed_response(
        self, request: Request, response: BodyFileResponse, spider: Spider
    ) -> BodyFileResponse:
        """Decode the HTTP Content-Encoding of a file-backed sitemap response
        file-to-file, never loading the compressed or the decoded body into
        memory.

        The response keeps being a :class:`~scrapy.http.BodyFileResponse`; only
        the file its ``body_file`` points at changes as layers are stripped.
        Each intermediate file is deleted once the next layer has been written.
        All blocking file I/O runs in a worker thread.

        The decode is registered with the spider's
        :class:`~scrapy.spiders._sitemap_files.SitemapFileRegistry` for as
        long as it may create files, so that the registry close never treats
        an in-flight decode as already reclaimed. If this coroutine is
        cancelled, the cancellation propagates unchanged and the decode
        output is reclaimed in the background once the worker thread
        finishes; if the spider starts closing while a decode runs, its
        result is released instead of being delivered.
        """
        content_encoding = response.headers.getlist("Content-Encoding")
        if not content_encoding:
            return response
        # An empty body with a Content-Encoding header has nothing to decode,
        # mirroring the in-memory middleware path.
        if await run_in_thread(_file_size, Path(response.body_file)) == 0:
            del response.headers["Content-Encoding"]
            return response
        max_size = request.meta.get("download_maxsize", self._max_size)
        warn_size = get_warnsize(request.meta, self._warn_size)
        to_decode, to_keep = self._split_encodings(content_encoding)
        if to_keep:
            # An encoding we cannot stream file-to-file (br, zstd or anything
            # unknown). Failing explicitly keeps compressed bytes from being
            # fed to the sitemap XML parser as an empty map.
            encodings = b", ".join(to_keep).decode()
            msg = (
                f"Could not decompress {response}: unsupported HTTP "
                f"Content-Encoding for file-backed sitemap downloads: "
                f"{encodings}"
            )
            logger.warning(msg)
            raise IgnoreRequest(msg)
        if len(to_decode) > self._max_encodings:
            e = _MaxEncodingCountExceeded(len(to_decode), self._max_encodings)
            msg = (
                f"Ignored response {response} because its Content-Encoding "
                f"header lists {e.count} supported encodings, which exceeds "
                f"COMPRESSION_MAX_ENCODINGS ({e.max_count})."
            )
            logger.warning(msg)
            raise IgnoreRequest(msg) from e

        registry: SitemapFileRegistry | None = getattr(
            spider, "_sitemap_file_registry", None
        )
        assert registry is not None
        final_path, layer_sizes = await self._decode_sitemap_file(
            response, registry, content_encoding, to_decode, max_size
        )

        if registry.closing:
            # The spider started closing while the decode was running: the
            # decoded file must not be delivered as a normal response, so it
            # is released instead of being handed to the parser.
            registry.release(final_path)
            raise IgnoreRequest(
                f"Ignored response {response} because the spider is closing."
            )

        for decoded_size in layer_sizes:
            if self.stats:
                self.stats.inc_value("httpcompression/response_bytes", decoded_size)
                self.stats.inc_value("httpcompression/response_count")
            if warn_size and decoded_size >= warn_size:
                logger.warning(
                    f"{response} body size after decompression "
                    f"({decoded_size} B) is larger than or equal to the "
                    f"download warning size ({warn_size} B)."
                )
        response.headers["Content-Encoding"] = to_keep
        if not to_keep:
            del response.headers["Content-Encoding"]
        return response.replace(body_file=final_path)

    async def _decode_sitemap_file(
        self,
        response: BodyFileResponse,
        registry: SitemapFileRegistry,
        content_encoding: list[bytes],
        to_decode: list[bytes],
        max_size: int,
    ) -> tuple[Path, list[int]]:
        """Run the file-to-file decode of *response*'s body file under a
        registry decode slot, translating worker-thread failures into the
        middleware's usual exceptions."""
        body_path = Path(response.body_file)
        try:
            slot = registry.begin_decode()
        except RegistryClosedError as e:
            # The spider is closing and its temporary files are being
            # reclaimed; this response cannot be delivered anymore, so its
            # body file goes back to the registry instead of the parser.
            registry.release(body_path)
            raise IgnoreRequest(
                f"Ignored response {response} because the sitemap file "
                f"registry is closed."
            ) from e
        try:
            if is_asyncio_available():
                return await self._decode_file_backed(
                    registry, slot, body_path, to_decode, max_size
                )
            # Deprecated non-asyncio reactor: the thread wait cannot be
            # shielded, so a cancellation here abandons the decode output to
            # the registry sweep on close.
            try:
                return await run_in_thread(
                    _decode_sitemap_file_layers,
                    body_path,
                    to_decode,
                    registry,
                    max_size,
                )
            finally:
                slot.close()
        except _DecompressionMaxSizeExceeded as e:
            msg = (
                f"Ignored response {response} because its decompressed body "
                f"({e.decompressed_size} B so far) exceeded DOWNLOAD_MAXSIZE "
                f"({max_size} B) during file-backed decompression."
            )
            logger.warning(msg)
            raise IgnoreRequest(msg) from e
        except _DECOMPRESSION_ERRORS as e:
            encodings = b", ".join(content_encoding).decode()
            raise DecompressionError(
                f"Could not decompress {response} ({encodings}): {e}"
            ) from e
        except RegistryClosedError as e:
            # The registry was synchronously closed while the decode thread
            # was running; the thread already removed its files.
            raise IgnoreRequest(
                f"Ignored response {response} because the sitemap file "
                f"registry was closed during decompression."
            ) from e
        except OSError as e:
            raise DecompressionError(f"Could not decompress {response}: {e}") from e

    async def _decode_file_backed(
        self,
        registry: SitemapFileRegistry,
        slot: _DecodeSlot,
        body_path: Path,
        to_decode: list[bytes],
        max_size: int,
    ) -> tuple[Path, list[int]]:
        """Await the file-to-file decode of *body_path*, shielded from
        cancellation.

        Cancelling this coroutine cannot stop the worker thread, so the
        decode runs on a shielded task whose handle survives the
        cancellation: a done callback then reclaims whatever the thread
        still produces, settles the decode slot, and retrieves any thread
        exception. The cancellation itself still propagates to the caller,
        and neither a success response nor decompression stats are produced
        for it.
        """
        decode_task = asyncio.ensure_future(
            run_in_thread(
                _decode_sitemap_file_layers,
                body_path,
                to_decode,
                registry,
                max_size,
            )
        )
        disposed = False
        try:
            return await asyncio.shield(decode_task)
        except asyncio.CancelledError:
            decode_task.add_done_callback(_dispose_cancelled_decode(slot, registry))
            # The slot now belongs to the done callback.
            disposed = True
            raise
        finally:
            if not disposed:
                slot.close()

    async def _handle_encoding(
        self, body: bytes, content_encoding: list[bytes], max_size: int
    ) -> tuple[bytes, list[bytes]]:
        to_decode, to_keep = self._split_encodings(content_encoding)
        if len(to_decode) > self._max_encodings:
            raise _MaxEncodingCountExceeded(len(to_decode), self._max_encodings)
        for encoding in to_decode:
            if len(body) >= _THREAD_DECODE_MIN_SIZE:
                body = await run_in_thread(self._decode, body, encoding, max_size)
            else:
                body = self._decode(body, encoding, max_size)
        return body, to_keep

    @staticmethod
    def _split_encodings(
        content_encoding: list[bytes],
    ) -> tuple[list[bytes], list[bytes]]:
        supported_encodings = {*ACCEPTED_ENCODINGS, b"x-gzip"}
        to_keep: list[bytes] = [
            encoding.strip().lower()
            for encoding in chain.from_iterable(
                encodings.split(b",") for encodings in content_encoding
            )
        ]
        to_decode: list[bytes] = []
        while to_keep:
            encoding = to_keep.pop()
            if encoding not in supported_encodings:
                to_keep.append(encoding)
                return to_decode, to_keep
            to_decode.append(encoding)
        return to_decode, to_keep

    @staticmethod
    def _decode(body: bytes, encoding: bytes, max_size: int) -> bytes:
        if encoding in {b"gzip", b"x-gzip"}:
            return gunzip(body, max_size=max_size)
        if encoding == b"deflate":
            return _inflate(body, max_size=max_size)
        if encoding == b"br":
            return _unbrotli(body, max_size=max_size)
        if encoding == b"zstd":
            return _unzstd(body, max_size=max_size)
        # shouldn't be reached
        return body  # pragma: no cover

    def _warn_unknown_encoding(
        self, response: Response, encodings: list[bytes]
    ) -> None:
        encodings_str = b",".join(encodings).decode()
        logger.warning(
            f"{self.__class__.__name__} cannot decode the response for {response.url} "
            f"from unsupported encoding(s) '{encodings_str}'."
        )


def _file_size(path: Path) -> int:
    return path.stat().st_size


def _dispose_cancelled_decode(
    slot: _DecodeSlot, registry: SitemapFileRegistry
) -> Callable[[asyncio.Future[tuple[Path, list[int]]]], None]:
    """Return a done callback reclaiming the output of a decode whose
    awaiting coroutine was cancelled.

    The worker thread keeps running after the cancellation; once it
    finishes, this releases the decoded file it produced (the thread itself
    already removed its files on failure), retrieves any thread exception so
    it is never reported as unretrieved, and settles the decode slot so the
    registry close can finish. Cleanup errors are logged, never raised:
    there is no caller left to report them to, and they must not mask the
    cancellation.
    """

    def dispose(decode_task: asyncio.Future[tuple[Path, list[int]]]) -> None:
        try:
            try:
                final_path, _layer_sizes = decode_task.result()
            except asyncio.CancelledError:
                logger.debug(
                    "The decode task of a cancelled file-backed sitemap "
                    "download was itself cancelled."
                )
                return
            except Exception:
                logger.exception(
                    "Decompression of a cancelled file-backed sitemap "
                    "download failed; its temporary files were removed."
                )
                return
            try:
                registry.release(final_path)
            except Exception:
                logger.exception(
                    "Could not remove the decoded file %s left by a "
                    "cancelled file-backed sitemap download.",
                    final_path,
                )
        finally:
            slot.close()

    return dispose


def _decode_sitemap_file_layers(
    body_path: Path,
    encodings: list[bytes],
    registry: SitemapFileRegistry | None,
    max_size: int,
) -> tuple[Path, list[int]]:
    """Stream-decode *encodings* of the sitemap at *body_path* file-to-file.

    Each layer is decoded from one temporary file into the next one in the
    registry directory; the source file of a layer is deleted once the next
    layer is written. Return the path of the final decoded file and the number
    of bytes produced by each layer, in decoding order.

    On failure, the partial output and the files still held are removed, so
    failed attempts never leave files behind. Decompression and size-limit
    exceptions propagate unchanged to the caller, as does
    :exc:`~scrapy.spiders._sitemap_files.RegistryClosedError` if the registry
    was swept while decoding (its files are gone by then, and the removal
    above degrades to a no-op).
    """

    def remove(path: Path) -> None:
        if registry is not None:
            registry.release(path)
        else:  # pragma: no cover - the spider always passes a registry
            with contextlib.suppress(OSError):
                path.unlink()

    current_path = body_path
    next_path: Path | None = None
    layer_sizes: list[int] = []
    try:
        for encoding in encodings:
            next_path = current_path.with_name(current_path.name + ".dec")
            with (
                current_path.open("rb") as input_file,
                next_path.open("wb") as output_file,
            ):
                if encoding in {b"gzip", b"x-gzip"}:
                    decoded_size = gunzip_file(
                        input_file, output_file, max_size=max_size
                    )
                else:  # b"deflate", the only other encoding negotiated
                    decoded_size = inflate_file(
                        input_file, output_file, max_size=max_size
                    )
            if registry is not None:
                registry.track(next_path)
            # The layer just decoded is no longer needed.
            remove(current_path)
            layer_sizes.append(decoded_size)
            current_path = next_path
            next_path = None
    except BaseException:
        # A failed attempt must leave no temporary files behind.
        if next_path is not None:
            remove(next_path)
        remove(current_path)
        raise
    return current_path, layer_sizes
