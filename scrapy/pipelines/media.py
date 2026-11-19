from __future__ import annotations

import functools
import logging
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Literal, NoReturn, TypedDict, Union, cast

from twisted import version as twisted_version
from twisted.internet.defer import Deferred, DeferredList
from twisted.python.failure import Failure
from twisted.python.versions import Version

from scrapy.exceptions import ScrapyDeprecationWarning
from scrapy.http.request import NO_CALLBACK, Request
from scrapy.settings import Settings
from scrapy.utils.datatypes import SequenceExclude
from scrapy.utils.defer import defer_result, mustbe_deferred
from scrapy.utils.log import failure_to_exc_info
from scrapy.utils.misc import arg_to_iter
from scrapy.utils.python import get_func_args, global_object_name

if TYPE_CHECKING:
    from collections.abc import Callable

    # typing.Self requires Python 3.11
    from typing_extensions import Self

    from scrapy import Spider
    from scrapy.crawler import Crawler
    from scrapy.http import Response
    from scrapy.utils.request import RequestFingerprinterProtocol


class FileInfo(TypedDict):
    url: str
    path: str
    checksum: str | None
    status: str


FileInfoOrError = Union[tuple[Literal[True], FileInfo], tuple[Literal[False], Failure]]

logger = logging.getLogger(__name__)


class _DownloadFanout:
    """Multicast a single download result to several target Deferreds.

    Targets sharing one request fingerprint reuse the same download, but each
    target processes the Response independently to produce its own files. A
    plain callback chain cannot do this (each stage would receive the previous
    stage's output instead of the download Response), so every fork receives
    an independent copy of the original result.
    """

    def __init__(self, download: Deferred[Response]):
        self.ready: Deferred[Response] = download
        # Consume a failed download on the shared chain itself; the Failure is
        # still copied to every fork, which carry the per-target error handling.
        download.addErrback(lambda failure: failure)
        download.addBoth(self._fire)
        self._forks: list[Deferred[Response]] = []
        self._result: Response | Failure | None = None

    def _fire(self, result: Response | Failure) -> None:
        self._result = result
        for fork in self._forks:
            if isinstance(result, Failure):
                fork.errback(result)
            else:
                fork.callback(result)
        self._forks.clear()

    def fork(self) -> Deferred[Response]:
        fork: Deferred[Response] = Deferred()
        if self._result is None:
            self._forks.append(fork)
            return fork
        # The download already finished, e.g. a synchronous download function:
        # give the fork an observer before firing it, so a Failure is not
        # reported as unhandled before the caller attaches its errbacks.
        fork.addBoth(lambda result: result)
        if isinstance(self._result, Failure):
            fork.errback(self._result)
        else:
            fork.callback(self._result)
        return fork


class MediaPipeline(ABC):
    crawler: Crawler
    _fingerprinter: RequestFingerprinterProtocol
    _modern_init = False

    LOG_FAILED_RESULTS: bool = True

    class SpiderInfo:
        def __init__(self, spider: Spider):
            self.spider: Spider = spider
            # fingerprints of requests whose download is in progress. The same
            # download is never started twice concurrently.
            self.downloading: set[bytes] = set()
            # fan-outs of in-progress downloads, keyed by request fingerprint
            self.downloads: dict[bytes, _DownloadFanout] = {}
            # cached terminal results, keyed by target key (see _get_target_key())
            self.downloaded: dict[Any, FileInfo | Failure] = {}
            # one deferred per item waiting for a given target result
            self.waiting: defaultdict[Any, list[Deferred[FileInfo]]] = defaultdict(
                list
            )

    def __init__(
        self,
        download_func: Callable[[Request, Spider], Response] | None = None,
        settings: Settings | dict[str, Any] | None = None,
        *,
        crawler: Crawler | None = None,
    ):
        self.download_func = download_func

        if crawler is not None:
            if settings is not None:
                warnings.warn(
                    f"MediaPipeline.__init__() was called with a crawler instance and a settings instance"
                    f" when creating {global_object_name(self.__class__)}. The settings instance will be ignored"
                    f" and crawler.settings will be used. The settings argument will be removed in a future Scrapy version.",
                    category=ScrapyDeprecationWarning,
                    stacklevel=2,
                )
            settings = crawler.settings
        elif isinstance(settings, dict) or settings is None:
            settings = Settings(settings)
        resolve = functools.partial(
            self._key_for_pipe, base_class_name="MediaPipeline", settings=settings
        )
        self.allow_redirects: bool = settings.getbool(
            resolve("MEDIA_ALLOW_REDIRECTS"), False
        )
        self._handle_statuses(self.allow_redirects)

        if crawler:
            self._finish_init(crawler)
            self._modern_init = True
        else:
            warnings.warn(
                f"MediaPipeline.__init__() was called without the crawler argument"
                f" when creating {global_object_name(self.__class__)}."
                f" This is deprecated and the argument will be required in future Scrapy versions.",
                category=ScrapyDeprecationWarning,
                stacklevel=2,
            )

    def _finish_init(self, crawler: Crawler) -> None:
        # This was done in from_crawler() before 2.12, now it's done in __init__()
        # if the crawler was passed to it and may be needed to be called in other
        # deprecated code paths explicitly too. After the crawler argument of __init__()
        # becomes mandatory this should be inlined there.
        self.crawler = crawler
        assert crawler.request_fingerprinter
        self._fingerprinter = crawler.request_fingerprinter

    def _handle_statuses(self, allow_redirects: bool) -> None:
        self.handle_httpstatus_list = None
        if allow_redirects:
            self.handle_httpstatus_list = SequenceExclude(range(300, 400))

    def _key_for_pipe(
        self,
        key: str,
        base_class_name: str | None = None,
        settings: Settings | None = None,
    ) -> str:
        class_name = self.__class__.__name__
        formatted_key = f"{class_name.upper()}_{key}"
        if (
            not base_class_name
            or class_name == base_class_name
            or (settings and not settings.get(formatted_key))
        ):
            return key
        return formatted_key

    @classmethod
    def from_crawler(cls, crawler: Crawler) -> Self:
        pipe: Self
        if hasattr(cls, "from_settings"):
            pipe = cls.from_settings(crawler.settings)  # type: ignore[attr-defined]
            warnings.warn(
                f"{global_object_name(cls)} has from_settings() and either doesn't have"
                " from_crawler() or calls MediaPipeline.from_crawler() from it,"
                " so from_settings() was used to create the instance of it."
                " This is deprecated and calling from_settings() will be removed"
                " in a future Scrapy version. Please move the initialization code into"
                " from_crawler() or __init__().",
                category=ScrapyDeprecationWarning,
            )
        elif "crawler" in get_func_args(cls.__init__):
            pipe = cls(crawler=crawler)
        else:
            pipe = cls()
            warnings.warn(
                f"{global_object_name(cls)}.__init__() doesn't take a crawler argument."
                " This is deprecated and the argument will be required in future Scrapy versions.",
                category=ScrapyDeprecationWarning,
            )
        if not pipe._modern_init:
            pipe._finish_init(crawler)
        return pipe

    def open_spider(self, spider: Spider) -> None:
        self.spiderinfo = self.SpiderInfo(spider)

    def process_item(
        self, item: Any, spider: Spider
    ) -> Deferred[list[FileInfoOrError]]:
        info = self.spiderinfo
        requests = arg_to_iter(self.get_media_requests(item, info))
        dlist = [self._process_request(r, info, item) for r in requests]
        dfd = cast(
            "Deferred[list[FileInfoOrError]]", DeferredList(dlist, consumeErrors=True)
        )
        return dfd.addCallback(self.item_completed, item, info)

    def _process_request(
        self, request: Request, info: SpiderInfo, item: Any
    ) -> Deferred[FileInfo]:
        fp = self._fingerprinter.fingerprint(request)
        eb = request.errback
        request.callback = NO_CALLBACK
        request.errback = None

        target = self._get_target_key(request, info, item, fp)

        # Return cached result if this target was already processed.
        # Results are cached per target, so items sharing a request but
        # producing different files never receive another item's result.
        if target in info.downloaded:
            d = defer_result(info.downloaded[target])
            if eb:
                d.addErrback(eb)
            return d

        # Otherwise, wait for this target's result
        wad: Deferred[FileInfo] = Deferred()
        if eb:
            wad.addErrback(eb)

        # Check if this target is being processed right now to avoid doing it
        # twice. Waiting entries exist exactly while a target is in progress;
        # they are popped when its result is cached.
        target_waiters = info.waiting.get(target)
        if target_waiters is not None:
            target_waiters.append(wad)
            return wad
        info.waiting[target].append(wad)

        # The first request for this target drives its processing.
        dfd: Deferred[FileInfo | None] = mustbe_deferred(
            self.media_to_download, request, info, item=item
        )
        dfd2: Deferred[FileInfo] = dfd.addCallback(
            self._check_media_to_download, request, info, item=item, fp=fp
        )
        dfd2.addErrback(self._log_exception)
        dfd2.addBoth(self._cache_result_and_execute_waiters, target, info)
        return dfd2.addBoth(lambda _: wad)  # it must return wad at last

    def _check_media_to_download(
        self,
        result: FileInfo | None,
        request: Request,
        info: SpiderInfo,
        item: Any,
        fp: bytes | None = None,
    ) -> FileInfo | Deferred[FileInfo]:
        if result is not None:
            return result
        if fp is None:
            fp = self._fingerprinter.fingerprint(request)

        # Reuse an in-flight download of the same request, but run
        # media_downloaded() for this target so its own files are written.
        if fp in info.downloading:
            dfd = info.downloads[fp].fork()
        else:
            dfd = self._start_download(request, info, fp).fork()

        dfd2: Deferred[FileInfo] = dfd.addCallback(
            self.media_downloaded, request, info, item=item
        )
        dfd2.addErrback(self.media_failed, request, info)
        return dfd2

    def _start_download(
        self, request: Request, info: SpiderInfo, fp: bytes
    ) -> _DownloadFanout:
        dfd: Deferred[Response]
        if self.download_func:
            # this ugly code was left only to support tests. TODO: remove
            dfd = mustbe_deferred(self.download_func, request, info.spider)
        else:
            self._modify_media_request(request)
            assert self.crawler.engine
            dfd = self.crawler.engine.download(request)

        # The fan-out shares the (possibly not yet fired) download with later
        # targets of the same request. It is only shared until the download
        # itself is done: afterwards the Response is no longer kept around and
        # new targets download again (or reuse their own up-to-date files
        # through media_to_download()).
        fanout = _DownloadFanout(dfd)
        info.downloading.add(fp)
        info.downloads[fp] = fanout
        fanout.ready.addBoth(self._clear_in_flight_download, fp, info)
        return fanout

    def _clear_in_flight_download(
        self, result: Response | Failure, fp: bytes, info: SpiderInfo
    ) -> Response | Failure:
        # Runs once the shared download itself is finished, after every fork
        # created up to that point has been scheduled, so late targets simply
        # start a new download.
        info.downloading.discard(fp)
        info.downloads.pop(fp, None)
        return result

    def _log_exception(self, result: Failure) -> Failure:
        logger.exception(result)
        return result

    def _modify_media_request(self, request: Request) -> None:
        if self.handle_httpstatus_list:
            request.meta["handle_httpstatus_list"] = self.handle_httpstatus_list
        else:
            request.meta["handle_httpstatus_all"] = True

    def _cache_result_and_execute_waiters(
        self, result: FileInfo | Failure, target: Any, info: SpiderInfo
    ) -> None:
        if isinstance(result, Failure):
            # minimize cached information for failure
            result.cleanFailure()
            result.frames = []
            if twisted_version < Version("twisted", 24, 10, 0):
                result.stack = []  # type: ignore[method-assign]
            # This code fixes a memory leak by avoiding to keep references to
            # the Request and Response objects on the Media Pipeline cache.
            #
            # What happens when the media_downloaded callback raises an
            # exception, for example a FileException('download-error') when
            # the Response status code is not 200 OK, is that the original
            # StopIteration exception (which in turn contains the failed
            # Response and by extension, the original Request) gets encapsulated
            # within the FileException context.
            #
            # Originally, Scrapy was using twisted.internet.defer.returnValue
            # inside functions decorated with twisted.internet.defer.inlineCallbacks,
            # encapsulating the returned Response in a _DefGen_Return exception
            # instead of a StopIteration.
            #
            # To avoid keeping references to the Response and therefore Request
            # objects on the Media Pipeline cache, we should wipe the context of
            # the encapsulated exception when it is a StopIteration instance
            context = getattr(result.value, "__context__", None)
            if isinstance(context, StopIteration):
                result.value.__context__ = None

        info.downloaded[target] = result  # cache result per target
        for wad in info.waiting.pop(target):
            # Each waiter gets its own Deferred chain, so Failures are consumed
            # once per waiter instead of leaving unhandled errors on a shared
            # download Deferred.
            defer_result(result).chainDeferred(wad)

    def _get_target_key(
        self,
        request: Request,
        info: SpiderInfo,
        item: Any,
        fp: bytes | None = None,
    ) -> Any:
        """Identify the set of on-disk artifacts this request must deliver for
        the current item.

        Caching and de-duplication happen per target: equal requests whose
        target artifacts differ (because ``file_path()`` / ``thumb_path()``
        take the item into account) are processed independently, while equal
        requests delivering exactly the same artifacts share the work.
        """
        if fp is None:
            fp = self._fingerprinter.fingerprint(request)
        return fp

    # Overridable Interface
    @abstractmethod
    def media_to_download(
        self, request: Request, info: SpiderInfo, *, item: Any = None
    ) -> Deferred[FileInfo | None]:
        """Check request before starting download"""
        raise NotImplementedError

    @abstractmethod
    def get_media_requests(self, item: Any, info: SpiderInfo) -> list[Request]:
        """Returns the media requests to download"""
        raise NotImplementedError

    @abstractmethod
    def media_downloaded(
        self,
        response: Response,
        request: Request,
        info: SpiderInfo,
        *,
        item: Any = None,
    ) -> FileInfo:
        """Handler for success downloads"""
        raise NotImplementedError

    @abstractmethod
    def media_failed(
        self, failure: Failure, request: Request, info: SpiderInfo
    ) -> NoReturn:
        """Handler for failed downloads"""
        raise NotImplementedError

    def item_completed(
        self, results: list[FileInfoOrError], item: Any, info: SpiderInfo
    ) -> Any:
        """Called per item when all media requests has been processed"""
        if self.LOG_FAILED_RESULTS:
            for ok, value in results:
                if not ok:
                    assert isinstance(value, Failure)
                    logger.error(
                        "%(class)s found errors processing %(item)s",
                        {"class": self.__class__.__name__, "item": item},
                        exc_info=failure_to_exc_info(value),
                        extra={"spider": info.spider},
                    )
        return item

    @abstractmethod
    def file_path(
        self,
        request: Request,
        response: Response | None = None,
        info: SpiderInfo | None = None,
        *,
        item: Any = None,
    ) -> str:
        """Returns the path where downloaded media should be stored"""
        raise NotImplementedError
