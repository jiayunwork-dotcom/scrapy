from __future__ import annotations

import logging
import re
import warnings

# Iterable is needed at the run time for the SitemapSpider._parse_sitemap() annotation
from collections.abc import AsyncIterator, Iterable, Sequence  # noqa: TC003
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self, cast
from urllib.parse import urljoin

import lxml.etree  # type: ignore[import-untyped]

from scrapy import signals
from scrapy.exceptions import IgnoreRequest, ScrapyDeprecationWarning
from scrapy.http import BodyFileResponse, Request, Response, XmlResponse
from scrapy.responsetypes import responsetypes
from scrapy.spiders import Spider
from scrapy.spiders._sitemap_files import (
    SITEMAP_FILE_BACKED_META,
    SITEMAP_FILE_DIR_META,
    SitemapFileRegistry,
)
from scrapy.utils._compression import (
    _DECOMPRESSION_ERRORS,
    _DecompressionMaxSizeExceeded,
    gunzip,
    gunzip_file,
    gzip_magic_number,
    is_gzip_file,
)
from scrapy.utils._sitemap import Sitemap, sitemap_urls_from_robots
from scrapy.utils.httpobj import urlparse_cached

if TYPE_CHECKING:
    from scrapy.crawler import Crawler
    from scrapy.http.request import CallbackT
    from scrapy.robotstxt import RobotParser

logger = logging.getLogger(__name__)

# Exceptions that make a sitemap file's own gzip layer unusable. Defined as a
# concrete tuple (not built inline) so it is valid in an except clause and
# keeps the strict, fail-loud behavior of file-backed decompression.
_SITEMAP_GZIP_ERRORS: tuple[type[BaseException], ...] = (
    *_DECOMPRESSION_ERRORS,
    OSError,
)


class IgnoreSitemap(IgnoreRequest):
    """Raised when a file-backed sitemap cannot be turned into URLs (a broken
    gzip layer, a truncated document or an exceeded size limit), so that the
    failure is explicit instead of the compressed bytes being parsed as an
    empty map."""


class SitemapSpider(Spider):
    """Spider that crawls a site by discovering its URLs using `sitemaps
    <https://www.sitemaps.org/index.html>`_.

    It supports nested sitemaps and discovering sitemap URLs from `robots.txt
    <https://www.robotstxt.org/>`_.

    With the :setting:`SITEMAP_FILE_BACKED` setting enabled, sitemap documents
    are downloaded into Scrapy-managed temporary files and parsed lazily from
    disk instead of being kept in memory. The response handed to
    :meth:`sitemap_request` is then a
    :class:`~scrapy.http.BodyFileResponse` whose file is valid until the
    document has been parsed, its entry stream has been closed, or processing
    has failed; the file (and every copy of the response pointing at it) must
    not be used afterwards.
    """

    sitemap_urls: Sequence[str] = ()
    """URLs pointing to the sitemaps whose URLs you want to crawl.

    You can also point to a `robots.txt <https://www.robotstxt.org/>`_ and it
    will be parsed to extract sitemap URLs from it.
    """

    sitemap_rules: Sequence[tuple[re.Pattern[str] | str, str | CallbackT]] = [
        ("", "parse")
    ]
    """``(regex, callback)`` tuples where:

    -   ``regex`` is a regular expression to match URLs extracted from
        sitemaps. ``regex`` can be either a str or a compiled regex object.

    -   ``callback`` is the callback to use for processing the URLs that match
        the regular expression. ``callback`` can be a string (indicating the
        name of a spider method) or a callable.

    For example:

    .. code-block:: python

        sitemap_rules = [("/product/", "parse_product")]

    Rules are applied in order, and only the first one that matches will be
    used.

    The default value makes all URLs found in sitemaps be processed with the
    :meth:`~scrapy.Spider.parse` callback.
    """

    sitemap_follow: Sequence[re.Pattern[str] | str] = [""]
    """Regexes of sitemap URLs that should be followed: those linked from
    `sitemap index files <https://www.sitemaps.org/protocol.html#index>`_, and
    those declared through the ``Sitemap`` directive of a :file:`robots.txt`
    file.

    By default, all sitemaps are followed.
    """

    sitemap_alternate_links: bool = False
    """Specifies if alternate links for one ``url`` should be followed. These are
    links for the same website in another language passed within the same
    ``url`` block.

    For example:

    .. code-block:: xml

        <url>
            <loc>http://example.com/</loc>
            <xhtml:link rel="alternate" hreflang="de" href="http://example.com/de"/>
        </url>

    When enabled, this would retrieve both URLs. When disabled, only
    ``http://example.com/`` would be retrieved.
    """

    sitemap_schemes: Sequence[str] | None = None
    """URL schemes that the URLs found in sitemaps and :file:`robots.txt` files
    may use.

    .. versionadded:: 2.20.0

    If ``None``, they may only use ``http``, ``https`` or the scheme of the
    sitemap or :file:`robots.txt` file they come from. Use ``()`` to follow
    URLs regardless of their scheme.
    """

    _max_size: int
    _warn_size: int
    _obey_robotstxt: bool
    _sitemap_file_backed: bool
    _sitemap_file_registry: SitemapFileRegistry | None

    @classmethod
    def from_crawler(cls, crawler: Crawler, *args: Any, **kwargs: Any) -> Self:
        spider = super().from_crawler(crawler, *args, **kwargs)
        spider._max_size = getattr(
            spider, "download_maxsize", spider.settings.getint("DOWNLOAD_MAXSIZE")
        )
        spider._warn_size = getattr(
            spider, "download_warnsize", spider.settings.getint("DOWNLOAD_WARNSIZE")
        )
        spider._obey_robotstxt = crawler.settings.getbool("ROBOTSTXT_OBEY")
        # Strict boolean: unlike getbool(), only a real bool is accepted, so a
        # typo such as SITEMAP_FILE_BACKED = "true" fails loudly before the
        # first request is sent.
        file_backed = crawler.settings.get("SITEMAP_FILE_BACKED", False)
        if not isinstance(file_backed, bool):
            raise ValueError(
                "The SITEMAP_FILE_BACKED setting only accepts a boolean "
                f"(True or False), got {file_backed!r} of type "
                f"{type(file_backed).__name__}."
            )
        spider._sitemap_file_backed = file_backed
        spider._sitemap_file_registry = None
        if file_backed:
            spider._sitemap_file_registry = SitemapFileRegistry()
            crawler.signals.connect(
                spider._close_sitemap_file_registry, signals.spider_closed
            )
        if spider._obey_robotstxt:
            crawler.signals.connect(spider._robots_parsed, signals.robots_parsed)
        return spider

    def _close_sitemap_file_registry(self) -> None:
        if self._sitemap_file_registry is not None:
            self._sitemap_file_registry.close()
            self._sitemap_file_registry = None

    def __init__(self, *a: Any, **kw: Any):
        super().__init__(*a, **kw)
        self._cbs: list[tuple[re.Pattern[str], CallbackT]] = []
        for r, c in self.sitemap_rules:
            if isinstance(c, str):
                c = cast("CallbackT", getattr(self, c))  # noqa: PLW2901
            self._cbs.append((regex(r), c))
        self._follow: list[re.Pattern[str]] = [regex(x) for x in self.sitemap_follow]
        self._robots_sitemaps: dict[str, list[str]] = {}
        self._signal_only_netlocs: set[str] = set()
        # Defaults matching SITEMAP_FILE_BACKED = False; from_crawler() enables
        # the file-backed machinery when the setting is on.
        self._sitemap_file_backed: bool = False
        self._sitemap_file_registry: SitemapFileRegistry | None = None

    def _robots_parsed(
        self, robotparser: RobotParser, request: Request, response: Response
    ) -> None:
        netloc = urlparse_cached(request).netloc
        urls = [urljoin(request.url, url) for url in robotparser.sitemaps()]
        if netloc in self._signal_only_netlocs:
            # No request for this host's robots.txt was sent from start(), so
            # _parse_sitemap() will never run for it; schedule its sitemaps
            # directly instead of stashing them for a response that never arrives.
            self._signal_only_netlocs.discard(netloc)
            for url in urls:
                if self._scheme_allowed(url, response.url) and any(
                    x.search(url) for x in self._follow
                ):
                    self.crawler.engine.crawl(
                        self._mark_sitemap_request(
                            self.sitemap_request(
                                url,
                                self._parse_sitemap,
                                {"loc": url},
                                source="robotstxt",
                                response=response,
                            )
                        )
                    )
        else:
            self._robots_sitemaps[netloc] = urls

    async def start(self) -> AsyncIterator[Any]:
        sitemap_netlocs = {
            urlparse_cached(url).netloc
            for url in self.sitemap_urls
            if not url.endswith("/robots.txt")
        }
        for url in self.sitemap_urls:
            netloc = urlparse_cached(url).netloc
            if (
                self._obey_robotstxt
                and url.endswith("/robots.txt")
                and netloc in sitemap_netlocs
            ):
                # RobotsTxtMiddleware will fetch and parse this robots.txt
                # anyway, to check the other sitemap_urls requests to the same
                # host, so rely on the robots_parsed signal for it instead of
                # requesting it a second time here.
                self._signal_only_netlocs.add(netloc)
                continue
            # robots.txt itself keeps being read the regular way; only sitemap
            # documents are downloaded to files in file-backed mode.
            request = Request(url, self._parse_sitemap)
            if self._sitemap_file_backed and not url.endswith("/robots.txt"):
                self._mark_sitemap_request(request)
            yield request

    def _mark_sitemap_request(self, request: Request) -> Request:
        """Mark *request* as a file-backed sitemap request when the mode is on.

        The download handler writes its body to a Scrapy-managed temporary
        file (``body_file=True``) inside the spider's sitemap temporary
        directory, and the HTTP compression middleware streams any HTTP
        Content-Encoding file-to-file. Only requests to sitemap documents are
        marked; robots.txt and final page requests are not.
        """
        if not self._sitemap_file_backed:
            return request
        registry = self._sitemap_file_registry
        assert registry is not None
        request.meta[SITEMAP_FILE_BACKED_META] = True
        request.meta["body_file"] = True
        request.meta[SITEMAP_FILE_DIR_META] = str(registry.directory)
        return request

    def sitemap_filter(
        self, entries: Iterable[dict[str, Any]]
    ) -> Iterable[dict[str, Any]]:
        """Yield the sitemap entries from *entries* that should be processed.

        Override it to select sitemap entries based on their attributes. For
        example, given the following sitemap entry:

        .. code-block:: xml

            <url>
                <loc>http://example.com/</loc>
                <lastmod>2005-01-01</lastmod>
            </url>

        You can filter entries by date as follows:

        .. code-block:: python

            from datetime import datetime
            from scrapy.spiders import SitemapSpider


            class FilteredSitemapSpider(SitemapSpider):
                name = "filtered_sitemap_spider"
                allowed_domains = ["example.com"]
                sitemap_urls = ["http://example.com/sitemap.xml"]

                def sitemap_filter(self, entries):
                    for entry in entries:
                        date_time = datetime.strptime(entry["lastmod"], "%Y-%m-%d")
                        if date_time.year >= 2005:
                            yield entry

        This would retrieve only entries modified on 2005 and the following
        years.

        Entries are dict objects extracted from the sitemap document. Usually,
        the key is the tag name and the value is the text inside it.

        It's important to notice that:

        -   as the ``loc`` attribute is required, entries without this tag are
            discarded
        -   alternate links are stored in a list with the key ``alternate``
            (see :attr:`sitemap_alternate_links`)
        -   namespaces are removed, so lxml tags named as ``{namespace}tagname``
            become only ``tagname``

        The default implementation yields all entries, observing other
        attributes and their settings.

        When :setting:`SITEMAP_FILE_BACKED` is enabled, *entries* is a
        single-pass stream backed by a file: iterate it once and yield entries
        as you go, without collecting them (for example with ``list()``), as
        buffering the entries defeats the flat-memory guarantee. Stopping the
        iteration early abandons the unread remainder of the file.
        """
        yield from entries

    def sitemap_request(
        self,
        loc: str,
        callback: CallbackT,
        entry: dict[str, Any],
        *,
        source: Literal["robotstxt", "sitemapindex", "urlset"],
        response: Response,
    ) -> Request:
        """Return the request for the *loc* URL of the sitemap *entry*, to be
        parsed with *callback*.

        .. versionadded:: 2.20.0

        Override this method to build those requests differently, e.g. to
        carry sitemap data and request metadata over to *callback*:

        .. code-block:: python

            def sitemap_request(self, loc, callback, entry, *, source, response):
                meta = {
                    "lastmod": entry.get("lastmod"),
                    "site": response.meta.get("site"),
                }
                return Request(loc, callback, meta=meta)

        *entry* is the sitemap entry that *loc* comes from, as described in
        :meth:`sitemap_filter`. When :attr:`sitemap_alternate_links` is
        enabled, the same entry is used for every alternate URL.

        *source* tells where *loc* comes from: ``"urlset"`` for a URL to
        crawl, ``"sitemapindex"`` for a sitemap linked from a sitemap index,
        and ``"robotstxt"`` for a sitemap linked from a :file:`robots.txt`
        file, where *entry* only has ``loc``.

        *response* is the response that *entry* was read from. When
        :setting:`SITEMAP_FILE_BACKED` is enabled it is a
        :class:`~scrapy.http.BodyFileResponse` whose decoded XML lives in a
        temporary file; the file stays readable for the whole processing of
        this sitemap and is reclaimed once parsing ends, is closed early, or
        fails. Do not keep the response or its file path for use after this
        method returns.
        """
        return Request(loc, callback=callback)

    def _parse_sitemap(self, response: Response) -> Iterable[Request]:
        if response.url.endswith("/robots.txt"):
            urls = list(sitemap_urls_from_robots(response.body, base_url=response.url))
            netloc = urlparse_cached(response).netloc
            for url in self._robots_sitemaps.pop(netloc, ()):
                if url not in urls:
                    urls.append(url)
            return (
                self._mark_derived_sitemap_request(
                    self.sitemap_request(
                        url,
                        self._parse_sitemap,
                        {"loc": url},
                        source="robotstxt",
                        response=response,
                    )
                )
                for url in urls
                if self._scheme_allowed(url, response.url)
                and any(x.search(url) for x in self._follow)
            )

        if (
            isinstance(response, BodyFileResponse)
            and response.request is not None
            and response.request.meta.get(SITEMAP_FILE_BACKED_META)
        ):
            return self._parse_sitemap_file(response)

        body = self._get_sitemap_body(response)
        if not body:
            logger.warning(
                "Ignoring invalid sitemap: %(response)s",
                {"response": response},
                extra={"spider": self},
            )
            return ()

        s = Sitemap(body)

        if s.type == "sitemapindex":
            index_entries = list(
                self._get_urls_from_sitemapindex(self.sitemap_filter(s), response.url)
            )
            return (
                self._mark_derived_sitemap_request(
                    self.sitemap_request(
                        loc,
                        self._parse_sitemap,
                        entry,
                        source="sitemapindex",
                        response=response,
                    )
                )
                for loc, entry in index_entries
            )

        if s.type == "urlset":
            urlset_entries = list(
                self._get_urls_and_callbacks_from_urlset(
                    self.sitemap_filter(s), response.url
                )
            )
            return (
                self.sitemap_request(loc, c, entry, source="urlset", response=response)
                for loc, c, entry in urlset_entries
            )

        logger.warning(
            "Ignoring invalid sitemap: %(response)s",
            {"response": response},
            extra={"spider": self},
        )

        return ()

    def _mark_derived_sitemap_request(self, request: Request) -> Request:
        """Mark a request to another sitemap (from a sitemap index or
        robots.txt) as file-backed when the mode is on. Requests to final
        pages are not marked."""
        return self._mark_sitemap_request(request)

    def _parse_sitemap_file(
        self, response: BodyFileResponse
    ) -> Iterable[Request]:
        """Stream-parse a sitemap whose body lives in a temporary file.

        The body is never loaded into memory: the file is opened, optionally
        gunzipped (for a sitemap that is itself a gzip file without an HTTP
        Content-Encoding), parsed lazily with :class:`Sitemap`, and released
        back to the :class:`SitemapFileRegistry` as soon as parsing ends, is
        closed early, or fails.
        """
        return self._iter_file_sitemap_requests(response)

    def _iter_file_sitemap_requests(
        self, response: BodyFileResponse
    ) -> Iterable[Request]:
        registry = self._sitemap_file_registry
        assert registry is not None
        body_path = Path(response.body_file)
        # A file produced for a previous attempt must never be reused: the
        # download handler allocates a fresh path per attempt, and only this
        # response's path is parsed here.
        gz_path: Path | None = None
        sitemap: Sitemap | None = None
        try:
            gz_path = self._maybe_gunzip_sitemap_file(response, body_path, registry)
            xml_path = gz_path if gz_path is not None else body_path
            xml_file = xml_path.open("rb")
            try:
                try:
                    sitemap = Sitemap.from_file(xml_file)
                except (OSError, lxml.etree.LxmlError) as e:
                    # An unreadable or unparseable document is a real failure,
                    # not an empty sitemap: raise it instead of letting the
                    # recovery parser turn it into a misleading empty map.
                    raise IgnoreSitemap(
                        f"Ignoring sitemap {response}: it could not be parsed: {e}"
                    ) from e
                yield from self._iter_requests_from_file_sitemap(sitemap, response)
                # Reaching here means the whole document was parsed without an
                # error; only then is it counted as read to completion.
                sitemap.completed = True
            finally:
                if sitemap is not None:
                    sitemap.close()
                else:
                    xml_file.close()
        finally:
            # Reclaim every file created for this response. The registry also
            # sweeps anything left behind on spider close.
            if gz_path is not None:
                registry.release(gz_path)
            registry.release(body_path)

    def _maybe_gunzip_sitemap_file(
        self,
        response: BodyFileResponse,
        body_path: Path,
        registry: SitemapFileRegistry,
    ) -> Path | None:
        """Gunzip a sitemap file that is itself a gzip document into a new
        tracked temporary file, and return its path.

        Return ``None`` when the downloaded file is not gzip. The gzip layer of
        the sitemap file counts separately from the HTTP Content-Encoding
        layers, which are handled by the compression middleware.
        """
        if not is_gzip_file(body_path):
            return None
        gz_path = body_path.with_name(body_path.name + ".gunzip")
        # circular import
        from scrapy.utils._download_handlers import get_warnsize  # noqa: PLC0415

        max_size = response.meta.get("download_maxsize", self._max_size)
        warn_size = get_warnsize(response.meta, self._warn_size)
        try:
            with body_path.open("rb") as input_file, gz_path.open(
                "wb"
            ) as output_file:
                decompressed_size = gunzip_file(
                    input_file, output_file, max_size=max_size
                )
        except _DecompressionMaxSizeExceeded as e:
            registry.release(gz_path)
            raise IgnoreSitemap(
                f"Ignoring sitemap {response}: its gunzipped body "
                f"({e.decompressed_size} B so far) exceeded "
                f"DOWNLOAD_MAXSIZE ({max_size} B)."
            ) from e
        except (*_SITEMAP_GZIP_ERRORS,) as e:
            registry.release(gz_path)
            raise IgnoreSitemap(
                f"Ignoring sitemap {response}: its gzip layer could "
                f"not be decoded: {e}"
            ) from e
        registry.track(gz_path)
        if warn_size and decompressed_size >= warn_size:
            logger.warning(
                f"{response} body size after decompression "
                f"({decompressed_size} B) is larger than or equal to "
                f"the download warning size ({warn_size} B)."
            )
        return gz_path

    def _iter_requests_from_file_sitemap(
        self, sitemap: Sitemap, response: BodyFileResponse
    ) -> Iterable[Request]:
        if sitemap.type == "sitemapindex":
            index_entries = self._get_urls_from_sitemapindex(
                self.sitemap_filter(sitemap), response.url
            )
            for loc, entry in index_entries:
                yield self._mark_derived_sitemap_request(
                    self.sitemap_request(
                        loc,
                        self._parse_sitemap,
                        entry,
                        source="sitemapindex",
                        response=response,
                    )
                )
            return
        if sitemap.type == "urlset":
            urlset_entries = self._get_urls_and_callbacks_from_urlset(
                self.sitemap_filter(sitemap), response.url
            )
            for loc, callback, entry in urlset_entries:
                yield self.sitemap_request(
                    loc, callback, entry, source="urlset", response=response
                )
            return
        logger.warning(
            "Ignoring invalid sitemap: %(response)s",
            {"response": response},
            extra={"spider": self},
        )

    def _get_urls_from_sitemapindex(
        self, it: Iterable[dict[str, Any]], base_url: str
    ) -> Iterable[tuple[str, dict[str, Any]]]:
        for loc, entry in _iterlocs(it, self.sitemap_alternate_links):
            loc = urljoin(base_url, loc)  # noqa: PLW2901
            if self._scheme_allowed(loc, base_url) and any(
                x.search(loc) for x in self._follow
            ):
                yield loc, entry

    def _get_urls_and_callbacks_from_urlset(
        self, it: Iterable[dict[str, Any]], base_url: str
    ) -> Iterable[tuple[str, CallbackT, dict[str, Any]]]:
        for loc, entry in _iterlocs(it, self.sitemap_alternate_links):
            loc = urljoin(base_url, loc)  # noqa: PLW2901
            if not self._scheme_allowed(loc, base_url):
                continue
            for r, c in self._cbs:
                if r.search(loc):
                    yield loc, c, entry
                    break

    def _scheme_allowed(self, url: str, source_url: str) -> bool:
        if self.sitemap_schemes is None:
            schemes = {"http", "https", urlparse_cached(source_url).scheme}
        else:
            schemes = set(self.sitemap_schemes)
        return not schemes or urlparse_cached(url).scheme in schemes

    def _get_sitemap_body(self, response: Response) -> bytes | None:
        """Return the sitemap body contained in the given response,
        or None if the response is not a sitemap.
        """
        # Download handlers other than the built-in ones may use their own
        # response classes, so the content type, the URL and the body are taken
        # into account as well.
        if (
            isinstance(response, XmlResponse)
            or responsetypes.from_args(
                headers=response.headers, url=response.url, body=response.body
            )
            is XmlResponse
        ):
            return response.body
        if gzip_magic_number(response):
            uncompressed_size = len(response.body)
            max_size = response.meta.get("download_maxsize", self._max_size)
            # circular import
            from scrapy.utils._download_handlers import get_warnsize  # noqa: PLC0415

            warn_size = get_warnsize(response.meta, self._warn_size)
            try:
                body = gunzip(response.body, max_size=max_size)
            except _DecompressionMaxSizeExceeded:
                return None
            if uncompressed_size < warn_size <= len(body):
                logger.warning(
                    f"{response} body size after decompression ({len(body)} B) "
                    f"is larger than the download warning size ({warn_size} B)."
                )
            return body
        # actual gzipped sitemap files are decompressed above ;
        # if we are here (response body is not gzipped)
        # and have a response for .xml.gz,
        # it usually means that it was already gunzipped
        # by HttpCompression middleware,
        # the HTTP response being sent with "Content-Encoding: gzip"
        # without actually being a .xml.gz file in the first place,
        # merely XML gzip-compressed on the fly,
        # in other word, here, we have plain XML
        if response.url.endswith(".xml") or response.url.endswith(".xml.gz"):
            return response.body
        return None


def regex(x: re.Pattern[str] | str) -> re.Pattern[str]:
    if isinstance(x, str):
        return re.compile(x)
    return x


def iterloc(it: Iterable[dict[str, Any]], alt: bool = False) -> Iterable[str]:
    warnings.warn(
        "scrapy.spiders.sitemap.iterloc() is deprecated.",
        ScrapyDeprecationWarning,
        stacklevel=2,
    )
    return (loc for loc, _ in _iterlocs(it, alt))


def _iterlocs(
    it: Iterable[dict[str, Any]], alt: bool = False
) -> Iterable[tuple[str, dict[str, Any]]]:
    for d in it:
        if loc := d["loc"]:
            yield loc, d

        # Also consider alternate URLs (xhtml:link rel="alternate")
        if alt and (alt_list := d.get("alternate")):
            for loc in alt_list:
                yield loc, d
