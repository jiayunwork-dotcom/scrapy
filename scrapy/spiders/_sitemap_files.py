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
the spider closes are removed by :meth:`close`, which also removes the
temporary directory.

Files requested explicitly through the ``body_file`` request meta key with a
user-provided path never go through this registry and are never deleted here.
"""

from __future__ import annotations

import contextlib
import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

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


@contextlib.contextmanager
def _suppress_oserror() -> Iterator[None]:
    try:
        yield
    except OSError as exc:
        logger.debug("Ignoring error while removing a temporary file: %s", exc)


class SitemapFileRegistry:
    """Owns the temporary directory and tracks the files created in it."""

    def __init__(self, prefix: str = "scrapy-sitemap-") -> None:
        self._tempdir = tempfile.TemporaryDirectory(prefix=prefix)
        self._paths: set[Path] = set()
        self._closed = False

    @property
    def directory(self) -> Path:
        """Path of the directory holding the temporary files."""
        return Path(self._tempdir.name)

    def track(self, path: str | Path) -> Path:
        """Start tracking *path*.

        The file is expected to be created by the caller, inside
        :attr:`directory`. Tracking it guarantees its reclamation even if the
        caller crashes before releasing it.
        """
        path = Path(path)
        self._paths.add(path)
        return path

    def release(self, path: str | Path) -> None:
        """Delete *path* and stop tracking it.

        A missing file is not an error: a previous cleanup may already have
        removed it.
        """
        path = Path(path)
        self._paths.discard(path)
        with _suppress_oserror():
            path.unlink()

    def close(self) -> None:
        """Delete every file still tracked and remove the temporary directory.

        Safe to call more than once.
        """
        if self._closed:
            return
        self._closed = True
        for path in tuple(self._paths):
            with _suppress_oserror():
                path.unlink()
        self._paths.clear()
        self._tempdir.cleanup()
