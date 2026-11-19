"""Regression tests for per-item delivery of local media storage.

A single download URL can back several items whose ``file_path()`` /
``thumb_path()`` hooks return item-specific paths. The media pipeline must:

* share one concurrent download for identical requests but write and verify
  every target's own files,
* cache terminal results per target (not per request fingerprint),
* consider an image target complete only when its original image *and* every
  thumbnail configured for that item exist and are fresh,
* keep one writable target delivering its own result when another target
  sharing the same download fails to write,
* still de-duplicate work when the target paths are identical.

The tests use temporary directories, a controllable download callback whose
firing time is driven by the test, and Pillow to read generated images back.
"""

from __future__ import annotations

import hashlib
import io
import os
import time
from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp

import pytest
from twisted.internet.defer import Deferred, inlineCallbacks
from twisted.python.failure import Failure
from twisted.trial import unittest

from scrapy.http import Request, Response
from scrapy.pipelines.files import FileException, FilesPipeline, _md5sum
from scrapy.pipelines.images import ImagesPipeline
from scrapy.spiders import Spider
from scrapy.utils.test import get_crawler

try:
    from PIL import Image
except ImportError:
    pytest.skip("Pillow is required for these tests", allow_module_level=True)
else:
    encoders = {"jpeg_encoder", "jpeg_decoder"}
    if not encoders.issubset(set(Image.core.__dict__)):  # type: ignore[attr-defined]
        pytest.skip("Missing JPEG encoders", allow_module_level=True)


URL = "http://example.com/shared-manual.pdf"
IMAGE_URL = "http://example.com/shared-picture.jpg"
FILE_BODY = b"manual contents v1\n" * 100


def _image_body(size=(64, 48), color=(120, 60, 200)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "JPEG", quality=95)
    return buf.getvalue()


class ControllableDownloader:
    """Download stand-in whose Deferred is fired explicitly by the test.

    The number of actual downloads and every request seen are recorded. When
    ``fail`` is set, ``fire()`` errbacks the pending Deferreds instead of
    returning responses.
    """

    def __init__(self, body: bytes = FILE_BODY, url: str = URL, fail: bool = False):
        self.body = body
        self.url = url
        self.fail = fail
        self.calls = 0
        self.requests: list[Request] = []
        self.pending: list[Deferred] = []

    def __call__(self, request, spider):
        self.calls += 1
        self.requests.append(request)
        dfd: Deferred = Deferred()
        self.pending.append(dfd)
        return dfd

    def fire(self):
        """Fire all currently pending download Deferreds."""
        pending = self.pending
        self.pending = []
        for dfd in pending:
            if self.fail:
                dfd.errback(Failure(Exception("download boom")))
            else:
                dfd.callback(Response(self.url, status=200, body=self.body))


class ItemFilesPipeline(FilesPipeline):
    """file_path() puts each item's file in its own directory."""

    def file_path(self, request, response=None, info=None, *, item=None):
        return f"{item['catalog']}/manual.bin"

    def get_media_requests(self, item, info):
        return [Request(u) for u in item["file_urls"]]


class ItemImagesPipeline(ImagesPipeline):
    """Original images are shared, thumbnails go into the item's directory."""

    def file_path(self, request, response=None, info=None, *, item=None):
        # The original image path intentionally does not depend on the item,
        # like the default implementation.
        return super().file_path(request, response=response, info=info, item=item)

    def thumb_path(self, request, thumb_id, response=None, info=None, *, item=None):
        return f"{item['catalog']}/thumbs/{thumb_id}.jpg"

    def get_media_requests(self, item, info):
        return [Request(u) for u in item["image_urls"]]


def _item(catalog, urls=(URL,)):
    return {"catalog": catalog, "file_urls": list(urls)}


def _image_item(catalog, urls=(IMAGE_URL,)):
    return {"catalog": catalog, "image_urls": list(urls)}


def _make_files_pipeline(tempdir, downloader, settings=None):
    crawler = get_crawler(Spider, {"FILES_STORE": tempdir, **(settings or {})})
    spider = Spider("media.example")
    spider.crawler = crawler
    pipeline = ItemFilesPipeline(tempdir, crawler=crawler)
    pipeline.download_func = downloader
    pipeline.open_spider(spider)
    return pipeline


def _make_images_pipeline(tempdir, downloader, settings=None):
    crawler = get_crawler(
        Spider,
        {"IMAGES_STORE": tempdir, "IMAGES_THUMBS": {"small": (16, 16)}, **(settings or {})},
    )
    spider = Spider("media.example")
    spider.crawler = crawler
    pipeline = ItemImagesPipeline(tempdir, crawler=crawler)
    pipeline.download_func = downloader
    pipeline.open_spider(spider)
    return pipeline


def _read_file(tempdir, relative_path):
    return Path(tempdir, *relative_path.split("/")).read_bytes()


def _file_exists(tempdir, relative_path):
    return Path(tempdir, *relative_path.split("/")).exists()


def _set_mtime(tempdir, relative_path, mtime):
    os.utime(Path(tempdir, *relative_path.split("/")), (mtime, mtime))


class TestFilesPipelinePerItemTargets(unittest.TestCase):
    def setUp(self):
        self.tempdir = mkdtemp()
        self.downloader = ControllableDownloader()
        self.pipeline = _make_files_pipeline(self.tempdir, self.downloader)
        self.spider = self.pipeline.spiderinfo.spider

    def tearDown(self):
        rmtree(self.tempdir)

    @inlineCallbacks
    def test_concurrent_items_same_url_get_distinct_files(self):
        """Two items arriving while the one download is in flight must each
        get their own path, checksum and on-disk file."""
        item_a = _item("catalog-a")
        item_b = _item("catalog-b")

        d_a = self.pipeline.process_item(item_a, self.spider)
        d_b = self.pipeline.process_item(item_b, self.spider)

        # one shared download, both targets waiting on it
        assert self.downloader.calls == 1
        assert len(self.pipeline.spiderinfo.waiting) == 2

        self.downloader.fire()
        result_a = yield d_a
        result_b = yield d_b

        files_a = result_a["files"]
        files_b = result_b["files"]
        assert [f["path"] for f in files_a] == ["catalog-a/manual.bin"]
        assert [f["path"] for f in files_b] == ["catalog-b/manual.bin"]

        expected_checksum = _md5sum(io.BytesIO(FILE_BODY))
        assert files_a[0]["checksum"] == expected_checksum
        assert files_b[0]["checksum"] == expected_checksum

        # the checksum written back matches the bytes readable for that item
        assert _md5sum(io.BytesIO(_read_file(self.tempdir, "catalog-a/manual.bin"))) == expected_checksum
        assert _md5sum(io.BytesIO(_read_file(self.tempdir, "catalog-b/manual.bin"))) == expected_checksum
        assert _read_file(self.tempdir, "catalog-a/manual.bin") == FILE_BODY
        assert _read_file(self.tempdir, "catalog-b/manual.bin") == FILE_BODY

    @inlineCallbacks
    def test_late_item_after_completion_gets_its_own_file(self):
        """An item arriving after the first target finished cannot reuse the
        first item's cached FileInfo."""
        d_a = self.pipeline.process_item(_item("catalog-a"), self.spider)
        self.downloader.fire()
        result_a = yield d_a
        assert result_a["files"][0]["path"] == "catalog-a/manual.bin"
        assert self.downloader.calls == 1

        d_b = self.pipeline.process_item(_item("catalog-b"), self.spider)
        self.downloader.fire()
        result_b = yield d_b

        assert result_b["files"][0]["path"] == "catalog-b/manual.bin"
        assert self.downloader.calls == 2
        assert _file_exists(self.tempdir, "catalog-a/manual.bin")
        assert _file_exists(self.tempdir, "catalog-b/manual.bin")

        # cached results are stored per target
        assert len(self.pipeline.spiderinfo.downloaded) == 2

    @inlineCallbacks
    def test_identical_targets_are_deduplicated(self):
        """The same item target requested twice (identical item-aware paths)
        still downloads once and reuses the cached result."""
        d1 = self.pipeline.process_item(_item("catalog-a"), self.spider)
        d2 = self.pipeline.process_item(_item("catalog-a"), self.spider)
        assert self.downloader.calls == 1
        self.downloader.fire()
        r1 = yield d1
        r2 = yield d2

        assert r1["files"][0]["path"] == r2["files"][0]["path"]
        assert self.downloader.calls == 1
        assert len(self.pipeline.spiderinfo.downloaded) == 1

    @inlineCallbacks
    def test_third_target_arrives_after_completion_is_independent(self):
        """The exact issue description: after two concurrent items finish, a
        third one with the same URL still gets its own file."""
        d_a = self.pipeline.process_item(_item("catalog-a"), self.spider)
        d_b = self.pipeline.process_item(_item("catalog-b"), self.spider)
        self.downloader.fire()
        yield d_a
        yield d_b
        assert self.downloader.calls == 1

        d_c = self.pipeline.process_item(_item("catalog-c"), self.spider)
        self.downloader.fire()
        result_c = yield d_c

        assert result_c["files"][0]["path"] == "catalog-c/manual.bin"
        assert self.downloader.calls == 2
        assert _file_exists(self.tempdir, "catalog-c/manual.bin")
        assert len(self.pipeline.spiderinfo.downloaded) == 3

    @inlineCallbacks
    def test_target_joining_in_flight_download_writes_its_own_file(self):
        """An item whose target differs must join the in-flight download
        instead of starting a second one, but still produce its own file."""
        d_a = self.pipeline.process_item(_item("catalog-a"), self.spider)
        # target B joins while the single download is still pending
        d_b = self.pipeline.process_item(_item("catalog-b"), self.spider)
        d_c = self.pipeline.process_item(_item("catalog-c"), self.spider)
        assert self.downloader.calls == 1
        assert len(self.pipeline.spiderinfo.waiting) == 3

        self.downloader.fire()
        r_a = yield d_a
        r_b = yield d_b
        r_c = yield d_c
        paths = [
            r_a["files"][0]["path"],
            r_b["files"][0]["path"],
            r_c["files"][0]["path"],
        ]
        assert paths == [
            "catalog-a/manual.bin",
            "catalog-b/manual.bin",
            "catalog-c/manual.bin",
        ]
        assert self.downloader.calls == 1
        for catalog in ("catalog-a", "catalog-b", "catalog-c"):
            assert _read_file(self.tempdir, f"{catalog}/manual.bin") == FILE_BODY

    @inlineCallbacks
    def test_repeated_identical_target_after_completion_uses_cache(self):
        """Default-naming style callers keep request-level de-duplication
        even after the first processing finished."""
        d_a = self.pipeline.process_item(_item("catalog-a"), self.spider)
        self.downloader.fire()
        yield d_a

        d_a2 = self.pipeline.process_item(_item("catalog-a"), self.spider)
        result = yield d_a2
        assert self.downloader.calls == 1
        assert result["files"][0]["status"] in {"downloaded", "cached", "uptodate"}
        assert len(self.pipeline.spiderinfo.downloaded) == 1

    @inlineCallbacks
    def test_uptodate_target_skips_download_with_its_own_stat(self):
        """A fresh file on disk for this target means no download, and the
        checksum/path come from this item's own file."""
        path = Path(self.tempdir, "catalog-a", "manual.bin")
        path.parent.mkdir(parents=True)
        path.write_bytes(FILE_BODY)

        result = yield self.pipeline.process_item(_item("catalog-a"), self.spider)
        # nothing was ever downloaded
        assert self.downloader.calls == 0
        info = result["files"][0]
        assert info["status"] == "uptodate"
        assert info["path"] == "catalog-a/manual.bin"
        assert info["checksum"] == _md5sum(io.BytesIO(FILE_BODY))

    @inlineCallbacks
    def test_expired_target_is_downloaded_again(self):
        path = Path(self.tempdir, "catalog-a", "manual.bin")
        path.parent.mkdir(parents=True)
        path.write_bytes(b"stale")
        old = time.time() - (self.pipeline.expires + 1) * 60 * 60 * 24
        _set_mtime(self.tempdir, "catalog-a/manual.bin", old)

        dfd = self.pipeline.process_item(_item("catalog-a"), self.spider)
        self.downloader.fire()
        result = yield dfd

        assert self.downloader.calls == 1
        assert result["files"][0]["status"] == "downloaded"
        assert _read_file(self.tempdir, "catalog-a/manual.bin") == FILE_BODY

    @inlineCallbacks
    def test_write_failure_of_one_target_keeps_other_target(self):
        """If one directory cannot be written, the other target sharing the
        download still delivers; failures go through the item failure path."""
        original_persist = self.pipeline.store.persist_file

        def flaky_persist(p, buf, info, meta=None, headers=None):
            if p.startswith("catalog-b/"):
                raise OSError("disk full for catalog-b")
            return original_persist(p, buf, info, meta=meta, headers=headers)

        self.pipeline.store.persist_file = flaky_persist
        self.pipeline.LOG_FAILED_RESULTS = False

        d_a = self.pipeline.process_item(_item("catalog-a"), self.spider)
        d_b = self.pipeline.process_item(_item("catalog-b"), self.spider)
        assert self.downloader.calls == 1
        self.downloader.fire()
        result_a = yield d_a
        result_b = yield d_b

        # the healthy target is fully delivered
        assert [f["path"] for f in result_a["files"]] == ["catalog-a/manual.bin"]
        assert _file_exists(self.tempdir, "catalog-a/manual.bin")

        # the broken target surfaces through the ordinary failure channel
        assert result_b["files"] == []
        cached_b = next(
            v
            for k, v in self.pipeline.spiderinfo.downloaded.items()
            if k[1] == "catalog-b/manual.bin"
        )
        assert isinstance(cached_b, Failure)
        assert isinstance(cached_b.value, FileException)

    @inlineCallbacks
    def test_download_failure_reaches_every_waiting_item(self):
        self.downloader.fail = True
        self.pipeline.LOG_FAILED_RESULTS = False

        d_a = self.pipeline.process_item(_item("catalog-a"), self.spider)
        d_b = self.pipeline.process_item(_item("catalog-b"), self.spider)
        assert self.downloader.calls == 1
        self.downloader.fire()
        result_a = yield d_a
        result_b = yield d_b

        assert result_a["files"] == []
        assert result_b["files"] == []
        # each waiting item got a (Failure) result cached per target
        assert len(self.pipeline.spiderinfo.downloaded) == 2
        assert all(
            isinstance(v, Failure)
            for v in self.pipeline.spiderinfo.downloaded.values()
        )

    @inlineCallbacks
    def test_targets_are_independent_across_separate_runs(self):
        """A fresh file written in run A must not make run B report uptodate
        for its own missing file."""
        d_a = self.pipeline.process_item(_item("catalog-a"), self.spider)
        self.downloader.fire()
        result_a = yield d_a
        assert result_a["files"][0]["path"] == "catalog-a/manual.bin"

        pipeline_b = _make_files_pipeline(self.tempdir, self.downloader)
        spider_b = pipeline_b.spiderinfo.spider

        d_b = pipeline_b.process_item(_item("catalog-b"), spider_b)
        self.downloader.fire()
        result_b = yield d_b
        assert result_b["files"][0]["path"] == "catalog-b/manual.bin"
        assert _file_exists(self.tempdir, "catalog-b/manual.bin")


class TestImagesPipelinePerItemThumbnails(unittest.TestCase):
    THUMBS = {"small": (16, 16)}

    def setUp(self):
        self.tempdir = mkdtemp()
        self.image_body = _image_body()
        self.downloader = ControllableDownloader(body=self.image_body, url=IMAGE_URL)
        self.pipeline = _make_images_pipeline(self.tempdir, self.downloader)
        self.spider = self.pipeline.spiderinfo.spider

    def tearDown(self):
        rmtree(self.tempdir)

    def _orig_path(self):
        guid = hashlib.sha1(IMAGE_URL.encode()).hexdigest()
        return f"full/{guid}.jpg"

    @inlineCallbacks
    def test_full_uptodate_including_thumbnails(self):
        """Fresh original AND fresh item-specific thumbnails: no download."""
        # simulate a previous run that delivered everything
        dfd = self.pipeline.process_item(_image_item("catalog-a"), self.spider)
        self.downloader.fire()
        yield dfd
        assert self.downloader.calls == 1
        assert _file_exists(self.tempdir, self._orig_path())
        assert _file_exists(self.tempdir, "catalog-a/thumbs/small.jpg")

        # new crawl, same store
        pipeline2 = _make_images_pipeline(self.tempdir, self.downloader)
        spider2 = pipeline2.spiderinfo.spider

        result = yield pipeline2.process_item(_image_item("catalog-a"), spider2)
        assert self.downloader.calls == 1  # no extra download
        info = result["images"][0]
        assert info["status"] == "uptodate"
        assert info["path"] == self._orig_path()
        assert info["checksum"] == _md5sum(io.BytesIO(self.image_body))

    @inlineCallbacks
    def test_fresh_original_but_missing_thumbnail_forces_download(self):
        """The reproduction scenario: another catalog delivered the shared
        original image, but this item's own thumbnails do not exist."""
        # run A delivers original + A's thumbnail
        dfd = self.pipeline.process_item(_image_item("catalog-a"), self.spider)
        self.downloader.fire()
        yield dfd
        assert self.downloader.calls == 1

        # brand new crawl handling catalog B; original image not expired
        pipeline_b = _make_images_pipeline(self.tempdir, self.downloader)
        spider_b = pipeline_b.spiderinfo.spider
        assert not _file_exists(self.tempdir, "catalog-b/thumbs/small.jpg")

        d_b = pipeline_b.process_item(_image_item("catalog-b"), spider_b)
        # the original was fresh but the thumbnail was missing: re-download
        self.downloader.fire()
        result_b = yield d_b

        assert self.downloader.calls == 2
        assert result_b["images"][0]["path"] == self._orig_path()
        thumb_path = Path(self.tempdir, "catalog-b", "thumbs", "small.jpg")
        assert thumb_path.exists()
        with Image.open(thumb_path) as thumb:
            thumb.load()
            assert max(thumb.size) <= 16

    @inlineCallbacks
    def test_fresh_original_but_expired_thumbnail_forces_download(self):
        # deliver everything once
        dfd = self.pipeline.process_item(_image_item("catalog-a"), self.spider)
        self.downloader.fire()
        yield dfd

        # age only the thumbnail beyond the expiry window
        old = time.time() - (self.pipeline.expires + 1) * 60 * 60 * 24
        _set_mtime(self.tempdir, "catalog-a/thumbs/small.jpg", old)

        pipeline2 = _make_images_pipeline(self.tempdir, self.downloader)
        spider2 = pipeline2.spiderinfo.spider

        d2 = pipeline2.process_item(_image_item("catalog-a"), spider2)
        self.downloader.fire()
        result = yield d2

        assert self.downloader.calls == 2
        assert result["images"][0]["status"] == "downloaded"
        with Image.open(Path(self.tempdir, "catalog-a", "thumbs", "small.jpg")) as im:
            im.load()

    @inlineCallbacks
    def test_concurrent_items_share_original_but_each_get_thumbnail(self):
        d_a = self.pipeline.process_item(_image_item("catalog-a"), self.spider)
        d_b = self.pipeline.process_item(_image_item("catalog-b"), self.spider)
        assert self.downloader.calls == 1
        self.downloader.fire()
        result_a = yield d_a
        result_b = yield d_b

        # original result fields are identical for both items
        assert result_a["images"][0]["path"] == result_b["images"][0]["path"]
        assert result_a["images"][0]["checksum"] == result_b["images"][0]["checksum"]

        for catalog in ("catalog-a", "catalog-b"):
            thumb = Path(self.tempdir, catalog, "thumbs", "small.jpg")
            assert thumb.exists()
            with Image.open(thumb) as im:
                im.load()
                assert max(im.size) <= 16

        # both target results were cached independently
        assert len(self.pipeline.spiderinfo.downloaded) == 2

    @inlineCallbacks
    def test_thumbnail_write_failure_isolated_between_targets(self):
        original_persist = self.pipeline.store.persist_file

        def flaky_persist(p, buf, info, meta=None, headers=None):
            if p.startswith("catalog-b/"):
                raise OSError("cannot write catalog-b thumbnails")
            return original_persist(p, buf, info, meta=meta, headers=headers)

        self.pipeline.store.persist_file = flaky_persist
        self.pipeline.LOG_FAILED_RESULTS = False

        d_a = self.pipeline.process_item(_image_item("catalog-a"), self.spider)
        d_b = self.pipeline.process_item(_image_item("catalog-b"), self.spider)
        self.downloader.fire()
        result_a = yield d_a
        result_b = yield d_b

        assert result_a["images"][0]["path"] == self._orig_path()
        assert _file_exists(self.tempdir, "catalog-a/thumbs/small.jpg")
        assert result_b["images"] == []
        assert not _file_exists(self.tempdir, "catalog-b/thumbs/small.jpg")
