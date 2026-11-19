"""Standalone, re-runnable reproduction of the per-item media storage bug.

Run with:  python -I repro_media_targets.py

It uses only temporary directories, an in-memory controllable download
callback and Pillow to read generated images back. Every check prints its
result; the script exits non-zero if any check fails.

Scenarios:

FilesPipeline
  F1. Two items with the same URL arrive while the download is in flight:
      each must get its own path/checksum and its own on-disk file.
  F2. A third item arrives after the download finished: same requirements.
  F3. Identical targets (same item twice) still download only once.
  F4. One of two per-item target directories fails to write: the other
      target still delivers; the failed one goes through the item failure
      channel; every waiting item gets a result.

ImagesPipeline
  I1. Full hit: fresh original image AND fresh item-specific thumbnails ->
      no download, uptodate.
  I2. Fresh original image but a missing item-specific thumbnail -> the
      item is not reported complete; the image is re-downloaded and the
      thumbnail generated.
  I3. Fresh original image but an expired thumbnail -> re-download.
  I4. Two items share one original image concurrently and each reads back
      its own thumbnail; the download happens once.
  I5. One thumbnail directory cannot be written: the other item still
      gets its thumbnail.
"""

from __future__ import annotations

import hashlib
import io
import os
import sys
import tempfile
import time
from pathlib import Path

from twisted.internet.defer import Deferred, inlineCallbacks
from twisted.python.failure import Failure

from scrapy.http import Request, Response
from scrapy.pipelines.files import FileException, FilesPipeline, _md5sum
from scrapy.pipelines.images import ImagesPipeline
from scrapy.spiders import Spider
from scrapy.utils.test import get_crawler

try:
    from PIL import Image
except ImportError:
    print("Pillow is required", file=sys.stderr)
    raise

URL = "http://example.com/shared-manual.pdf"
IMAGE_URL = "http://example.com/shared-picture.jpg"
BODY = b"manual contents v1\n" * 128

RESULTS = {"passed": 0, "failed": 0}


def check(name, condition, detail=""):
    if condition:
        RESULTS["passed"] += 1
        print(f"  PASS  {name}")
    else:
        RESULTS["failed"] += 1
        print(f"  FAIL  {name} {detail}")


def image_body():
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), (120, 60, 200)).save(buf, "JPEG", quality=95)
    return buf.getvalue()


class ControlledDownload:
    def __init__(self, body, url):
        self.body = body
        self.url = url
        self.calls = 0
        self.pending = []

    def __call__(self, request, spider):
        self.calls += 1
        dfd = Deferred()
        self.pending.append(dfd)
        return dfd

    def fire(self):
        pending = self.pending
        self.pending = []
        for dfd in pending:
            dfd.callback(Response(self.url, status=200, body=self.body))


class PerItemFilesPipeline(FilesPipeline):
    def file_path(self, request, response=None, info=None, *, item=None):
        return f"{item['catalog']}/manual.bin"

    def get_media_requests(self, item, info):
        return [Request(u) for u in item["file_urls"]]


class PerItemImagesPipeline(ImagesPipeline):
    def thumb_path(self, request, thumb_id, response=None, info=None, *, item=None):
        return f"{item['catalog']}/thumbs/{thumb_id}.jpg"

    def get_media_requests(self, item, info):
        return [Request(u) for u in item["image_urls"]]


def build_pipeline(store, cls, settings_key, download):
    crawler = get_crawler(Spider, {settings_key: store, "IMAGES_THUMBS": {"small": (16, 16)}})
    spider = Spider("media.example")
    spider.crawler = crawler
    pipe = cls(store, crawler=crawler)
    pipe.download_func = download
    pipe.open_spider(spider)
    return pipe, spider


def file_item(catalog):
    return {"catalog": catalog, "file_urls": [URL]}


def image_item(catalog):
    return {"catalog": catalog, "image_urls": [IMAGE_URL]}


def read(store, rel):
    path = Path(store, *rel.split("/"))
    if not path.exists():
        return None
    return path.read_bytes()


def checksum_of(store, rel):
    data = read(store, rel)
    if data is None:
        return None
    return _md5sum(io.BytesIO(data))


def exists(store, rel):
    return Path(store, *rel.split("/")).exists()


def age(store, rel, seconds):
    p = Path(store, *rel.split("/"))
    if p.exists():
        os.utime(p, (seconds, seconds))


@inlineCallbacks
def run_files_scenarios(store):
    print("FilesPipeline")
    download = ControlledDownload(BODY, URL)
    pipe, spider = build_pipeline(store, PerItemFilesPipeline, "FILES_STORE", download)
    checksum = _md5sum(io.BytesIO(BODY))

    # F1: concurrent items, same URL, item-specific paths
    da = pipe.process_item(file_item("catalog-a"), spider)
    db = pipe.process_item(file_item("catalog-b"), spider)
    check("F1 single download for concurrent items", download.calls == 1)
    download.fire()
    ra = yield da
    rb = yield db
    check(
        "F1 item A own path",
        [f["path"] for f in ra["files"]] == ["catalog-a/manual.bin"],
        str(ra.get("files")),
    )
    check(
        "F1 item B own path",
        [f["path"] for f in rb["files"]] == ["catalog-b/manual.bin"],
        str(rb.get("files")),
    )
    check("F1 checksums match readable files (A)",
          ra["files"][0]["checksum"] == checksum == checksum_of(store, "catalog-a/manual.bin"))
    check("F1 checksums match readable files (B)",
          rb["files"][0]["checksum"] == checksum == checksum_of(store, "catalog-b/manual.bin"))
    check("F1 both files on disk",
          read(store, "catalog-a/manual.bin") == BODY and read(store, "catalog-b/manual.bin") == BODY)

    # F2: late item after completion
    dc = pipe.process_item(file_item("catalog-c"), spider)
    download.fire()
    rc = yield dc
    check("F2 late item gets its own path",
          rc["files"][0]["path"] == "catalog-c/manual.bin", str(rc.get("files")))
    check("F2 second download happened for the new target", download.calls == 2)
    check("F2 per-target cache entries", len(pipe.spiderinfo.downloaded) == 3)

    # F3: identical target deduplication
    d1 = pipe.process_item(file_item("catalog-a"), spider)
    d2 = pipe.process_item(file_item("catalog-a"), spider)
    r1 = yield d1
    r2 = yield d2
    check("F3 identical target needs no new download", download.calls == 2)
    check("F3 identical results reused", r1["files"] == r2["files"])

    # F4: write failure isolation
    original = pipe.store.persist_file

    def flaky(path, buf, info, meta=None, headers=None):
        if path.startswith("catalog-z/"):
            raise OSError("read-only directory")
        return original(path, buf, info, meta=meta, headers=headers)

    pipe.store.persist_file = flaky
    pipe.LOG_FAILED_RESULTS = False
    dx = pipe.process_item(file_item("catalog-x"), spider)
    dz = pipe.process_item(file_item("catalog-z"), spider)
    check("F4 still one shared download", download.calls == 3)
    download.fire()
    rx = yield dx
    rz = yield dz
    check("F4 healthy target delivered",
          [f["path"] for f in rx["files"]] == ["catalog-x/manual.bin"])
    check("F4 failed target uses failure channel (empty files list)", rz["files"] == [])
    check("F4 healthy file exists", exists(store, "catalog-x/manual.bin"))
    check("F4 failed file absent", not exists(store, "catalog-z/manual.bin"))
    failure_entries = [v for v in pipe.spiderinfo.downloaded.values() if isinstance(v, Failure)]
    check("F4 failure cached per target",
          any(isinstance(v.value, FileException) for v in failure_entries))


@inlineCallbacks
def run_images_scenarios(store):
    print("ImagesPipeline")
    body = image_body()
    download = ControlledDownload(body, IMAGE_URL)
    pipe, spider = build_pipeline(store, PerItemImagesPipeline, "IMAGES_STORE", download)
    guid = hashlib.sha1(IMAGE_URL.encode()).hexdigest()  # noqa: S324
    orig = f"full/{guid}.jpg"

    # I4 first: concurrent items share the original, distinct thumbnails
    da = pipe.process_item(image_item("catalog-a"), spider)
    db = pipe.process_item(image_item("catalog-b"), spider)
    check("I4 one download for two concurrent items", download.calls == 1)
    download.fire()
    ra = yield da
    rb = yield db
    check("I4 same original result for both",
          ra["images"][0]["path"] == rb["images"][0]["path"] == orig)
    check("I4 same original checksum",
          ra["images"][0]["checksum"] == rb["images"][0]["checksum"] == _md5sum(io.BytesIO(body)))
    for catalog in ("catalog-a", "catalog-b"):
        p = Path(store, catalog, "thumbs", "small.jpg")
        ok = p.exists()
        if ok:
            try:
                with Image.open(p) as im:
                    im.load()
                    ok = max(im.size) <= 16
            except OSError:
                ok = False
        check(f"I4 thumbnail readable for {catalog}", ok)
    check("I4 two per-target cache entries", len(pipe.spiderinfo.downloaded) == 2)

    # I1: full uptodate in a brand new crawl/pipeline instance
    pipe2, spider2 = build_pipeline(store, PerItemImagesPipeline, "IMAGES_STORE", download)
    r = yield pipe2.process_item(image_item("catalog-a"), spider2)
    check("I1 no new download when original+thumbnails fresh", download.calls == 1)
    check("I1 uptodate status", r["images"][0]["status"] == "uptodate")

    # I2: fresh original, missing item-specific thumbnail -> re-download.
    # Use a catalog never processed before, so its thumbnail cannot exist
    # while the shared original image does.
    missing_catalog = "catalog-missing"
    pipe3, spider3 = build_pipeline(store, PerItemImagesPipeline, "IMAGES_STORE", download)
    check("I2 precondition: new catalog thumbnail absent, original present",
          not exists(store, f"{missing_catalog}/thumbs/small.jpg") and exists(store, orig))
    before = download.calls
    d = pipe3.process_item(image_item(missing_catalog), spider3)
    download.fire()
    r = yield d
    check("I2 re-downloaded because thumbnail missing", download.calls == before + 1)
    thumb = Path(store, missing_catalog, "thumbs", "small.jpg")
    ok = thumb.exists()
    if ok:
        try:
            with Image.open(thumb) as im:
                im.load()
                ok = max(im.size) <= 16
        except OSError:
            ok = False
    check("I2 new thumbnail generated and readable", ok)

    # I3: fresh original, expired thumbnail -> re-download. Dedicated catalog
    # delivered first, then only its thumbnail is aged.
    expired_catalog = "catalog-expired"
    pipe_setup, spider_setup = build_pipeline(store, PerItemImagesPipeline, "IMAGES_STORE", download)
    d = pipe_setup.process_item(image_item(expired_catalog), spider_setup)
    download.fire()
    yield d
    old = time.time() - (pipe.expires + 1) * 86400
    age(store, f"{expired_catalog}/thumbs/small.jpg", old)
    pipe4, spider4 = build_pipeline(store, PerItemImagesPipeline, "IMAGES_STORE", download)
    before = download.calls
    d = pipe4.process_item(image_item(expired_catalog), spider4)
    download.fire()
    r = yield d
    check("I3 re-downloaded because thumbnail expired", download.calls == before + 1)
    check("I3 regenerated with downloaded status", r["images"][0]["status"] == "downloaded")

    # I5: thumbnail write failure isolation
    original = pipe.store.persist_file

    def flaky(path, buf, info, meta=None, headers=None):
        if path.startswith("catalog-z/"):
            raise OSError("cannot create thumbs")
        return original(path, buf, info, meta=meta, headers=headers)

    pipe.store.persist_file = flaky
    pipe.LOG_FAILED_RESULTS = False
    dx = pipe.process_item(image_item("catalog-x"), spider)
    dz = pipe.process_item(image_item("catalog-z"), spider)
    download.fire()
    rx = yield dx
    rz = yield dz
    check("I5 healthy item delivered original", rx["images"][0]["path"] == orig)
    check("I5 healthy item got its thumbnail", exists(store, "catalog-x/thumbs/small.jpg"))
    check("I5 failed item has no images result", rz["images"] == [])
    check("I5 failed thumbnail absent", not exists(store, "catalog-z/thumbs/small.jpg"))


@inlineCallbacks
def main():
    with tempfile.TemporaryDirectory(prefix="media-repro-files-") as files_store, \
            tempfile.TemporaryDirectory(prefix="media-repro-images-") as images_store:
        yield run_files_scenarios(files_store)
        yield run_images_scenarios(images_store)
    print(f"\n{RESULTS['passed']} passed, {RESULTS['failed']} failed")


if __name__ == "__main__":
    from twisted.internet import reactor

    def _done(_):
        reactor.stop()

    def _error(failure):
        print(failure.getTraceback(), file=sys.stderr)
        reactor.stop()

    main().addCallbacks(_done, _error)
    reactor.run()
    if RESULTS["failed"]:
        raise SystemExit(1)
