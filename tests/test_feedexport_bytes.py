"""Tests for byte-based feed batching (FEED_EXPORT_BATCH_BYTE_COUNT)."""

from __future__ import annotations

import gzip
import json
import tempfile
from pathlib import Path

import pytest
from twisted.internet import defer, reactor
from twisted.internet.task import deferLater
from twisted.trial import unittest

import scrapy
from scrapy import signals
from scrapy.exceptions import NotConfigured
from scrapy.extensions.feedexport import FeedExporter
from scrapy.extensions.postprocessing import PostProcessingManager
from scrapy.utils.defer import deferred_from_coro
from scrapy.utils.test import get_crawler


class _Spider(scrapy.Spider):
    name = "byte_batching_spider"

    def start_requests(self):
        return []

    def parse(self, response):
        pass


class WriteOnlySink:
    """Target that only supports ``write``/``flush`` (no tell/seek)."""

    def __init__(self):
        self.chunks: list[bytes] = []

    def write(self, data: bytes) -> int:
        self.chunks.append(bytes(data))
        return len(data)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass

    def getvalue(self) -> bytes:
        return b"".join(self.chunks)


class RecordingStorage:
    """Storage that exposes the exact bytes given to ``store``."""

    instances: list[RecordingStorage] = []

    def __init__(self, uri: str, *, feed_options=None):
        self.uri = uri
        self.data: bytes | None = None
        self.file = None
        self.store_deferred: defer.Deferred | None = None
        type(self).instances.append(self)

    def open(self, spider):
        self.file = tempfile.TemporaryFile()
        return self.file

    def store(self, file):
        file.seek(0)
        self.data = file.read()
        file.close()


class ControlledStorage(RecordingStorage):
    """Storage whose ``store`` Deferred is resolved manually, in any order."""

    def store(self, file):
        file.seek(0)
        self.data = file.read()
        self.store_deferred = defer.Deferred()
        return self.store_deferred


class ByteBatchingTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        RecordingStorage.instances = []
        ControlledStorage.instances = []

    def tearDown(self):
        for path in self.temp_dir.iterdir():
            path.unlink()
        self.temp_dir.rmdir()
        for storage in RecordingStorage.instances + ControlledStorage.instances:
            if storage.file is not None and not storage.file.closed:
                storage.file.close()

    def _make_exporter(self, settings_dict, spider_cls=_Spider):
        settings_dict = dict(settings_dict)
        settings_dict.setdefault("FEED_STORAGES", {})
        crawler = get_crawler(spider_cls, settings_dict)
        exporter = FeedExporter.from_crawler(crawler)
        spider = spider_cls.from_crawler(crawler)
        # Track every slot created, including those closed by a rotation.
        created: list = []
        original = exporter._start_new_batch

        def recording_start_new_batch(*args, **kwargs):
            slot = original(*args, **kwargs)
            created.append(slot)
            return slot

        exporter._start_new_batch = recording_start_new_batch
        exporter.created_slots = created
        exporter.open_spider(spider)
        return exporter, spider

    @defer.inlineCallbacks
    def run_feed(self, feeds, items, settings=None):
        exporter, spider = self._make_exporter({"FEEDS": feeds, **(settings or {})})
        for item in items:
            exporter.item_scraped(item, spider)
        yield deferred_from_coro(exporter.close_spider(spider))
        return exporter, exporter.created_slots

    def _files(self, fmt="json"):
        result = {}
        for path in sorted(self.temp_dir.iterdir()):
            result[path.name] = path.read_bytes()
        return result


class TestByteBatching(ByteBatchingTestCase):
    @defer.inlineCallbacks
    def test_json_exact_limit(self):
        # [{"a": "xxxxxxxx"}] is 19 bytes.
        feeds = {
            str(self.temp_dir / "out-%(batch_id)d.json"): {
                "format": "json",
                "batch_byte_count": 19,
            }
        }
        _exporter, slots = yield self.run_feed(
            feeds, [{"a": "x" * 8}], {"FEED_EXPORT_INDENT": None}
        )
        files = self._files()
        assert list(files) == ["out-1.json"]
        assert files["out-1.json"] == b'[{"a": "xxxxxxxx"}]'
        assert slots[0].serialized_bytes == 19
        assert slots[0].stored_bytes == 19
        assert not slots[0].oversized

    @defer.inlineCallbacks
    def test_json_one_byte_below_limit_is_oversized(self):
        feeds = {
            str(self.temp_dir / "out-%(batch_id)d.json"): {
                "format": "json",
                "batch_byte_count": 18,
            }
        }
        _exporter, slots = yield self.run_feed(
            feeds, [{"a": "x" * 8}], {"FEED_EXPORT_INDENT": None}
        )
        files = self._files()
        assert files == {"out-1.json": b'[{"a": "xxxxxxxx"}]'}
        assert slots[0].oversized
        assert slots[0].serialized_bytes == 19

    @defer.inlineCallbacks
    def test_json_rotation_and_indent(self):
        items = [{"a": "x"}, {"a": "y"}, {"a": "z"}]
        indent = 2
        # Every item serialized with indent=2 is 21 bytes; the framing grows.
        feeds = {
            str(self.temp_dir / "out-%(batch_id)d.json"): {
                "format": "json",
                "batch_byte_count": 30,
                "indent": indent,
            }
        }
        _exporter, slots = yield self.run_feed(
            feeds, items, {"FEED_EXPORT_INDENT": None}
        )
        files = self._files()
        # Two items fit in 30 bytes only without the closing framing; verify
        # every file is a valid JSON list with the expected order.
        seen = []
        for name in sorted(files):
            data = files[name]
            assert len(data) <= 30 or len(json.loads(data)) == 1
            seen.extend(json.loads(data))
        assert seen == items
        for slot in slots:
            assert slot.serialized_bytes == slot.stored_bytes
            assert slot.serialized_bytes is not None
            assert slot.serialized_bytes <= 30

    @defer.inlineCallbacks
    def test_count_and_byte_limits_combined(self):
        items = [{"a": str(i)} for i in range(5)]
        feeds = {
            str(self.temp_dir / "out-%(batch_id)d.jsonlines"): {
                "format": "jsonlines",
                "batch_byte_count": 100,
                "batch_item_count": 2,
            }
        }
        _exporter, slots = yield self.run_feed(
            feeds, items, {"FEED_EXPORT_INDENT": None}
        )
        files = self._files()
        assert [len(v.splitlines()) for _, v in sorted(files.items())] == [2, 2, 1]
        counts = [slot.itemcount for slot in slots]
        assert counts == [2, 2, 1]

    @defer.inlineCallbacks
    def test_byte_limit_rotates(self):
        items = [{"a": "1"}, {"a": "2"}, {"a": "3"}]
        # Each jsonlines record is 11 bytes: '{"a": "1"}\n'
        feeds = {
            str(self.temp_dir / "out-%(batch_id)d.jsonlines"): {
                "format": "jsonlines",
                "batch_byte_count": 22,
            }
        }
        _exporter, slots = yield self.run_feed(feeds, items)
        files = self._files()
        assert files == {
            "out-1.jsonlines": b'{"a": "1"}\n{"a": "2"}\n',
            "out-2.jsonlines": b'{"a": "3"}\n',
        }
        assert [slot.itemcount for slot in slots] == [2, 1]

    @defer.inlineCallbacks
    def test_oversized_item_gets_its_own_file(self):
        items = [{"a": "x" * 20}, {"a": "y"}, {"a": "z"}]
        feeds = {
            str(self.temp_dir / "out-%(batch_id)d.json"): {
                "format": "json",
                "batch_byte_count": 20,
            }
        }
        _exporter, slots = yield self.run_feed(
            feeds, items, {"FEED_EXPORT_INDENT": None}
        )
        files = self._files()
        assert list(files) == ["out-1.json", "out-2.json", "out-3.json"]
        assert files["out-1.json"] == b'[{"a": "xxxxxxxxxxxxxxxxxxxx"}]'
        assert [slot.itemcount for slot in slots] == [1, 1, 1]
        assert [slot.oversized for slot in slots] == [True, False, False]
        assert slots[0].serialized_bytes == len(files["out-1.json"])

    @defer.inlineCallbacks
    def test_multibyte_text(self):
        items = [{"a": "日本語"}, {"a": "テスト"}]
        feeds = {
            str(self.temp_dir / "out-%(batch_id)d.jsonlines"): {
                "format": "jsonlines",
                "batch_byte_count": 20,
            }
        }
        _exporter, slots = yield self.run_feed(
            feeds, items, {"FEED_EXPORT_ENCODING": "utf-8"}
        )
        files = self._files()
        assert len(files) == 2
        for data in files.values():
            assert len(data) <= 20
            assert json.loads(data) in items
        for slot in slots:
            assert slot.serialized_bytes == len(data)
            assert slot.serialized_bytes == slot.stored_bytes

    @defer.inlineCallbacks
    def test_csv_quoting_and_multiline(self):
        items = [{"a": 'x,"q"\ny', "b": "é"}, {"a": "2", "b": "3"}]
        feeds = {
            str(self.temp_dir / "out-%(batch_id)d.csv"): {
                "format": "csv",
                "batch_byte_count": 1000,
            }
        }
        _exporter, slots = yield self.run_feed(feeds, items)
        files = self._files("csv")
        assert files == {"out-1.csv": b'a,b\r\n"x,""q""\ny",\xc3\xa9\r\n2,3\r\n'}
        assert slots[0].serialized_bytes == len(files["out-1.csv"])

    @defer.inlineCallbacks
    def test_csv_fields_pinned_across_batches(self):
        items = [{"b": "1", "a": "2"}, {"a": "x"}, {"b": "y", "c": "z"}]
        feeds = {
            str(self.temp_dir / "out-%(batch_id)d.csv"): {
                "format": "csv",
                "batch_byte_count": 10,
            }
        }
        _exporter, _slots = yield self.run_feed(feeds, items)
        files = self._files("csv")
        assert files == {
            "out-1.csv": b"b,a\r\n1,2\r\n",
            "out-2.csv": b"b,a\r\n,x\r\n",
            "out-3.csv": b"b,a\r\ny,\r\n",
        }

    @defer.inlineCallbacks
    def test_csv_configured_fields(self):
        items = [{"b": "1", "a": "2"}]
        feeds = {
            str(self.temp_dir / "out-%(batch_id)d.csv"): {
                "format": "csv",
                "batch_byte_count": 100,
                "fields": ["a"],
            }
        }
        _exporter, _slots = yield self.run_feed(feeds, items)
        assert self._files("csv") == {"out-1.csv": b"a\r\n2\r\n"}

    @defer.inlineCallbacks
    def test_feeds_are_isolated(self):
        class ItemA(scrapy.Item):
            a = scrapy.Field()
            b = scrapy.Field()

        class ItemB(scrapy.Item):
            b = scrapy.Field()
            c = scrapy.Field()

        class OnlyA(scrapy.extensions.feedexport.ItemFilter):
            def accepts(self, item):
                return isinstance(item, ItemA)

        class OnlyB(scrapy.extensions.feedexport.ItemFilter):
            def accepts(self, item):
                return isinstance(item, ItemB)

        items = [ItemA(a="1", b="2"), ItemB(b="x", c="y"), ItemA(a="3", b="4")]
        feeds = {
            str(self.temp_dir / "A-%(batch_id)02d.csv"): {
                "format": "csv",
                "batch_byte_count": 12,
                "item_filter": OnlyA,
            },
            str(self.temp_dir / "B-%(batch_id)02d.csv"): {
                "format": "csv",
                "batch_byte_count": 12,
                "item_filter": OnlyB,
            },
        }
        _exporter, slots = yield self.run_feed(feeds, items)
        files = self._files()
        assert files == {
            "A-01.csv": b"a,b\r\n1,2\r\n",
            "A-02.csv": b"a,b\r\n3,4\r\n",
            "B-01.csv": b"b,c\r\nx,y\r\n",
        }
        assert sorted(slot.itemcount for slot in slots) == [1, 1, 1]

    @defer.inlineCallbacks
    def test_empty_feed_oversized(self):
        feeds = {
            str(self.temp_dir / "out-%(batch_id)d.json"): {
                "format": "json",
                "batch_byte_count": 1,
                "store_empty": True,
            }
        }
        _exporter, slots = yield self.run_feed(feeds, [], {"FEED_EXPORT_INDENT": None})
        files = self._files()
        assert files == {"out-1.json": b"[]"}
        assert slots[0].oversized
        assert slots[0].serialized_bytes == 2

    @defer.inlineCallbacks
    def test_empty_feed_not_stored(self):
        feeds = {
            str(self.temp_dir / "out-%(batch_id)d.json"): {
                "format": "json",
                "batch_byte_count": 10,
                "store_empty": False,
            }
        }
        _exporter, slots = yield self.run_feed(feeds, [])
        assert self._files() == {}
        assert slots[0].serialized_bytes is None

    @defer.inlineCallbacks
    def test_serializer_called_once_and_item_not_mutated(self):
        calls = []

        class Item(scrapy.Item):
            a = scrapy.Field(
                serializer=lambda value: (calls.append(value), value.upper())[1]
            )

        items = [Item(a="x"), Item(a="y"), Item(a="z")]
        snapshots = [dict(item) for item in items]
        feeds = {
            str(self.temp_dir / "out-%(batch_id)d.jsonlines"): {
                "format": "jsonlines",
                "batch_byte_count": 12,
            }
        }
        _exporter, _slots = yield self.run_feed(feeds, items)
        assert calls == ["x", "y", "z"]
        assert [dict(item) for item in items] == snapshots
        files = self._files()
        assert [json.loads(data) for data in files.values()] == [
            {"a": "X"},
            {"a": "Y"},
            {"a": "Z"},
        ]

    @defer.inlineCallbacks
    def test_gzip_stored_bytes(self):
        items = [{"a": "1"}, {"a": "2"}, {"a": "3"}]
        feeds = {
            str(self.temp_dir / "out-%(batch_id)d.jsonlines"): {
                "format": "jsonlines",
                "batch_byte_count": 20,
                "postprocessing": [
                    "scrapy.extensions.postprocessing.GzipPlugin",
                ],
                "gzip_mtime": 0,
                "gzip_filename": "",
            }
        }
        _exporter, slots = yield self.run_feed(feeds, items)
        files = self._files()
        assert len(files) == 3
        records = []
        for name in sorted(files):
            raw = files[name]
            decompressed = gzip.decompress(raw)
            assert decompressed in (b'{"a": "1"}\n', b'{"a": "2"}\n', b'{"a": "3"}\n')
            records.append(json.loads(decompressed))
        assert records == items
        for slot in slots:
            assert slot.serialized_bytes == len(
                gzip.decompress(files[slot.uri.rsplit("/", 1)[-1]])
            )
        # stored_bytes matches the compressed file, not the serialized one.
        assert [slot.stored_bytes for slot in slots] == [
            len(files[name]) for name in sorted(files)
        ]
        assert slots[0].serialized_bytes != slots[0].stored_bytes

    @defer.inlineCallbacks
    def test_no_postprocessing_sizes_equal(self):
        items = [{"a": "1"}]
        feeds = {
            str(self.temp_dir / "out-%(batch_id)d.json"): {
                "format": "json",
                "batch_byte_count": 100,
            }
        }
        _exporter, slots = yield self.run_feed(feeds, items)
        slot = slots[0]
        assert slot.serialized_bytes == slot.stored_bytes
        assert slot.stored_bytes == len(self._files()["out-1.json"])

    @defer.inlineCallbacks
    def test_byte_limit_zero_keeps_old_behaviour(self):
        items = [{"a": "1"}, {"a": "2"}]
        feeds = {
            str(self.temp_dir / "out.json"): {"format": "json", "batch_byte_count": 0}
        }
        _exporter, slots = yield self.run_feed(
            feeds, items, {"FEED_EXPORT_INDENT": None}
        )
        expected = b'[{"a": "1"},{"a": "2"}]'
        assert self._files() == {"out.json": expected}
        assert slots[0].stored_bytes == len(expected)
        assert slots[0].serialized_bytes == len(expected)


class TestByteBatchingConfiguration:
    def _exporter(self, feeds, **settings):
        crawler = get_crawler(_Spider, {"FEEDS": feeds, **settings})
        return FeedExporter.from_crawler(crawler)

    def test_setting_fallback(self):
        exporter = self._exporter(
            {"file:///tmp/out-%(batch_id)d.json": {"format": "json"}},
            FEED_EXPORT_BATCH_BYTE_COUNT=10,
        )
        assert (
            exporter.feeds["file:///tmp/out-%(batch_id)d.json"]["batch_byte_count"]
            == 10
        )

    def test_feed_option_overrides_setting(self):
        exporter = self._exporter(
            {
                "file:///tmp/out-%(batch_id)d.json": {
                    "format": "json",
                    "batch_byte_count": 5,
                }
            },
            FEED_EXPORT_BATCH_BYTE_COUNT=10,
        )
        assert (
            exporter.feeds["file:///tmp/out-%(batch_id)d.json"]["batch_byte_count"] == 5
        )

    @pytest.mark.parametrize("value", [True, False, -1, 1.5, "10", None])
    def test_invalid_feed_option(self, value):
        with pytest.raises(NotConfigured):
            self._exporter(
                {
                    "file:///tmp/out-%(batch_id)d.json": {
                        "format": "json",
                        "batch_byte_count": value,
                    }
                }
            )

    @pytest.mark.parametrize("value", [-1, 1.5, "10"])
    def test_invalid_setting(self, value):
        with pytest.raises(NotConfigured):
            self._exporter(
                {"file:///tmp/out-%(batch_id)d.json": {"format": "json"}},
                FEED_EXPORT_BATCH_BYTE_COUNT=value,
            )

    def test_setting_bool_rejected(self):
        with pytest.raises(NotConfigured):
            self._exporter(
                {"file:///tmp/out-%(batch_id)d.json": {"format": "json"}},
                FEED_EXPORT_BATCH_BYTE_COUNT=True,
            )

    def test_missing_batch_placeholder(self):
        with pytest.raises(NotConfigured):
            self._exporter(
                {
                    "file:///tmp/out.json": {
                        "format": "json",
                        "batch_byte_count": 10,
                    }
                }
            )

    @pytest.mark.parametrize("format", ["xml", "marshal", "pickle"])
    def test_unsupported_exporter(self, format):
        with pytest.raises(NotConfigured):
            self._exporter(
                {
                    "file:///tmp/out-%(batch_id)d.out": {
                        "format": format,
                        "batch_byte_count": 10,
                    }
                }
            )

    @pytest.mark.parametrize(
        "postprocessing",
        [
            ["scrapy.extensions.postprocessing.Bz2Plugin"],
            ["scrapy.extensions.postprocessing.LZMAPlugin"],
            [
                "scrapy.extensions.postprocessing.GzipPlugin",
                "scrapy.extensions.postprocessing.Bz2Plugin",
            ],
        ],
    )
    def test_unsupported_postprocessing(self, postprocessing):
        with pytest.raises(NotConfigured):
            self._exporter(
                {
                    "file:///tmp/out-%(batch_id)d.jsonlines": {
                        "format": "jsonlines",
                        "batch_byte_count": 10,
                        "postprocessing": postprocessing,
                    }
                }
            )

    def test_gzip_supported(self):
        exporter = self._exporter(
            {
                "file:///tmp/out-%(batch_id)d.jsonlines": {
                    "format": "jsonlines",
                    "batch_byte_count": 10,
                    "postprocessing": [
                        "scrapy.extensions.postprocessing.GzipPlugin",
                    ],
                }
            }
        )
        assert exporter.feeds

    def test_disabled_allows_other_formats_and_plugins(self):
        exporter = self._exporter(
            {
                "file:///tmp/out.xml": {"format": "xml"},
                "file:///tmp/out.json": {
                    "format": "json",
                    "postprocessing": [
                        "scrapy.extensions.postprocessing.LZMAPlugin",
                    ],
                },
            }
        )
        assert exporter.feeds


class TestPostProcessingByteAccounting(unittest.TestCase):
    def test_write_only_target_and_gzip(self):
        sink = WriteOnlySink()
        manager = PostProcessingManager(
            ["scrapy.extensions.postprocessing.GzipPlugin"],
            sink,
            {"gzip_mtime": 0, "gzip_filename": ""},
        )
        payload = b"hello" * 100
        manager.write(payload)
        assert manager.input_bytes == len(payload)
        assert manager.output_bytes == len(sink.getvalue())
        manager.close()
        # The gzip header and trailer were written while closing.
        assert manager.output_bytes == len(sink.getvalue())
        assert gzip.decompress(sink.getvalue()) == payload

    def test_no_plugins(self):
        sink = WriteOnlySink()
        manager = PostProcessingManager([], sink, {})
        manager.write(b"abc")
        manager.close()
        assert manager.input_bytes == 3
        assert manager.output_bytes == 3

    def test_tell_does_not_need_seek(self):
        sink = WriteOnlySink()
        manager = PostProcessingManager([], sink, {})
        manager.write(b"abc")
        assert manager.tell() == 3


class TestStoreCompletionOrdering(ByteBatchingTestCase):
    def _setup(self, feeds):
        settings = {
            "FEEDS": {
                str(self.temp_dir / name): options for name, options in feeds.items()
            },
            "FEED_STORAGES": {"": ControlledStorage, "file": ControlledStorage},
        }
        return self._make_exporter(settings)

    @defer.inlineCallbacks
    def test_signals_after_store_and_ordered_completion(self):
        feeds = {
            "a-%(batch_id)d.jsonlines": {
                "format": "jsonlines",
                "batch_byte_count": 10,
            },
            "b-%(batch_id)d.jsonlines": {
                "format": "jsonlines",
                "batch_byte_count": 10,
            },
        }
        exporter, spider = self._setup(feeds)
        order = []
        crawler = exporter.crawler

        def on_slot(slot, **kwargs):
            order.append(("slot", slot.uri))

        def on_exporter(**kwargs):
            order.append(("exporter",))

        crawler.signals.connect(on_slot, signal=signals.feed_slot_closed)
        crawler.signals.connect(on_exporter, signal=signals.feed_exporter_closed)

        for item in [{"a": "1"}, {"a": "2"}]:
            exporter.item_scraped(item, spider)

        close_d = deferred_from_coro(exporter.close_spider(spider))
        yield deferLater(reactor, 0, lambda: None)
        stores = [s for s in ControlledStorage.instances if s.store_deferred]
        assert len(stores) == 4  # two batches per feed
        assert order == []
        assert (
            crawler.stats.get_value("feedexport/success_count/ControlledStorage")
            is None
        )

        # Resolve the last store first.
        stores[-1].store_deferred.callback(None)
        assert order == [("slot", stores[-1].uri)]

        # Fail one feed; the other feeds must still finish.
        stores[0].store_deferred.errback(ValueError("boom"))
        assert ("slot", stores[0].uri) in order
        assert crawler.stats.get_value("feedexport/failed_count/ControlledStorage") == 1

        for store in stores[1:-1]:
            store.store_deferred.callback(None)
        yield close_d

        assert order[-1] == ("exporter",)
        assert len(order) == 5
        assert (
            crawler.stats.get_value("feedexport/success_count/ControlledStorage") == 3
        )
        assert crawler.stats.get_value("feedexport/failed_count/ControlledStorage") == 1
        for store in stores:
            assert store.data is not None

    @defer.inlineCallbacks
    def test_slot_signal_handlers_are_awaited(self):
        feeds = {
            "a-%(batch_id)d.jsonlines": {
                "format": "jsonlines",
                "batch_byte_count": 10,
            }
        }
        exporter, spider = self._setup(feeds)
        order = []

        def on_slot(slot, **kwargs):
            d = defer.Deferred()
            d.addCallback(lambda _: order.append("slot-done"))
            order.append("slot-called")
            d.callback(None)
            return d

        def on_exporter(**kwargs):
            order.append("exporter")

        exporter.crawler.signals.connect(on_slot, signal=signals.feed_slot_closed)
        exporter.crawler.signals.connect(
            on_exporter, signal=signals.feed_exporter_closed
        )

        exporter.item_scraped({"a": "1"}, spider)
        close_d = deferred_from_coro(exporter.close_spider(spider))
        yield deferLater(reactor, 0, lambda: None)
        store = next(s for s in ControlledStorage.instances if s.store_deferred)
        assert order == []
        store.store_deferred.callback(None)
        yield close_d
        assert order == ["slot-called", "slot-done", "exporter"]

    @defer.inlineCallbacks
    def test_store_receives_raw_file_not_temporary_leaks(self):
        feeds = {"a-%(batch_id)d.json": {"format": "json", "batch_byte_count": 100}}
        exporter, spider = self._setup(feeds)
        exporter.item_scraped({"a": "1"}, spider)
        close_d = deferred_from_coro(exporter.close_spider(spider))
        yield deferLater(reactor, 0, lambda: None)
        store = next(s for s in ControlledStorage.instances if s.store_deferred)
        # Bytes are known before the store finishes, but success is not
        # reported yet.
        slot = exporter.slots[0]
        assert slot.serialized_bytes is not None
        assert crawler_success(exporter) is None
        store.store_deferred.callback(None)
        yield close_d
        assert store.data == b'[\n{"a": "1"}\n]'
        assert crawler_success(exporter) == 1


def crawler_success(exporter):
    return exporter.crawler.stats.get_value(
        "feedexport/success_count/ControlledStorage"
    )
