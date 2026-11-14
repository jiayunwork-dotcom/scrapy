from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, ClassVar

import pytest
from twisted.internet.defer import CancelledError as TwistedCancelledError
from twisted.internet.defer import Deferred

from scrapy import Request, Spider, signals
from scrapy.core.itemkeys import _ItemScheduler, _validate_item_key
from scrapy.core.scraper import Scraper
from scrapy.exceptions import DropItem, ScrapyDeprecationWarning
from scrapy.pipelines import ItemPipelineManager
from scrapy.utils.asyncio import call_later, is_asyncio_available, sleep
from scrapy.utils.defer import deferred_from_coro, maybe_deferred_to_future
from scrapy.utils.spider import DefaultSpider
from scrapy.utils.test import get_crawler
from tests.utils.decorators import coroutine_test

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Coroutine

# The cancellation exception raised into a coroutine depends on the running
# mode: asyncio.CancelledError with an asyncio loop, the Twisted one otherwise.
_CANCEL_ERRORS = (asyncio.CancelledError, TwistedCancelledError)


def _spawn(coro: Coroutine[Any, Any, Any]) -> Deferred[Any]:
    """Schedule *coro* in a way that supports cancellation in every mode."""
    return deferred_from_coro(coro)


async def _wait(deferred: Deferred[Any]) -> Any:
    """Await a Deferred from a coroutine in every reactor mode."""
    return await maybe_deferred_to_future(deferred)


def _business_key(item: Any, spider: Spider) -> Any:
    return item["key"]


class _Barrier:
    """A one-shot barrier that works under the asyncio reactor, the
    non-asyncio reactor and the reactorless mode."""

    def __init__(self) -> None:
        self._deferred: Deferred[None] = Deferred()

    def set(self) -> None:
        if not self._deferred.called:
            self._deferred.callback(None)

    async def wait(self) -> None:
        await maybe_deferred_to_future(self._deferred)


class _RecordingPipeline:
    """Pipeline that records a full trace and can block on barriers."""

    trace: ClassVar[list[tuple[str, int]]] = []
    enter_barrier: ClassVar[_Barrier | None] = None
    release_barrier: ClassVar[_Barrier | None] = None

    async def process_item(self, item: Any) -> Any:
        self.trace.append(("pipeline-enter", item["id"]))
        if self.enter_barrier is not None and item["id"] == 1:
            self.enter_barrier.set()
        if self.release_barrier is not None and item["id"] == 1:
            await self.release_barrier.wait()
        else:
            # give other items a chance to enter the pipeline, so that items
            # with a free key genuinely overlap instead of running back to back
            await sleep(0.02)
        self.trace.append(("pipeline-leave", item["id"]))
        return item


class _KeyedSpider(Spider):
    name = "keyed"
    start_items: ClassVar[list[Any]]

    async def start(self) -> AsyncIterator[Any]:
        for item in self.start_items:
            yield item


def _make_crawler(settings: dict[str, Any] | None = None) -> Any:
    base_settings: dict[str, Any] = {"ITEM_PIPELINES": {_RecordingPipeline: 1}}
    if settings:
        base_settings.update(settings)
    return get_crawler(_KeyedSpider, base_settings)


class TestItemKeyValidation:
    @pytest.mark.parametrize(
        "value",
        [None, "a", 0, 1, -5, ("a",), (1,), ("a", 1), ("a", "b", 3)],
    )
    def test_valid(self, value: Any) -> None:
        _validate_item_key(value)

    @pytest.mark.parametrize(
        "value",
        [True, False, 1.5, b"a", (), ("a", True), (1, 2.0), [1, 2], {"a": 1}],
    )
    def test_invalid(self, value: Any) -> None:
        with pytest.raises(TypeError):
            _validate_item_key(value)


class TestSettingsValidation:
    @coroutine_test
    async def test_non_callable_key_raises_value_error(self) -> None:
        crawler = get_crawler(DefaultSpider, {"ITEM_PIPELINE_KEY": 123})
        crawler.spider = crawler._create_spider()
        with pytest.raises(ValueError, match="ITEM_PIPELINE_KEY"):
            Scraper(crawler)

    @coroutine_test
    async def test_import_path_key(self) -> None:
        crawler = get_crawler(DefaultSpider, {"ITEM_PIPELINE_KEY": "json.dumps"})
        crawler.spider = crawler._create_spider()
        scraper = Scraper(crawler)
        assert scraper.item_scheduler is not None

    @pytest.mark.parametrize("value", [0, -1, True, False, "5", 1.5])
    @coroutine_test
    async def test_bad_max_active_raises_value_error(self, value: Any) -> None:
        crawler = get_crawler(
            DefaultSpider,
            {
                "ITEM_PIPELINE_KEY": _business_key,
                "ITEM_PIPELINE_MAX_ACTIVE": value,
            },
        )
        crawler.spider = crawler._create_spider()
        with pytest.raises(ValueError, match="ITEM_PIPELINE_MAX_ACTIVE"):
            Scraper(crawler)

    @coroutine_test
    async def test_disabled_by_default(self) -> None:
        crawler = get_crawler(DefaultSpider)
        crawler.spider = crawler._create_spider()
        scraper = Scraper(crawler)
        assert scraper.item_scheduler is None


class TestKeyOrdering:
    @coroutine_test
    async def test_same_key_serial_different_key_parallel(self) -> None:
        _RecordingPipeline.trace = []
        items = [
            {"id": 1, "key": "a"},
            {"id": 2, "key": "a"},
            {"id": 3, "key": "b"},
        ]
        _KeyedSpider.start_items = items
        crawler = _make_crawler(
            {"ITEM_PIPELINE_KEY": _business_key, "ITEM_PIPELINE_MAX_ACTIVE": 10}
        )
        await crawler.crawl_async()
        trace = _RecordingPipeline.trace
        # item 3 (different key) runs while item 1 is in the pipeline
        assert trace.index(("pipeline-enter", 3)) < trace.index(("pipeline-leave", 1))
        # item 2 (same key) only enters after item 1 fully leaves
        assert trace.index(("pipeline-leave", 1)) < trace.index(("pipeline-enter", 2))
        assert crawler.stats.get_value("item_scraped_count") == 3

    @coroutine_test
    async def test_same_key_ordered_across_responses(self) -> None:
        trace: list[tuple[str, int]] = []

        class TracePipeline:
            async def process_item(self, item: Any) -> Any:
                trace.append(("pipeline-enter", item["id"]))
                await sleep(0)
                trace.append(("pipeline-leave", item["id"]))
                return item

        class CrossResponseSpider(Spider):
            name = "cross"

            async def start(self) -> AsyncIterator[Any]:
                yield {"id": 1, "key": "a"}
                yield Request("data:,")

            def parse(self, response: Any) -> Any:
                return [
                    {"id": 2, "key": "a"},
                    {"id": 3, "key": "c"},
                ]

        crawler = get_crawler(
            CrossResponseSpider,
            {
                "ITEM_PIPELINES": {TracePipeline: 1},
                "ITEM_PIPELINE_KEY": _business_key,
                "CONCURRENT_ITEMS": 1,
            },
        )
        await crawler.crawl_async()
        # the start item (1) and the response item (2) share key "a" and are
        # serialized in reception order; item 3 has its own key
        entered = [item_id for name, item_id in trace if name == "pipeline-enter"]
        assert entered.index(1) < entered.index(2)
        assert trace.index(("pipeline-leave", 1)) < trace.index(("pipeline-enter", 2))

    @coroutine_test
    async def test_same_key_from_start_ordered(self) -> None:
        _RecordingPipeline.trace = []
        _KeyedSpider.start_items = [
            {"id": 1, "key": "a"},
            {"id": 2, "key": "a"},
        ]
        crawler = _make_crawler(
            {"ITEM_PIPELINE_KEY": _business_key, "CONCURRENT_ITEMS": 1}
        )
        await crawler.crawl_async()
        trace = _RecordingPipeline.trace
        assert trace == [
            ("pipeline-enter", 1),
            ("pipeline-leave", 1),
            ("pipeline-enter", 2),
            ("pipeline-leave", 2),
        ]

    @coroutine_test
    async def test_async_signal_terminal_is_inside_the_key(self) -> None:
        trace: list[tuple[str, int]] = []

        class TracePipeline:
            async def process_item(self, item: Any) -> Any:
                trace.append(("pipeline", item["id"]))
                return item

        async def receiver(item: Any, response: Any, spider: Spider) -> None:
            trace.append(("signal-enter", item["id"]))
            await sleep(0.02)
            trace.append(("signal-leave", item["id"]))

        _KeyedSpider.start_items = [{"id": 1, "key": "a"}, {"id": 2, "key": "a"}]
        crawler = get_crawler(
            _KeyedSpider,
            {
                "ITEM_PIPELINES": {TracePipeline: 1},
                "ITEM_PIPELINE_KEY": _business_key,
                "CONCURRENT_ITEMS": 1,
            },
        )
        crawler.signals.connect(receiver, signals.item_scraped)
        await crawler.crawl_async()
        # the next same-key item cannot enter the pipeline before the previous
        # item's asynchronous terminal signal receiver finished
        assert trace.index(("signal-leave", 1)) < trace.index(("pipeline", 2))

    @coroutine_test
    async def test_key_is_fixed_before_pipeline_changes(self) -> None:
        trace: list[tuple[str, int]] = []

        class ReplacingPipeline:
            async def process_item(self, item: Any) -> Any:
                trace.append(("pipeline-enter", item["id"]))
                item["key"] = "changed"
                replaced = {"id": item["id"], "key": "other", "replaced": True}
                await sleep(0)
                trace.append(("pipeline-leave", item["id"]))
                return replaced

        _KeyedSpider.start_items = [
            {"id": 1, "key": "a"},
            {"id": 2, "key": "a"},
        ]
        crawler = get_crawler(
            _KeyedSpider,
            {
                "ITEM_PIPELINES": {ReplacingPipeline: 1},
                "ITEM_PIPELINE_KEY": _business_key,
                "CONCURRENT_ITEMS": 1,
            },
        )
        scraped: list[Any] = []

        def on_scraped(item: Any, response: Any, spider: Spider) -> None:
            scraped.append(item)

        crawler.signals.connect(on_scraped, signals.item_scraped)
        await crawler.crawl_async()
        # both items were grouped by the original key, so they serialize
        assert trace.index(("pipeline-leave", 1)) < trace.index(("pipeline-enter", 2))
        # the replaced item object is what reaches the terminal signal
        assert all(item.get("replaced") for item in scraped)


class TestTerminalOutcomes:
    @coroutine_test
    async def test_three_outcomes_serialize_and_continue(self) -> None:
        trace: list[tuple[str, int]] = []

        class OutcomePipeline:
            async def process_item(self, item: Any) -> Any:
                trace.append(("pipeline-enter", item["id"]))
                await sleep(0)
                if item["id"] == 2:
                    raise DropItem("dropped")
                if item["id"] == 3:
                    raise ValueError("broken")
                trace.append(("pipeline-leave", item["id"]))
                return item

        _KeyedSpider.start_items = [
            {"id": 1, "key": "a"},
            {"id": 2, "key": "a"},
            {"id": 3, "key": "a"},
            {"id": 4, "key": "a"},
        ]
        crawler = get_crawler(
            _KeyedSpider,
            {
                "ITEM_PIPELINES": {OutcomePipeline: 1},
                "ITEM_PIPELINE_KEY": _business_key,
                "CONCURRENT_ITEMS": 1,
            },
        )

        def on_scraped(item: Any, response: Any, spider: Spider) -> None:
            trace.append(("scraped", item["id"]))

        def on_dropped(
            item: Any, response: Any, spider: Spider, exception: Any
        ) -> None:
            trace.append(("dropped", item["id"]))

        def on_error(item: Any, response: Any, spider: Spider, failure: Any) -> None:
            trace.append(("error", item["id"]))

        crawler.signals.connect(on_scraped, signals.item_scraped)
        crawler.signals.connect(on_dropped, signals.item_dropped)
        crawler.signals.connect(on_error, signals.item_error)
        await crawler.crawl_async()
        # every item starts only after the previous one reached a terminal
        assert trace.index(("scraped", 1)) < trace.index(("pipeline-enter", 2))
        assert trace.index(("dropped", 2)) < trace.index(("pipeline-enter", 3))
        assert trace.index(("error", 3)) < trace.index(("pipeline-enter", 4))
        assert ("scraped", 4) in trace
        # DropItem and errors do not retry: every item entered once
        entered = [item_id for name, item_id in trace if name == "pipeline-enter"]
        assert entered == [1, 2, 3, 4]


class TestKeyFailures:
    @coroutine_test
    async def test_key_exception_goes_to_item_error(self) -> None:
        def failing_key(item: Any, spider: Spider) -> Any:
            if item["id"] == 2:
                raise RuntimeError("boom")
            return item["id"]

        ran: list[int] = []

        class Pipeline:
            async def process_item(self, item: Any) -> Any:
                ran.append(item["id"])
                return item

        _KeyedSpider.start_items = [{"id": 1}, {"id": 2}, {"id": 3}]
        crawler = get_crawler(
            _KeyedSpider,
            {"ITEM_PIPELINES": {Pipeline: 1}, "ITEM_PIPELINE_KEY": failing_key},
        )
        errors: list[tuple[int, str]] = []

        def on_error(item: Any, response: Any, spider: Spider, failure: Any) -> None:
            errors.append((item["id"], type(failure.value).__name__))

        crawler.signals.connect(on_error, signals.item_error)
        await crawler.crawl_async()
        assert ran == [1, 3]
        assert (2, "RuntimeError") in errors
        assert crawler.stats.get_value("item_scraped_count") == 2

    @pytest.mark.parametrize("bad_value", [3.14, True, ()])
    @coroutine_test
    async def test_invalid_key_return_goes_to_item_error(self, bad_value: Any) -> None:
        def invalid_key(item: Any, spider: Spider) -> Any:
            return bad_value

        ran: list[int] = []

        class Pipeline:
            async def process_item(self, item: Any) -> Any:
                ran.append(item["id"])
                return item

        _KeyedSpider.start_items = [{"id": 1}]
        crawler = get_crawler(
            _KeyedSpider,
            {"ITEM_PIPELINES": {Pipeline: 1}, "ITEM_PIPELINE_KEY": invalid_key},
        )
        errors: list[str] = []

        def on_error(item: Any, response: Any, spider: Spider, failure: Any) -> None:
            errors.append(type(failure.value).__name__)

        crawler.signals.connect(on_error, signals.item_error)
        await crawler.crawl_async()
        assert ran == []
        assert errors == ["TypeError"]


class TestActiveLimit:
    @coroutine_test
    async def test_active_caps_concurrency_of_independent_keys(self) -> None:
        state = {"running": 0, "max": 0}

        class CappingPipeline:
            async def process_item(self, item: Any) -> Any:
                state["running"] += 1
                state["max"] = max(state["max"], state["running"])
                await sleep(0.01)
                state["running"] -= 1
                return item

        _KeyedSpider.start_items = [{"id": i, "key": f"k{i}"} for i in range(20)]
        crawler = get_crawler(
            _KeyedSpider,
            {
                "ITEM_PIPELINES": {CappingPipeline: 1},
                "ITEM_PIPELINE_KEY": _business_key,
                "ITEM_PIPELINE_MAX_ACTIVE": 3,
            },
        )
        await crawler.crawl_async()
        assert state["max"] == 3
        assert crawler.stats.get_value("item_scraped_count") == 20

    @coroutine_test
    async def test_waiting_item_does_not_count_as_active(self) -> None:
        enter = _Barrier()
        release = _Barrier()
        _RecordingPipeline.trace = []
        _RecordingPipeline.enter_barrier = enter
        _RecordingPipeline.release_barrier = release
        _KeyedSpider.start_items = [
            {"id": 1, "key": "a"},
            {"id": 2, "key": "a"},
            {"id": 3, "key": "b"},
        ]
        crawler = _make_crawler(
            {
                "ITEM_PIPELINE_KEY": _business_key,
                "ITEM_PIPELINE_MAX_ACTIVE": 2,
                # two output workers so item 2 reaches the scheduler while
                # item 1 is still in the pipeline; the shared key then holds
                # item 2 in the scheduler's waiting queue
                "CONCURRENT_ITEMS": 2,
            }
        )
        task = _spawn(crawler.crawl_async())
        await enter.wait()
        scheduler = crawler.engine.scraper.item_scheduler
        assert scheduler is not None
        # let item 2's output worker reach the scheduler and queue on the key
        while not scheduler._waiting:
            await sleep(0)
        # item 1 is active; item 2 waits on the key without taking a slot
        assert scheduler._active_count == 1
        assert [entry.item["id"] for entry in scheduler._waiting] == [2]
        release.set()
        await _wait(task)
        _RecordingPipeline.enter_barrier = None
        _RecordingPipeline.release_barrier = None
        assert crawler.stats.get_value("item_scraped_count") == 3

    @coroutine_test
    async def test_none_keys_run_parallel_but_capped(self) -> None:
        state = {"running": 0, "max": 0}

        class CappingPipeline:
            async def process_item(self, item: Any) -> Any:
                state["running"] += 1
                state["max"] = max(state["max"], state["running"])
                await sleep(0.01)
                state["running"] -= 1
                return item

        def none_key(item: Any, spider: Spider) -> None:
            return None

        _KeyedSpider.start_items = [{"id": i} for i in range(10)]
        crawler = get_crawler(
            _KeyedSpider,
            {
                "ITEM_PIPELINES": {CappingPipeline: 1},
                "ITEM_PIPELINE_KEY": none_key,
                "ITEM_PIPELINE_MAX_ACTIVE": 4,
            },
        )
        await crawler.crawl_async()
        assert state["max"] == 4
        assert crawler.stats.get_value("item_scraped_count") == 10


class TestCancellation:
    @coroutine_test
    async def test_cancel_waiting_item_skips_pipeline_and_signal(self) -> None:
        enter = _Barrier()
        release = _Barrier()
        _RecordingPipeline.trace = []
        _RecordingPipeline.enter_barrier = enter
        _RecordingPipeline.release_barrier = release
        _KeyedSpider.start_items = [
            {"id": 1, "key": "a"},
            {"id": 2, "key": "a"},
            {"id": 3, "key": "a"},
        ]
        crawler = _make_crawler(
            {
                "ITEM_PIPELINE_KEY": _business_key,
                # enough output workers that item 2 reaches the scheduler and
                # waits on the key while item 1 is still in the pipeline
                "CONCURRENT_ITEMS": 3,
            }
        )
        terminal: list[tuple[str, int]] = []

        def on_scraped(item: Any, response: Any, spider: Spider) -> None:
            terminal.append(("scraped", item["id"]))

        crawler.signals.connect(on_scraped, signals.item_scraped)
        task = _spawn(crawler.crawl_async())
        await enter.wait()
        scheduler = crawler.engine.scraper.item_scheduler
        assert scheduler is not None
        # let item 2's output worker reach the scheduler and queue on the key
        while not any(entry.item["id"] == 2 for entry in scheduler._waiting):
            await sleep(0)
        waiting = [entry for entry in scheduler._waiting if entry.item["id"] == 2]
        assert len(waiting) == 1
        crawler.engine.scraper._cancel_item(waiting[0])
        release.set()
        await _wait(task)
        _RecordingPipeline.enter_barrier = None
        _RecordingPipeline.release_barrier = None
        entered = [
            item_id
            for name, item_id in _RecordingPipeline.trace
            if name == "pipeline-enter"
        ]
        assert 2 not in entered
        assert all(item_id != 2 for _, item_id in terminal)
        # the other items are unaffected
        assert entered == [1, 3]
        assert scheduler._pending_count == 0
        assert not scheduler._active_keys
        assert not scheduler._waiting

    @coroutine_test
    async def test_cancel_running_item_keeps_key_until_downstream_ends(
        self,
    ) -> None:
        trace: list[tuple[str, int]] = []
        hold = _Barrier()
        release = _Barrier()

        class SlowPipeline:
            async def process_item(self, item: Any) -> Any:
                trace.append(("pipeline-enter", item["id"]))
                if item["id"] == 1:
                    try:
                        await sleep(10)
                    except _CANCEL_ERRORS:
                        trace.append(("pipeline-cancelled", item["id"]))
                        hold.set()
                        # the downstream write keeps running after the outer
                        # cancellation, holding the key while it finishes
                        await release.wait()
                        trace.append(("pipeline-after-cancel", item["id"]))
                        raise
                trace.append(("pipeline-leave", item["id"]))
                return item

        crawler = get_crawler(
            DefaultSpider,
            {
                "ITEM_PIPELINES": {SlowPipeline: 1},
                "ITEM_PIPELINE_KEY": _business_key,
            },
        )
        crawler.spider = crawler._create_spider()
        scraper = Scraper(crawler)
        await scraper.open_spider_async()
        running = _spawn(
            scraper.start_itemproc_async({"id": 1, "key": "a"}, response=None)
        )
        waiting = _spawn(
            scraper.start_itemproc_async({"id": 2, "key": "a"}, response=None)
        )
        await sleep(0.05)
        running.cancel()
        await hold.wait()
        await sleep(0.02)
        # the next same-key item cannot start while the cancelled item is still
        # finishing downstream
        assert ("pipeline-enter", 2) not in trace
        assert scraper.item_scheduler is not None
        assert scraper.item_scheduler._active_count == 1
        release.set()
        if is_asyncio_available():
            # With an asyncio loop the outer task surfaces the cancellation.
            with pytest.raises(_CANCEL_ERRORS):
                await _wait(running)
        else:
            # A non-asyncio reactor reports the cancellation through the
            # item_error path instead of re-raising it out of the coroutine.
            await _wait(running)
        await _wait(waiting)
        await scraper.close_spider_async()
        assert trace.index(("pipeline-after-cancel", 1)) < trace.index(
            ("pipeline-enter", 2)
        )


class TestCloseAndState:
    @coroutine_test
    async def test_close_waits_for_queued_items(self) -> None:
        scraped: list[int] = []

        class SlowPipeline:
            async def process_item(self, item: Any) -> Any:
                await sleep(0.01)
                scraped.append(item["id"])
                return item

        _KeyedSpider.start_items = [{"id": i, "key": f"k{i % 4}"} for i in range(30)]
        crawler = get_crawler(
            _KeyedSpider,
            {
                "ITEM_PIPELINES": {SlowPipeline: 1},
                "ITEM_PIPELINE_KEY": _business_key,
                "ITEM_PIPELINE_MAX_ACTIVE": 2,
            },
        )
        await crawler.crawl_async()
        assert len(scraped) == 30
        scheduler = crawler.engine.scraper.item_scheduler
        assert scheduler is not None
        assert scheduler._pending_count == 0
        assert not scheduler._active_keys
        assert not scheduler._waiting

    @coroutine_test
    async def test_state_reclaimed_after_many_independent_keys(self) -> None:
        class NoopPipeline:
            async def process_item(self, item: Any) -> Any:
                return item

        _KeyedSpider.start_items = [{"id": i, "key": f"k{i}"} for i in range(300)]
        crawler = get_crawler(
            _KeyedSpider,
            {
                "ITEM_PIPELINES": {NoopPipeline: 1},
                "ITEM_PIPELINE_KEY": _business_key,
                "ITEM_PIPELINE_MAX_ACTIVE": 40,
            },
        )
        await crawler.crawl_async()
        scheduler = crawler.engine.scraper.item_scheduler
        assert scheduler is not None
        assert scheduler._pending_count == 0
        assert not scheduler._active_keys
        assert not scheduler._waiting
        assert crawler.stats.get_value("item_scraped_count") == 300


class TestCrawlerIsolation:
    @staticmethod
    def _make_spider(spider_name: str) -> type[Spider]:
        async def start(self: Any) -> AsyncIterator[Any]:
            for i in range(3):
                yield {"crawler": spider_name, "id": i, "key": "shared"}

        return type(
            f"Spider_{spider_name}",
            (Spider,),
            {"name": spider_name, "start": start},
        )

    @coroutine_test
    async def test_separate_schedulers_and_counts(self) -> None:
        class Pipeline:
            async def process_item(self, item: Any) -> Any:
                await sleep(0.01)
                return item

        crawlers = []
        tasks = []
        for name in ("x", "y"):
            spider_cls = self._make_spider(name)
            crawler = get_crawler(
                spider_cls,
                {
                    "ITEM_PIPELINES": {Pipeline: 1},
                    "ITEM_PIPELINE_KEY": _business_key,
                    "ITEM_PIPELINE_MAX_ACTIVE": 1,
                },
            )
            crawlers.append(crawler)
            tasks.append(_spawn(crawler.crawl_async()))
        for task in tasks:
            await _wait(task)
        schedulers = [c.engine.scraper.item_scheduler for c in crawlers]
        assert schedulers[0] is not schedulers[1]
        for scheduler in schedulers:
            assert isinstance(scheduler, _ItemScheduler)
            assert scheduler._pending_count == 0
        assert sum(c.stats.get_value("item_scraped_count") for c in crawlers) == 6


class TestCustomItemProcessor:
    @coroutine_test
    async def test_custom_processor_remains_compatible(self) -> None:
        trace: list[tuple[str, int]] = []

        class CustomManager(ItemPipelineManager):
            async def process_item_async(self, item: Any) -> Any:
                trace.append(("processor-enter", item["id"]))
                output = await super().process_item_async(item)
                trace.append(("processor-leave", item["id"]))
                return output

        class Pipeline:
            def process_item(self, item: Any) -> Any:
                return item

        _KeyedSpider.start_items = [
            {"id": 1, "key": "a"},
            {"id": 2, "key": "a"},
        ]
        crawler = get_crawler(
            _KeyedSpider,
            {
                "ITEM_PROCESSOR": CustomManager,
                "ITEM_PIPELINES": {Pipeline: 1},
                "ITEM_PIPELINE_KEY": _business_key,
                "CONCURRENT_ITEMS": 1,
            },
        )
        await crawler.crawl_async()
        assert trace.index(("processor-leave", 1)) < trace.index(("processor-enter", 2))
        assert crawler.stats.get_value("item_scraped_count") == 2


class TestDefaultDisabled:
    @coroutine_test
    async def test_behavior_unchanged_when_key_is_none(self) -> None:
        state = {"running": 0, "max": 0}

        class ParallelPipeline:
            async def process_item(self, item: Any) -> Any:
                state["running"] += 1
                state["max"] = max(state["max"], state["running"])
                await sleep(0.01)
                state["running"] -= 1
                return item

        _KeyedSpider.start_items = [{"id": i, "key": f"k{i}"} for i in range(10)]
        crawler = get_crawler(
            _KeyedSpider,
            # no ITEM_PIPELINE_KEY: same-key items must still run in parallel
            {"ITEM_PIPELINES": {ParallelPipeline: 1}},
        )
        await crawler.crawl_async()
        assert state["max"] > 1
        assert crawler.stats.get_value("item_scraped_count") == 10


class TestDeferredPipeline:
    @coroutine_test
    async def test_deferred_pipeline_respects_the_key(self) -> None:
        trace: list[tuple[str, int]] = []

        class DeferredPipeline:
            def process_item(self, item: Any) -> Deferred[Any]:
                trace.append(("pipeline-enter", item["id"]))
                d: Deferred[Any] = Deferred()

                def fire() -> None:
                    trace.append(("pipeline-leave", item["id"]))
                    d.callback(item)

                call_later(0.02, fire)
                return d

        _KeyedSpider.start_items = [
            {"id": 1, "key": "a"},
            {"id": 2, "key": "a"},
        ]
        crawler = get_crawler(
            _KeyedSpider,
            {
                "ITEM_PIPELINES": {DeferredPipeline: 1},
                "ITEM_PIPELINE_KEY": _business_key,
                "CONCURRENT_ITEMS": 1,
            },
        )
        with pytest.warns(ScrapyDeprecationWarning, match="returned a Deferred"):
            await crawler.crawl_async()
        assert trace.index(("pipeline-leave", 1)) < trace.index(("pipeline-enter", 2))
