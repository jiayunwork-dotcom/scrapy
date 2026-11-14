from __future__ import annotations

import asyncio
from collections import deque
from typing import TYPE_CHECKING, Any, TypeAlias

from twisted.internet.defer import CancelledError, Deferred

from scrapy.utils.defer import maybe_deferred_to_future

if TYPE_CHECKING:
    from collections.abc import Callable


ItemKey: TypeAlias = tuple[str | int, ...] | str | int


def _validate_item_key(value: Any) -> None:
    """Check that *value* is an acceptable return value of the
    :setting:`ITEM_PIPELINE_KEY` callable, raising :exc:`TypeError` otherwise."""
    if value is None:
        return
    if isinstance(value, (str, int)) and not isinstance(value, bool):
        return
    if (
        isinstance(value, tuple)
        and value
        and all(
            isinstance(part, (str, int)) and not isinstance(part, bool)
            for part in value
        )
    ):
        return
    raise TypeError(
        "The ITEM_PIPELINE_KEY callable must return None, a string, a non-bool"
        " integer, or a non-empty tuple of strings and non-bool integers, got"
        f" {value!r} of type {type(value).__name__}"
    )


class _ItemEntry:
    """The scheduling state of one item from its reception until its terminal
    signal has been processed."""

    __slots__ = ("cancelled", "item", "key", "ready", "response", "started")

    def __init__(self, item: Any, response: Any, key: ItemKey | None) -> None:
        self.item = item
        self.response = response
        # Computed once, before any pipeline runs; later item changes do not
        # affect the ordering key.
        self.key = key
        # Fired once the item may enter the item processor.
        self.ready: Deferred[None] = Deferred()
        # Set when the scheduler hands the entry to the item processor.
        self.started = False
        # Set when the task carrying the item is cancelled while it waits.
        self.cancelled = False

    def cancel(self) -> None:
        """Cancel an entry that has not started processing yet."""
        if self.started or self.cancelled:
            return
        self.cancelled = True
        if not self.ready.called:
            self.ready.errback(asyncio.CancelledError())


class _ItemScheduler:
    """Orders item processing by business key and bounds the number of items
    whose processing is in progress.

    Items are received in the order in which the spider output produces them.
    Items sharing a key run one after another; items with different keys, or a
    ``None`` key, run in parallel as long as the active limit allows it.
    """

    def __init__(
        self,
        max_active: int,
        key_function: Callable[..., Any] | None,
    ) -> None:
        self._max_active = max_active
        self._key_function = key_function
        # Entries still waiting to start, in reception order.
        self._waiting: deque[_ItemEntry] = deque()
        # The entry in progress for each key, if any.
        self._active_keys: dict[ItemKey, _ItemEntry] = {}
        self._active_count = 0
        self._pumping = False

    @property
    def _pending_count(self) -> int:
        """Number of received items that have not reached a terminal signal."""
        return self._active_count + len(self._waiting)

    def receive(self, item: Any, response: Any, spider: Any) -> _ItemEntry:
        """Register a newly received item and compute its key.

        Exceptions raised by the key callable, and :exc:`TypeError` for an
        invalid return value, propagate to the caller, which reports them
        through the :signal:`item_error` signal.
        """
        key: ItemKey | None = None
        if self._key_function is not None:
            key = self._key_function(item, spider)
            _validate_item_key(key)
        entry = _ItemEntry(item, response, key)
        self._waiting.append(entry)
        self._pump()
        return entry

    async def wait_to_run(self, entry: _ItemEntry) -> None:
        """Wait until the entry's key is free and an active slot is available.

        A cancellation while waiting removes the entry from the scheduling
        state, so the item never enters the item processor and produces no
        terminal signal. Once the entry has started, a cancellation is passed
        on to the item processor and the entry keeps its key and active slot.
        """
        if entry.started:
            return
        try:
            await maybe_deferred_to_future(entry.ready)
        except (asyncio.CancelledError, CancelledError):
            # The ready Deferred may have been errbacked by entry.cancel(),
            # or the task may have been cancelled while suspended on it.
            if entry.started:
                raise
            entry.cancel()
            self._discard(entry)
            raise
        assert entry.started

    def task_done(self, entry: _ItemEntry) -> None:
        """Release the key and the active slot of a finished entry and start
        whatever can start next."""
        assert entry.started
        self._active_count -= 1
        if entry.key is not None:
            assert self._active_keys.get(entry.key) is entry
            del self._active_keys[entry.key]
        self._pump()

    def cancel_waiting(self, entry: _ItemEntry) -> None:
        """Cancel an entry that has not started processing yet, if it still
        waits."""
        if entry.started or entry.cancelled:
            return
        self._discard(entry, cancel=True)

    def _discard(self, entry: _ItemEntry, *, cancel: bool = False) -> None:
        try:
            self._waiting.remove(entry)
        except ValueError:
            return
        if cancel:
            entry.cancel()
        self._pump()

    def _pump(self) -> None:
        # Reentrant pumps happen when a ready Deferred resumes a coroutine
        # that finishes its whole item within the callback chain. The outer
        # pump then re-checks the current state of every entry.
        if self._pumping:
            return
        self._pumping = True
        try:
            for entry in list(self._waiting):
                if entry.cancelled:
                    continue
                if self._active_count >= self._max_active:
                    break
                if entry.key is not None and entry.key in self._active_keys:
                    continue
                entry.started = True
                self._active_count += 1
                if entry.key is not None:
                    self._active_keys[entry.key] = entry
                self._waiting.remove(entry)
                entry.ready.callback(None)
        finally:
            self._pumping = False
