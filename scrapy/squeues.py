"""
Scheduler queues
"""

from __future__ import annotations

import marshal
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, cast

from queuelib import queue

from scrapy.utils.request import request_from_dict

if TYPE_CHECKING:
    from collections.abc import Callable
    from os import PathLike

    from scrapy import Request
    from scrapy.crawler import Crawler


def _with_mkdir(queue_class: type[queue.BaseQueue]) -> type[queue.BaseQueue]:
    class DirectoriesCreated(queue_class):  # type: ignore[valid-type,misc]
        def __init__(self, path: str | PathLike[str], *args: Any, **kwargs: Any):
            dirname = Path(path).parent
            if not dirname.exists():
                dirname.mkdir(parents=True, exist_ok=True)
            super().__init__(path, *args, **kwargs)

    return DirectoriesCreated


class _EntryAwareSQLiteQueueMixin:
    """Adds id-aware access on top of queuelib's SQLite queues.

    The id is the SQLite row id the queue assigns on push; :meth:`pop_entry`
    deletes that exact row inside a single transaction so a crash between the
    staging write and the queue removal cannot lose the request.
    """

    _sql_peek_entry = "SELECT id, item FROM queue ORDER BY id LIMIT 1"

    def peek_entry(self) -> tuple[int, bytes] | None:
        with self._db as conn:  # type: ignore[attr-defined]
            for entry_id, item in conn.execute(self._sql_peek_entry):
                return entry_id, item
        return None

    def pop_entry(self, entry_id: int) -> bytes | None:
        with self._db as conn:  # type: ignore[attr-defined]
            row = conn.execute(
                "SELECT item FROM queue WHERE id = ?", (entry_id,)
            ).fetchone()
            if row is None:
                return None
            conn.execute("DELETE FROM queue WHERE id = ?", (entry_id,))
            return cast("bytes", row[0])

    def stamp_entry(self, entry_id: int, item: bytes) -> bool:
        """Replace the payload of row *entry_id* in place with *item*.

        The row id and the queue position are preserved, so this is the
        crash-safe way to attach a handoff token to an entry written by a
        Scrapy version that did not have one. Returns ``False`` when the row
        no longer exists.
        """
        with self._db as conn:  # type: ignore[attr-defined]
            cursor = conn.execute(
                "UPDATE queue SET item = ? WHERE id = ?", (item, entry_id)
            )
            return bool(cursor.rowcount > 0)

    def iter_items(self) -> Any:
        """Yield the serialized payload of every stored entry without removing
        any of them."""
        with self._db as conn:  # type: ignore[attr-defined]
            for (item,) in conn.execute("SELECT item FROM queue"):
                yield item


class _FifoEntrySQLiteQueue(_EntryAwareSQLiteQueueMixin, queue.FifoSQLiteQueue):
    pass


class _LifoEntrySQLiteQueue(_EntryAwareSQLiteQueueMixin, queue.LifoSQLiteQueue):
    _sql_peek_entry = "SELECT id, item FROM queue ORDER BY id DESC LIMIT 1"


def _serializable_queue(
    queue_class: type[queue.BaseQueue],
    serialize: Callable[[Any], bytes],
    deserialize: Callable[[bytes], Any],
) -> type[queue.BaseQueue]:
    class SerializableQueue(queue_class):  # type: ignore[valid-type,misc]
        def push(self, obj: Any) -> None:
            s = serialize(obj)
            super().push(s)

        def pop(self) -> Any | None:
            s = super().pop()
            if s:
                return deserialize(s)
            return None

        def peek(self) -> Any | None:
            """Returns the next object to be returned by :meth:`pop`,
            but without removing it from the queue.

            Raises :exc:`NotImplementedError` if the underlying queue class does
            not implement a ``peek`` method, which is optional for queues.
            """
            try:
                s = super().peek()
            except AttributeError as ex:
                raise NotImplementedError(
                    "The underlying queue class does not implement 'peek'"
                ) from ex
            if s:
                return deserialize(s)
            return None

        def peek_entry(self) -> tuple[int, Any] | None:
            """Returns the ``(id, serialized object)`` pair of the next entry
            without removing it, or ``None`` when the queue is empty.

            Only the SQLite queues assign ids to their entries.
            """
            try:
                entry = super().peek_entry()
            except AttributeError as ex:
                raise NotImplementedError(
                    "The underlying queue class does not implement 'peek_entry'"
                ) from ex
            if entry is None:
                return None
            entry_id, s = entry
            return entry_id, deserialize(s)

        def pop_entry(self, entry_id: int) -> Any | None:
            """Removes the entry identified by *entry_id* and returns its
            deserialized object, or ``None`` when it no longer exists."""
            s = super().pop_entry(entry_id)
            if s is None:
                return None
            return deserialize(s)

        def iter_objects(self) -> Any:
            """Yield every stored object without removing it."""
            for s in super().iter_items():
                yield deserialize(s)

        def stamp_entry(self, entry_id: int, obj: Any) -> bool:
            """Replace the object stored under *entry_id* in place."""
            return bool(super().stamp_entry(entry_id, serialize(obj)))

    return SerializableQueue


def _scrapy_serialization_queue(
    queue_class: type[queue.BaseQueue],
) -> type[queue.BaseQueue]:
    class ScrapyRequestQueue(queue_class):  # type: ignore[valid-type,misc]
        def __init__(self, crawler: Crawler, key: str):
            self.spider = crawler.spider
            super().__init__(key)

        @classmethod
        def from_crawler(
            cls, crawler: Crawler, key: str, *args: Any, **kwargs: Any
        ) -> Self:
            return cls(crawler, key)

        def push(self, request: Request) -> None:
            request_dict = request.to_dict(spider=self.spider)
            super().push(request_dict)

        def pop(self) -> Request | None:
            request = super().pop()
            if not request:
                return None
            return request_from_dict(request, spider=self.spider)

        def peek(self) -> Request | None:
            """Returns the next object to be returned by :meth:`pop`,
            but without removing it from the queue.

            Raises :exc:`NotImplementedError` if the underlying queue class does
            not implement a ``peek`` method, which is optional for queues.
            """
            request = super().peek()
            if not request:
                return None
            return request_from_dict(request, spider=self.spider)

    return ScrapyRequestQueue


def _scrapy_entry_queue(queue_class: type[queue.BaseQueue]) -> type[queue.BaseQueue]:
    """Like :func:`_scrapy_serialization_queue`, but with entry-aware access.

    The entry-aware methods expose the integer id the SQLite queues assign to
    each row, which the crash-safe scheduler handoff uses to identify a queue
    entry across the in-flight database and the request queue.
    """

    class ScrapyRequestEntryQueue(
        _scrapy_serialization_queue(queue_class)  # type: ignore[misc]
    ):
        def peek_entry(self) -> tuple[int, Request] | None:
            """Return the id and the request of the next entry without
            removing it, or ``None`` when the queue is empty."""
            entry = super().peek_entry()
            if entry is None:
                return None
            entry_id, request_dict = entry
            return entry_id, request_from_dict(request_dict, spider=self.spider)

        def pop_entry(self, entry_id: int) -> Request | None:
            """Remove and return the request stored under *entry_id*.

            Returning ``None`` means another process or call already consumed
            the entry, so the caller can treat the staged in-flight record as
            the surviving copy.
            """
            request_dict = super().pop_entry(entry_id)
            if request_dict is None:
                return None
            return request_from_dict(request_dict, spider=self.spider)

        def iter_requests(self) -> Any:
            """Yield every stored request without removing it."""
            for request_dict in super().iter_objects():
                yield request_from_dict(request_dict, spider=self.spider)

        def stamp_entry(self, entry_id: int, request: Request) -> bool:
            """Replace the request stored under *entry_id* in place."""
            return bool(
                super().stamp_entry(entry_id, request.to_dict(spider=self.spider))
            )

    return ScrapyRequestEntryQueue


def _scrapy_non_serialization_queue(
    queue_class: type[queue.BaseQueue],
) -> type[queue.BaseQueue]:
    class ScrapyRequestQueue(queue_class):  # type: ignore[valid-type,misc]
        @classmethod
        def from_crawler(cls, crawler: Crawler, *args: Any, **kwargs: Any) -> Self:
            return cls()

        def peek(self) -> Any | None:
            """Returns the next object to be returned by :meth:`pop`,
            but without removing it from the queue.

            Raises :exc:`NotImplementedError` if the underlying queue class does
            not implement a ``peek`` method, which is optional for queues.
            """
            try:
                s = super().peek()
            except AttributeError as ex:
                raise NotImplementedError(
                    "The underlying queue class does not implement 'peek'"
                ) from ex
            return s

    return ScrapyRequestQueue


def _pickle_serialize(obj: Any) -> bytes:
    try:
        return pickle.dumps(obj, protocol=4)
    # Both pickle.PicklingError and AttributeError can be raised by pickle.dump(s)
    # TypeError is raised from parsel.Selector
    except (pickle.PicklingError, AttributeError, TypeError) as e:
        raise ValueError(str(e)) from e


# queue.*Queue aren't subclasses of queue.BaseQueue
_PickleFifoSerializationDiskQueue = _serializable_queue(
    _with_mkdir(queue.FifoDiskQueue),  # type: ignore[arg-type]
    _pickle_serialize,
    pickle.loads,
)
_PickleLifoSerializationDiskQueue = _serializable_queue(
    _with_mkdir(queue.LifoDiskQueue),  # type: ignore[arg-type]
    _pickle_serialize,
    pickle.loads,
)
_MarshalFifoSerializationDiskQueue = _serializable_queue(
    _with_mkdir(queue.FifoDiskQueue),  # type: ignore[arg-type]
    marshal.dumps,
    marshal.loads,
)
_MarshalLifoSerializationDiskQueue = _serializable_queue(
    _with_mkdir(queue.LifoDiskQueue),  # type: ignore[arg-type]
    marshal.dumps,
    marshal.loads,
)
_PickleFifoSerializationSQLiteQueue = _serializable_queue(
    _with_mkdir(_FifoEntrySQLiteQueue),  # type: ignore[arg-type]
    _pickle_serialize,
    pickle.loads,
)
_PickleLifoSerializationSQLiteQueue = _serializable_queue(
    _with_mkdir(_LifoEntrySQLiteQueue),  # type: ignore[arg-type]
    _pickle_serialize,
    pickle.loads,
)
_MarshalFifoSerializationSQLiteQueue = _serializable_queue(
    _with_mkdir(queue.FifoSQLiteQueue),  # type: ignore[arg-type]
    marshal.dumps,
    marshal.loads,
)
_MarshalLifoSerializationSQLiteQueue = _serializable_queue(
    _with_mkdir(queue.LifoSQLiteQueue),  # type: ignore[arg-type]
    marshal.dumps,
    marshal.loads,
)

# public queue classes
PickleFifoDiskQueue = _scrapy_serialization_queue(_PickleFifoSerializationDiskQueue)
PickleLifoDiskQueue = _scrapy_serialization_queue(_PickleLifoSerializationDiskQueue)
MarshalFifoDiskQueue = _scrapy_serialization_queue(_MarshalFifoSerializationDiskQueue)
MarshalLifoDiskQueue = _scrapy_serialization_queue(_MarshalLifoSerializationDiskQueue)
PickleFifoSQLiteQueue = _scrapy_entry_queue(_PickleFifoSerializationSQLiteQueue)
PickleLifoSQLiteQueue = _scrapy_entry_queue(_PickleLifoSerializationSQLiteQueue)
MarshalFifoSQLiteQueue = _scrapy_serialization_queue(
    _MarshalFifoSerializationSQLiteQueue
)
MarshalLifoSQLiteQueue = _scrapy_serialization_queue(
    _MarshalLifoSerializationSQLiteQueue
)
FifoMemoryQueue = _scrapy_non_serialization_queue(queue.FifoMemoryQueue)  # type: ignore[arg-type]
LifoMemoryQueue = _scrapy_non_serialization_queue(queue.LifoMemoryQueue)  # type: ignore[arg-type]
