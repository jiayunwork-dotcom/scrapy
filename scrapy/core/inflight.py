from __future__ import annotations

import pickle
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

from scrapy.utils.request import request_from_dict

if TYPE_CHECKING:
    from collections.abc import Iterator

    from scrapy.http.request import Request
    from scrapy.spiders import Spider


class _InflightRequestStore:
    """SQLite-backed storage of the requests that left the scheduler but did
    not finish processing yet.

    Every stored request is either acknowledged through :meth:`ack` once its
    processing reaches its completion boundary, or returned again by
    :meth:`recover` when the job is resumed after an unclean shutdown.

    Each write is committed before the method returns, so that a killed process
    leaves behind a consistent record of what was in flight. Acknowledged rows
    are deleted as they are acknowledged, and the rows of recovered requests
    are deleted while they are re-enqueued and written again when those
    requests leave the scheduler, which keeps both the database and its memory
    use proportional to the number of requests that are pending or in flight
    rather than to the whole crawl history.
    """

    def __init__(self, path: str | Path, spider: Spider) -> None:
        self._spider = spider
        self._path = Path(path, "requests.inflight.db")
        self._db = sqlite3.connect(self._path)
        self._db.execute("PRAGMA journal_mode=WAL").close()
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS inflight ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " request BLOB NOT NULL)"
        )
        # Maps live requests to the row that holds their serialized form,
        # including separate rows for identical concurrent requests.
        self._row_ids: dict[Request, int] = {}

    def add(self, request: Request, data: bytes) -> int:
        """Persist *request*, already serialized as *data*, before it leaves
        the scheduler, and remember the row so that :meth:`ack` can remove it.
        """
        cursor = self._db.execute("INSERT INTO inflight (request) VALUES (?)", (data,))
        self._db.commit()
        row_id = cursor.lastrowid
        assert row_id is not None
        self._row_ids[request] = row_id
        return row_id

    def ack(self, request: Request) -> None:
        """Remove the record of *request* after its processing completed."""
        row_id = self._row_ids.pop(request)
        self._db.execute("DELETE FROM inflight WHERE id = ?", (row_id,))
        self._db.commit()

    def discard_recovered(self, row_id: int) -> None:
        """Delete a row consumed by :meth:`recover` once its request has been
        re-enqueued into the regular queues."""
        self._db.execute("DELETE FROM inflight WHERE id = ?", (row_id,))
        self._db.commit()

    def recover(self) -> Iterator[tuple[int, Request]]:
        """Yield ``(row id, request)`` pairs left over by a previous process,
        oldest first."""
        for row_id, data in self._db.execute(
            "SELECT id, request FROM inflight ORDER BY id"
        ):
            # The job directory is treated with the same trust as the spider
            # code itself; see the JOBDIR security warning in the docs.
            yield (
                row_id,
                request_from_dict(
                    pickle.loads(data),  # noqa: S301
                    spider=self._spider,
                ),
            )

    def close(self) -> None:
        self._db.commit()
        self._db.execute("PRAGMA wal_checkpoint(TRUNCATE)").close()
        self._db.close()
