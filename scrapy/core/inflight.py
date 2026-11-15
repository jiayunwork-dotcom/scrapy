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
    """SQLite-backed storage of the requests that left, or are in the middle
    of leaving, the scheduler but did not finish processing yet.

    A request goes through two persistent states while the scheduler hands it
    over to the engine:

    1.  :meth:`stage` inserts its serialized form *before* it is removed from
        the request queue. If the process dies at any point around that write,
        the request exists in the queue, in this store, or in both, but never
        in neither.

    2.  Once the queue removal is also persisted, the staged row is the only
        remaining copy and plays the role of the in-flight record.

    Every stored request is either acknowledged through :meth:`ack` once its
    processing reaches the completion boundary, or returned again by
    :meth:`recover` when the job is resumed after an unclean shutdown. The
    scheduler reconciles rows that still have a counterpart in the request
    queue so the handoff never produces two schedulable copies.

    Each write is committed before the method returns, so that a killed process
    leaves behind a consistent record of what was in flight. Acknowledged rows
    are deleted as they are acknowledged, and the rows of recovered requests
    are deleted while they are re-enqueued and written again when those
    requests leave the scheduler, which keeps both the database and its memory
    use proportional to the number of requests that are pending, handing off
    or in flight rather than to the whole crawl history.
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
            " request BLOB NOT NULL,"
            " handoff_token TEXT)"
        )
        self._ensure_handoff_column()
        # Maps live requests to the row that holds their serialized form,
        # including separate rows for identical concurrent requests.
        self._row_ids: dict[Request, int] = {}

    def _ensure_handoff_column(self) -> None:
        # Databases written by the first in-flight tracking release predate
        # the handoff token; add the column so such job directories keep
        # opening without a migration step on the caller side.
        columns = {row[1] for row in self._db.execute("PRAGMA table_info(inflight)")}
        if "handoff_token" not in columns:
            self._db.execute("ALTER TABLE inflight ADD COLUMN handoff_token TEXT")
            self._db.commit()

    def stage(self, request: Request, data: bytes, handoff_token: str) -> int:
        """Persist *request*, already serialized as *data*, while it is still
        present in the request queue.

        *handoff_token* identifies the queue counterpart of the staged row, so
        recovery can keep exactly one copy when both sides survived a kill.
        Callers whose request instance is not the one the queue will hand back
        (the crash-safe disk handoff) re-link the returned instance with
        :meth:`claim`; other callers can use the staged row through *request*
        directly.
        """
        cursor = self._db.execute(
            "INSERT INTO inflight (request, handoff_token) VALUES (?, ?)",
            (data, handoff_token),
        )
        self._db.commit()
        row_id = cursor.lastrowid
        assert row_id is not None
        self._row_ids[request] = row_id
        return row_id

    def claim(self, request: Request, handoff_token: str) -> None:
        """Link the live *request* returned by the queue to the row previously
        staged with *handoff_token*, so :meth:`ack` can remove it."""
        row = self._db.execute(
            "SELECT id FROM inflight WHERE handoff_token = ? ORDER BY id DESC LIMIT 1",
            (handoff_token,),
        ).fetchone()
        if row is None:
            raise sqlite3.DatabaseError(
                f"No staged in-flight row found for handoff token {handoff_token!r}"
            )
        self._row_ids[request] = row[0]

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

    def recover(self) -> Iterator[tuple[int, Request, str | None]]:
        """Yield ``(row id, request, handoff token)`` triples left over by a
        previous process, oldest first. The handoff token is ``None`` for rows
        written by a Scrapy version that did not have the staging step."""
        for row_id, data, handoff_token in self._db.execute(
            "SELECT id, request, handoff_token FROM inflight ORDER BY id"
        ):
            # The job directory is treated with the same trust as the spider
            # code itself; see the JOBDIR security warning in the docs.
            yield (
                row_id,
                request_from_dict(
                    pickle.loads(data),  # noqa: S301
                    spider=self._spider,
                ),
                handoff_token,
            )

    def close(self) -> None:
        self._db.commit()
        self._db.execute("PRAGMA wal_checkpoint(TRUNCATE)").close()
        self._db.close()
