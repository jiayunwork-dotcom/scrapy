"""Small I/O helpers used by feed export accounting."""

from __future__ import annotations

from contextlib import suppress
from io import IOBase
from typing import IO


class CountingWriter(IOBase):
    """A write-only proxy that counts the bytes actually written.

    ``CountingWriter`` only relies on the ``write`` and ``flush`` methods of
    the wrapped object, so it also works with targets that do not implement
    ``tell``, ``seek`` or ``fileno`` (for example, network or stream targets).

    Closing the proxy does *not* close the wrapped object: ownership of the
    target stays with whoever created it.
    """

    def __init__(self, file: IO[bytes]):
        self.file = file
        self.byte_count = 0

    def writable(self) -> bool:
        return True

    def readable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return False

    def tell(self) -> int:
        # Plugins may call ``tell`` on the object they receive; report the
        # number of bytes written so far instead of relying on the target.
        return self.byte_count

    def write(self, data: bytes) -> int:  # type: ignore[override]
        written = self.file.write(data)
        if written is None:
            written = len(data)
        self.byte_count += written
        return written

    def flush(self) -> None:
        flush = getattr(self.file, "flush", None)
        if flush is None or getattr(self.file, "closed", False):
            return
        # The target may have been closed by someone else.
        with suppress(ValueError):
            flush()

    def close(self) -> None:
        if not self.closed:
            super().close()
