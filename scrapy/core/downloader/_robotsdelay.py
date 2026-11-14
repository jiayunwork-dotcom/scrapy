from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from time import monotonic
from typing import TYPE_CHECKING

from twisted.internet.defer import Deferred

from scrapy.exceptions import DownloadCancelledError
from scrapy.utils.asyncio import CallLaterResult, call_later
from scrapy.utils.defer import maybe_deferred_to_future

if TYPE_CHECKING:
    from scrapy.http import Request
    from scrapy.statscollectors import StatsCollector

# Request meta key holding ``(site, delay)``: the minimum interval, in seconds,
# that robots.txt asks for between the starts of the requests governed by it,
# and the key that identifies that robots.txt file. Set by
# RobotsTxtMiddleware.
_ROBOTS_DELAY_META_KEY = "_robots_delay"

# Request meta key that makes the downloader record the start of a request for
# a robots.txt interval without waiting for it, e.g. the robots.txt download
# itself.
_ROBOTS_START_META_KEY = "_robots_start"

# Request meta key that marks a request dropped while it waited for its
# robots.txt interval because the spider closed. The engine uses it to treat
# the drop as expected.
_ROBOTS_ABORTED_META_KEY = "_robots_aborted"


@dataclass(eq=False)
class _Waiter:
    site: str
    delay: float
    seq: int
    request: Request
    dfd: Deferred[_Waiter] = field(default_factory=Deferred)
    granted: bool = False
    started: bool = False
    waited: bool = False
    cancelled: bool = False


@dataclass(eq=False)
class _SiteState:
    last_start: float | None = None
    # The waiter that may proceed but whose request has not reached the
    # download handler yet. Only one request per site may be in that state, so
    # that starts do not happen in bursts.
    granted: _Waiter | None = None
    timer: CallLaterResult | None = None
    # (-priority, seq, waiter) entries, so that a higher priority request is
    # granted before a lower priority one that is waiting at the same time.
    waiters: list[tuple[int, int, _Waiter]] = field(default_factory=list)
    seq: int = 0

    def next_seq(self) -> int:
        self.seq += 1
        return self.seq

    def cancel_timer(self) -> None:
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None


class _RobotsThrottle:
    """Enforce the minimum interval that robots.txt asks for between the starts
    of the requests that the same robots.txt file governs.

    Requests are grouped by *site*, the key under which
    RobotsTxtMiddleware caches the robots.txt file, independently of download
    slots. A request may be granted only while no other request of its site has
    been granted without starting yet, and only once the interval has passed
    since the start of the previous request of the site.

    :meth:`acquire` grants a request permission to proceed. :meth:`started`
    must then be called when the request reaches the download handler, which is
    when its start actually happens.
    """

    def __init__(self, stats: StatsCollector) -> None:
        self._stats = stats
        self._sites: dict[str, _SiteState] = {}
        self._closed = False

    async def acquire(
        self, site: str, delay: float, priority: int, request: Request
    ) -> _Waiter:
        """Wait until a request of *site* with a minimum interval of *delay*
        seconds may start, and return the waiter that the caller must pass to
        :meth:`started` when the request starts."""
        if self._closed:
            raise DownloadCancelledError(
                "The downloader is shutting down and not accepting new requests"
            )
        state = self._sites.setdefault(site, _SiteState())
        now = monotonic()
        waiter = _Waiter(site, delay, state.next_seq(), request)
        # The request must wait if a request of its site has not started yet,
        # or if the interval since the last start has not elapsed.
        waiter.waited = state.granted is not None or (
            state.last_start is not None and state.last_start + delay > now
        )
        heapq.heappush(state.waiters, (-priority, waiter.seq, waiter))
        self._schedule(site, state)
        try:
            await maybe_deferred_to_future(waiter.dfd)
        except BaseException:
            self.abandon(waiter)
            raise
        return waiter

    def started(self, waiter: _Waiter) -> None:
        """Record that a granted request reached the download handler."""
        if waiter.started or waiter.cancelled:
            return
        waiter.started = True
        state = self._sites[waiter.site]
        if state.granted is waiter:
            state.granted = None
        state.last_start = monotonic()
        state.cancel_timer()
        self._schedule(waiter.site, state)

    def record_start(self, site: str) -> None:
        """Record the start of a request that is not throttled but still counts
        toward the interval, such as the robots.txt download itself."""
        if self._closed:
            return
        state = self._sites.setdefault(site, _SiteState())
        state.last_start = monotonic()
        state.cancel_timer()
        self._schedule(site, state)

    def abandon(self, waiter: _Waiter) -> None:
        """Release *waiter* without a start, e.g. because it was cancelled."""
        if waiter.started or waiter.cancelled:
            return
        waiter.cancelled = True
        state = self._sites[waiter.site]
        if state.granted is waiter:
            state.granted = None
        else:
            state.cancel_timer()
        self._schedule(waiter.site, state)

    def abort(self) -> int:
        """Fail every request still waiting for its interval and stop accepting
        new ones. Return how many requests were waiting.

        A granted request is left to proceed normally: its interval has already
        elapsed.
        """
        self._closed = True
        failed = 0
        for state in self._sites.values():
            state.cancel_timer()
            state.granted = None
            for _, _, waiter in state.waiters:
                if waiter.cancelled or waiter.granted or waiter.started:
                    continue
                waiter.cancelled = True
                failed += 1
                waiter.request.meta[_ROBOTS_ABORTED_META_KEY] = True
                if waiter.dfd.called:
                    continue
                waiter.dfd.errback(
                    DownloadCancelledError(
                        "Request dropped because the spider is closing"
                    )
                )
            state.waiters.clear()
        return failed

    def _schedule(self, site: str, state: _SiteState) -> None:
        if self._closed or state.granted is not None:
            return
        # The top of the heap may have changed, or its deadline may be earlier
        # than the one the current timer was set for, so recompute it.
        state.cancel_timer()
        while state.waiters:
            _, _, waiter = state.waiters[0]
            if waiter.cancelled or waiter.granted or waiter.started:
                heapq.heappop(state.waiters)
                continue
            now = monotonic()
            deadline = (
                now if state.last_start is None else state.last_start + waiter.delay
            )
            if deadline > now:
                state.timer = call_later(deadline - now, self._wake, site, state)
                return
            heapq.heappop(state.waiters)
            waiter.granted = True
            state.granted = waiter
            if not waiter.dfd.called:
                waiter.dfd.callback(waiter)
            return

    def _wake(self, site: str, state: _SiteState) -> None:
        state.timer = None
        self._schedule(site, state)
