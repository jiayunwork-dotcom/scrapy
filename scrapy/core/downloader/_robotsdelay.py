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
# RobotsTxtMiddleware. A delay of 0 means the request does not wait itself,
# but still records its start, so that later requests with a positive
# interval count it as their previous request.
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
    # Whether the request actually waited for an interval. Set once it has ever
    # been unable to start immediately, and never reset, so that the
    # robotstxt/delayed_count stat counts each request at most once.
    waited: bool = False
    cancelled: bool = False


@dataclass(eq=False)
class _SiteState:
    # Real start time of the previous non-exempt request of the site, or None
    # if none has started yet.
    last_start: float | None = None
    # The positive-interval waiter whose start has been reserved, i.e. whose
    # interval since last_start has elapsed, but whose request has not reached
    # the download handler yet. At most one such waiter exists at a time, so
    # positive-interval requests cannot start in bursts. Zero-interval
    # waiters are not held by it: they must be able to overtake a request that
    # is still queued inside a download slot.
    starting: _Waiter | None = None
    timer: CallLaterResult | None = None
    # (-priority, seq, waiter) entries. A higher priority entry, then an older
    # one, is granted first among the requests eligible at the same time.
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
    slots. For every non-exempt request R whose robots.txt interval is d, the
    real start of R (the moment it is handed to the download handler) happens
    at least d seconds after the real start of the previous non-exempt request
    of the site, whatever interval that previous request had, including 0.

    Enforcement has two stages. :meth:`acquire` releases a request as soon as
    its own interval since the last recorded start has elapsed. A
    positive-interval request is released only once no earlier
    positive-interval request of the site is still waiting to start, so such
    requests cannot start in bursts; zero-interval requests ignore that gate
    and may overtake requests still queued in their download slot. Waiting
    never holds a download concurrency spot. A released request may then have
    to queue inside a download slot, and a later request of the site can reach
    its download handler first. :meth:`check_start` therefore rechecks the
    interval at the moment the request is actually about to start and, if a
    newer real start invalidated its deadline, asks the caller to wait outside
    of any concurrency spot, after which the check runs again.

    :meth:`started` records the real start once it happens.
    """

    def __init__(self, stats: StatsCollector) -> None:
        self._stats = stats
        self._sites: dict[str, _SiteState] = {}
        self._closed = False

    async def acquire(
        self, site: str, delay: float, priority: int, request: Request
    ) -> _Waiter:
        """Wait until a request of *site* with a minimum interval of *delay*
        seconds may leave the interval queue, and return the waiter that the
        caller must pass to :meth:`check_start` and then :meth:`started`."""
        if self._closed:
            raise DownloadCancelledError(
                "The downloader is shutting down and not accepting new requests"
            )
        state = self._sites.setdefault(site, _SiteState())
        waiter = _Waiter(site, delay, state.next_seq(), request)
        heapq.heappush(state.waiters, (-priority, waiter.seq, waiter))
        self._schedule(site, state)
        try:
            await maybe_deferred_to_future(waiter.dfd)
        except BaseException:
            self.abandon(waiter)
            raise
        return waiter

    def check_start(self, waiter: _Waiter) -> float:
        """Return how long *waiter*, already released by :meth:`acquire`, must
        still wait before its request may reach the download handler.

        A non-positive value means it may start now. A positive value means a
        newer request of the site started after :meth:`acquire` released this
        one, e.g. through another download slot, and the interval of this
        request has to elapse since that newer real start first.
        """
        if waiter.started or waiter.cancelled:
            return 0.0
        state = self._sites[waiter.site]
        wait = self._deadline(state, waiter) - monotonic()
        if wait > 0:
            waiter.waited = True
        return wait

    def started(self, waiter: _Waiter) -> None:
        """Record that a released request reached the download handler."""
        if waiter.started or waiter.cancelled:
            return
        waiter.started = True
        state = self._sites[waiter.site]
        if state.starting is waiter:
            state.starting = None
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
        if state.starting is waiter:
            state.starting = None
        state.cancel_timer()
        self._schedule(waiter.site, state)

    def abort(self) -> int:
        """Fail every request still waiting for its interval and stop accepting
        new ones. Return how many requests were waiting.

        A released request is left to proceed normally: its interval has
        already elapsed, and any further wait it may need happens inside its
        download slot, which the downloader drains on close separately.
        """
        self._closed = True
        failed = 0
        for state in self._sites.values():
            state.cancel_timer()
            state.starting = None
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

    def _deadline(self, state: _SiteState, waiter: _Waiter) -> float:
        """Earliest monotonic time at which *waiter* may start: its interval
        after the last recorded start, or immediately if none happened yet."""
        if state.last_start is None:
            return monotonic()
        return state.last_start + waiter.delay

    def _schedule(self, site: str, state: _SiteState) -> None:
        if self._closed:
            return
        # The last start, the heap contents and the deadlines may all have
        # changed, so recompute everything instead of trusting the timer the
        # state was armed with.
        state.cancel_timer()
        now = monotonic()
        pending: list[tuple[int, int, _Waiter]] = []
        earliest: float | None = None
        while state.waiters:
            _, _, waiter = heapq.heappop(state.waiters)
            if waiter.cancelled or waiter.granted or waiter.started:
                continue
            deadline = self._deadline(state, waiter)
            if deadline > now:
                waiter.waited = True
                pending.append((-waiter.request.priority, waiter.seq, waiter))
                if earliest is None or deadline < earliest:
                    earliest = deadline
                continue
            if waiter.delay > 0 and state.starting is not None:
                # Another positive-interval request of the site has been
                # released but not started yet. Wait for it, so that requests
                # with an interval cannot start in bursts; zero-interval
                # requests skip the gate on purpose.
                waiter.waited = True
                pending.append((-waiter.request.priority, waiter.seq, waiter))
                continue
            if waiter.delay > 0:
                state.starting = waiter
            waiter.granted = True
            if not waiter.dfd.called:
                waiter.dfd.callback(waiter)
        for entry in pending:
            heapq.heappush(state.waiters, entry)
        if earliest is not None and earliest > now:
            state.timer = call_later(
                earliest - now, self._wake, site, state
            )
        # Gate-blocked waiters, or waiters whose deadline has elapsed but that
        # the gate holds back, need no timer: the reserved request starting or
        # being abandoned calls _schedule() again.

    def _wake(self, site: str, state: _SiteState) -> None:
        state.timer = None
        self._schedule(site, state)
