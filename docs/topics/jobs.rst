.. _topics-jobs:

=================================
Jobs: pausing and resuming crawls
=================================

Sometimes, for big sites, it's desirable to pause crawls and be able to resume
them later.

Scrapy supports this functionality out of the box through the following
facilities:

* a scheduler that persists scheduled requests on disk

* a duplicates filter that persists visited requests on disk

* an extension that keeps some spider state (key/value pairs) persistent
  between batches

.. _job-dir:

Job directory
=============

To enable persistence support, define a *job directory* through the
:setting:`JOBDIR` setting.

The job directory will store all required data to keep the state of a *single*
job (i.e. a spider run), so that if stopped cleanly, it can be resumed later.

.. warning:: This directory must *not* be shared by different spiders, or even
    different jobs of the same spider. See :ref:`job-dir-spider-name`.

.. warning:: Treat the job directory with the same security care as your
    Scrapy project source code. Do not point ``JOBDIR`` to a path that
    untrusted parties can write to.

See also :ref:`job-dir-contents`.

How to use it
=============

To start a spider with persistence support enabled, run it like this::

    scrapy crawl somespider -s JOBDIR=crawls/somespider-1

Then you can stop the spider safely at any time (by pressing Ctrl-C or sending
a signal), and resume it later by issuing the same command::

    scrapy crawl somespider -s JOBDIR=crawls/somespider-1

.. _job-dir-spider-name:

Deriving the job directory from the spider name
===============================================

One way to keep job directories apart is to build the path in
:meth:`~scrapy.Spider.update_settings`, so that a group of spiders shares the
same path except for the spider name:

.. code-block:: python

    from pathlib import Path

    from scrapy import Spider


    class BaseSpider(Spider):
        @classmethod
        def update_settings(cls, settings):
            super().update_settings(settings)
            settings.set("JOBDIR", str(Path("crawls", cls.name)), priority="spider")


.. _topics-keeping-persistent-state-between-batches:

Keeping persistent state between batches
========================================

Sometimes you'll want to keep persistent spider state between pause/resume
batches. Use the ``spider.state`` attribute for that. It should be a dict.
There's :ref:`a built-in extension <topics-extensions-ref-spiderstate>` that
takes care of serializing, storing and loading that attribute from the job
directory when the spider starts and stops.

Here's an example of a callback that uses the spider state (other spider code
is omitted for brevity):

.. code-block:: python

    def parse_item(self, response):
        # parse item here
        self.state["items_count"] = self.state.get("items_count", 0) + 1

Persistence gotchas
===================

There are a few things to keep in mind if you want to use Scrapy's persistence
support:

Pause limitations
-----------------

Job pausing and resuming is only supported when the spider is paused by
stopping it cleanly. Forced, sudden or otherwise unclean shutdown can lead to
data corruption in the job directory, which may prevent the spider from
resuming correctly. In-flight request recovery
(:ref:`inflight-request-recovery`) covers hard process termination on a healthy
machine, such as ``SIGKILL``, but neither it nor the rest of job persistence is
designed to survive disk hardware failure or loss of power.

Scrapy version changes
----------------------

The contents of a job directory are an implementation detail of the Scrapy
version that wrote them. A job must be resumed with the same Scrapy version
that paused it; after upgrading or downgrading Scrapy, start a new job with a
new job directory.

Feed exports
------------

When a job is resumed, the :ref:`feeds <topics-feed-exports>` of the previous
run already exist, so their :ref:`mode <feed-mode>` determines what happens to
the items of the resumed run: ``"append"`` adds them to those feeds,
``"overwrite"`` replaces the items of the previous run with them, and
``"create"`` does not write them anywhere.

Cookie expiration
-----------------

Cookies may expire, so if you don't resume your spider quickly, the scheduled
requests may no longer work. This won't be an issue if your spider doesn't rely
on cookies.

.. _request-serialization:

Request serialization
---------------------

For persistence to work, :class:`~scrapy.Request` objects must be
serializable with :mod:`pickle`, except for the :ref:`callback
<callbacks>` and :ref:`errback
<errbacks>` values passed to their ``__init__``
method, which must be methods of the running :class:`~scrapy.Spider` class.

Requests that cannot be serialized are kept in memory only: they are still
sent, but they are lost when the crawl is paused.

If you wish to log the requests that couldn't be serialized, you can set the
:setting:`SCHEDULER_DEBUG` setting to ``True`` in the project's settings page.
It is ``False`` by default.

.. note:: Because requests are serialized with :mod:`pickle`, the objects you
    store on a request, such as the values of its
    :attr:`~scrapy.Request.cb_kwargs` and :attr:`~scrapy.Request.meta`
    dictionaries, are deep-copied when the request is written to and later read
    back from the job directory. As a result, the callback receives a *copy* of
    those objects rather than the original ones, and changes made to the copy are
    not reflected in the original object. Keep this in mind if you rely on
    sharing mutable state through ``cb_kwargs`` or ``meta``.

.. _inflight-request-recovery:

In-flight request recovery
==========================

With :setting:`JOBDIR` alone, only the requests that are still stored in the
scheduler queues survive an unclean shutdown: once the scheduler hands a
request to the engine, nothing on disk remembers it anymore. If the process is
killed while that request is being downloaded, while its callback is running,
or while its items are still going through the item pipelines, the request is
lost, and resubmitting its URL is blocked by the dupefilter.

Setting :setting:`JOBDIR_TRACK_INFLIGHT` to ``True`` additionally persists
every request from the moment it leaves the scheduler until its processing
reaches its completion boundary. A process killed at any point in between
re-schedules the request the next time the same job directory is opened.

Crash-safe queue handoff
------------------------

Removing a request from the persistent request queue and writing its in-flight
record are two separate writes, so the request could otherwise disappear in
the narrow window between them: the queue entry is gone but the in-flight
record does not exist yet.

To close that window, the scheduler stages every disk-backed request in the
in-flight database *before* removing it from the request queue, and links the
two persistent copies with an internal handoff token. As a result, a request
that was successfully enqueued always has at least one durable representation
while it is being selected, dequeued and handed to the engine:

-   a kill before the staging write leaves the request in the request queue;

-   a kill after the staging write leaves its in-flight record;

-   if both copies survive, reopening the job directory keeps exactly one
    schedulable instance and discards the duplicated in-flight record, so
    repeated kills during handoff and recovery can never multiply the same
    request instance.

The token is scheduler-internal state. It is removed before the request
reaches the engine, never changes :attr:`~scrapy.Request.dont_filter`, is not
used for request fingerprints and is not part of the dupefilter history.

Supported configuration
-----------------------

The first version of this feature is meant for a single spider process, on a
single machine, using one job directory with the same Scrapy version and the
same spider code and settings on every run. It only works with:

-   the default :class:`~scrapy.core.scheduler.Scheduler`,

-   :class:`~scrapy.pqueues.ScrapyPriorityQueue` as
    :setting:`SCHEDULER_PRIORITY_QUEUE`,

-   :class:`~scrapy.squeues.PickleLifoSQLiteQueue` as
    :setting:`SCHEDULER_DISK_QUEUE` and
    :class:`~scrapy.squeues.PickleFifoSQLiteQueue` as
    :setting:`SCHEDULER_START_DISK_QUEUE`,

-   :class:`~scrapy.dupefilters.DiskDupeFilter` as
    :setting:`DUPEFILTER_CLASS`,

-   :setting:`JOBDIR_SYNC_EVERY` set to ``1``.

If any of these conditions is not met, or :setting:`JOBDIR` is not set, the
crawl raises an error before it starts instead of silently degrading. A job
directory that still contains in-flight records from a previous run cannot be
opened with :setting:`JOBDIR_TRACK_INFLIGHT` disabled.

An old job directory that has no in-flight records yet can be opened with
tracking enabled, and its existing pending request queue is reused as is.

Completion boundary
-------------------

A request stops being recoverable only after its processing is fully done:
the output of its callback or errback has been consumed, every new request it
produced went through the normal scheduling acceptance or rejection flow, and
every item it produced finished going through the item pipelines. Merely
receiving the response, or getting a generator back from the callback before
it is consumed, does not complete the request.

Callback exceptions, final download failures,
:exc:`~scrapy.exceptions.DropItem` and pipeline exceptions keep Scrapy's
existing logging, signals and crawl termination semantics; these normal
endings do not trigger extra retries, and once they are handled the request
is no longer recovered.

Requests are the recovery unit, never request fingerprints: two identical
requests with :attr:`~scrapy.Request.dont_filter` set that are in flight at the
same time are recovered independently, and acknowledging one does not
acknowledge the other. Recovery preserves the method, URL, body, headers,
cookies, ``meta``, ``cb_kwargs``, callback, errback, priority and
``dont_filter`` of each request. The dupefilter is bypassed for the one
re-scheduling of a recovered request without changing its ``dont_filter``
value, so a later, newly submitted request with the same URL is still filtered
as a duplicate.

Replacement requests
--------------------

Requests produced by downloader middlewares, such as redirects and
:class:`~scrapy.downloadermiddlewares.retry.RetryMiddleware` retries, are
covered as well. If the process is killed at the boundary where the original
request has ended and its replacement request is being enqueued, at least one
side of that chain survives and continues the crawl on restart. Ordinary child
requests yielded from a callback remain independently scheduled and
deduplicated.

Inline downloads made by directly awaiting a
:class:`~scrapy.Request` or through
:meth:`~scrapy.core.engine.ExecutionEngine.download_async` are recovered
through the outer scheduled request they belong to. Standalone inline
downloads with no outer scheduled request, e.g. those made from
:meth:`~scrapy.Spider.start`, are not covered.

Shutdown behavior
-----------------

During a graceful shutdown, requests that finish while the spider is closing
are acknowledged normally. During a fast shutdown, the in-flight downloads that
are dropped keep their recovery records and are re-scheduled on the next
start, while the fast shutdown itself still drops the downloads immediately
instead of waiting for the network. Requests that remain unfinished for any
other closing or cancellation reason keep their recovery responsibility as
well.

At-least-once delivery
----------------------

Recovery follows at-least-once semantics. If the process exits after the
output of a request happened but before the completion confirmation reached
disk, the request is processed again on restart, which may produce the same
items more than once. Scrapy does not provide exactly-once guarantees for
external writes: item pipelines and their downstream systems must handle
duplicate items themselves.

Only requests that follow Scrapy's existing request serialization rules are
accepted. A request that cannot be serialized is rejected when it is scheduled,
with the reason logged, rather than being reported as enqueued while only held
in memory. A failure to read, write, remove or confirm the records involved in
the queue handoff is not hidden behind a log line while the request is handed
to the engine anyway: the crawl ends through the existing error-based closing
mechanism with diagnostics, and requests that were durably accepted before the
failure remain recoverable when the job directory is opened again. When an
operation has already taken effect and the error only surfaces afterwards,
Scrapy follows the actual on-disk state rather than assuming the request still
belongs to the caller that saw the exception.

The in-flight records are removed as requests complete, so both their disk
footprint and their in-memory state scale with the number of pending,
handing-off and in-flight requests rather than with the whole completed crawl
history.

.. warning:: This feature protects against hard process termination on a
    healthy local machine, such as ``SIGKILL``. It does not cover disk
    hardware failure or loss of power, which can leave partially written files
    regardless of application-level synchronization.

.. _job-dir-contents:

Job directory contents
======================

The contents of a job directory depend on the components used during the job.
Components known to write in the job directory include the :ref:`scheduler
<topics-scheduler>` and the :class:`~scrapy.extensions.spiderstate.SpiderState`
extension. See the reference documentation of the corresponding components for
details.

For example, with default settings, the job directory may look like this:

.. code-block:: none

    ├── requests.queue
    |   ├── active.json
    |   └── {hostname}-{hash}
    |       └── {priority}{s?}
    |           ├── q{00000}
    |           └── info.json
    ├── requests.inflight.db
    ├── requests.seen
    └── spider.state

Where:

-   :class:`~scrapy.core.scheduler.Scheduler` creates the ``requests.queue/``
    directory and the ``active.json`` file, the latter containing the state
    data of :class:`~scrapy.pqueues.DownloaderAwarePriorityQueue`, written
    when the job stops and, with :setting:`JOBDIR_SYNC_EVERY`, as it runs.

-   :class:`~scrapy.pqueues.DownloaderAwarePriorityQueue` creates the
    ``{hostname}-{hash}`` directories.

-   :class:`~scrapy.pqueues.ScrapyPriorityQueue` creates the ``{priority}{s?}``
    directories.

-   :class:`scrapy.squeues.PickleFifoDiskQueue`, a subclass of
    :class:`queuelib.FifoDiskQueue` that uses :mod:`pickle` to serialize
    :class:`dict` representations of :class:`scrapy.Request` objects, creates
    the ``info.json`` and ``q{00000}`` files.

-   :class:`~scrapy.dupefilters.RFPDupeFilter` creates the ``requests.seen``
    file.

-   The in-flight request recovery described in
    :ref:`inflight-request-recovery` creates the ``requests.inflight.db``
    SQLite database when :setting:`JOBDIR_TRACK_INFLIGHT` is enabled.

-   :class:`~scrapy.extensions.spiderstate.SpiderState` creates the
    ``spider.state`` file.
