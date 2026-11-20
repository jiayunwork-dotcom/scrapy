"""Pytest entry point for the file-backed sitemap cancellation experiment.

Each sitemap size (8 MiB and 128 MiB of decoded XML) runs in its own
subprocess, so the memory and disk peaks of one size never pollute the
other. The experiment script (tests/experiments/sitemap_file_backed_cancel.py)
does the measurement and emits a JSON line; this module only checks the
contract:

* cancelling the download coroutine while the decode is parked reports the
  cancellation to the caller immediately, independent of the document size;
* once the worker thread is released, the cleanup finishes promptly (when
  the decode already finished) or as soon as the remaining decode is done
  (when it had not started), never leaving files, pending decode slots or
  success stats behind;
* neither the compressed nor the decoded body is pulled into Python memory
  to transfer ownership: the traced Python peak stays in the low
  single-digit MiB and the RSS peak stays flat across sizes;
* the temporary-disk peak is about the decoded size (the decode's own input
  and output files), with no extra copies;
* the same document still crawls successfully end to end, consumed entry
  by entry.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPT = (
    Path(__file__).resolve().parent / "experiments" / "sitemap_file_backed_cancel.py"
)

SIZES_MIB = [8, 128]
CANCEL_LATENCY_LIMIT = 1.0
CLEANUP_AFTER_DECODE_LIMIT = 2.0
CLEANUP_BEFORE_DECODE_LIMIT = 60.0
PYTHON_PEAK_LIMIT = 16 * 1024 * 1024
RSS_PEAK_DELTA_LIMIT = 64 * 1024 * 1024
# Subprocess runs include interpreter startup and the baseline crawl.
TIMEOUT = 600


def _run(size_mib: int) -> dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, "-I", str(SCRIPT), "--size-mib", str(size_mib)],
        check=True,
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
    )
    result: dict[str, Any] = json.loads(proc.stdout.strip().splitlines()[-1])
    return result


@pytest.mark.parametrize("size_mib", SIZES_MIB)
def test_file_backed_cancel_resources(size_mib: int) -> None:
    result = _run(size_mib)
    assert result.get("error") is None

    baseline = result["baseline"]
    # The document still crawls end to end, streamed entry by entry.
    assert baseline["entries"] > 0
    assert baseline["first_url_seconds"] is not None
    assert baseline["first_url_seconds"] < 30

    for phase in ("cancel_before_decode", "cancel_after_decode"):
        cancel = result[phase]
        assert cancel.get("error") is None
        # The caller observes the cancellation immediately.
        assert cancel["cancel_latency_seconds"] < CANCEL_LATENCY_LIMIT
        # Nothing is left behind: no files, no pending decode slots, no
        # success stats, no registry directory.
        assert cancel["files_left"] == []
        assert cancel["pending_decodes"] == 0
        assert cancel["httpcompression_count"] == 0
        assert cancel["registry_dir_exists"] is False
        # The decoded body never enters Python memory.
        assert cancel["traced_python_peak_bytes"] < PYTHON_PEAK_LIMIT

    # Once the worker thread is released, cleanup is prompt when the decode
    # already finished, and bounded by the remaining decode when it had not
    # started (a thread cannot be stopped mid-work).
    after = result["cancel_after_decode"]
    assert after["cleanup_seconds"] < CLEANUP_AFTER_DECODE_LIMIT
    before = result["cancel_before_decode"]
    assert before["cleanup_seconds"] < CLEANUP_BEFORE_DECODE_LIMIT

    # The temporary-disk peak is about the decoded size: the decode's own
    # input and output files, no extra copies.
    assert after["temp_disk_peak_bytes"] >= result["xml_bytes"] * 0.95
    assert after["temp_disk_peak_bytes"] < result["xml_bytes"] * 1.6


def test_file_backed_cancel_rss_is_flat() -> None:
    small = _run(SIZES_MIB[0])
    large = _run(SIZES_MIB[1])
    for phase in ("cancel_before_decode", "cancel_after_decode"):
        rss_delta = large[phase]["rss_peak_bytes"] - small[phase]["rss_peak_bytes"]
        assert rss_delta < RSS_PEAK_DELTA_LIMIT, (phase, rss_delta)
