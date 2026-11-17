"""Pytest entry point for the file-backed sitemap resource experiment.

Each combination of document size (4 MiB vs 40 MiB of XML) and compression
mode (plain, HTTP gzip, double gzip) runs in its own subprocess, so the peak
allocation of one configuration is measured in isolation and is never polluted
by another crawl. The experiment script itself does the measurement and emits
a JSON line; this module only runs it and checks the thresholds:

* the Python allocation peak difference between 40 MiB and 4 MiB must stay
  under 8 MiB;
* the RSS peak difference must stay under 24 MiB;
* the first URL must be produced without waiting for the whole download and
  decode to finish;
* no whole-document body may be held in Python memory at any point.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parent
    / "experiments"
    / "sitemap_file_backed_resources.py"
)

MODES = ["plain", "gzip", "gzip-gzip"]
PYTHON_PEAK_DELTA_LIMIT = 8 * 1024 * 1024
RSS_PEAK_DELTA_LIMIT = 24 * 1024 * 1024
# Subprocess runs include interpreter startup; give them room beyond the
# in-process timeout.
TIMEOUT = 600


def _run(size_mib: int, mode: str) -> dict:
    proc = subprocess.run(
        [sys.executable, "-I", str(SCRIPT), "--size-mib", str(size_mib),
         "--mode", mode],
        check=True,
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("mode", MODES)
def test_file_backed_memory_growth_is_flat(mode: str) -> None:
    small = _run(4, mode)
    large = _run(40, mode)

    # Both crawls actually consumed the document and produced the same number
    # of distinct pages from the fixed 32-URL pool.
    assert small["error"] is None
    assert large["error"] is None
    assert small["items_scraped"] == 32
    assert large["items_scraped"] == 32
    # The large document has roughly ten times as many entries.
    assert large["entries"] > small["entries"] * 8

    python_delta = (
        large["traced_python_peak_bytes"] - small["traced_python_peak_bytes"]
    )
    rss_delta = large["rss_peak_bytes"] - small["rss_peak_bytes"]
    assert python_delta < PYTHON_PEAK_DELTA_LIMIT, python_delta
    assert rss_delta < RSS_PEAK_DELTA_LIMIT, rss_delta

    # The whole decoded document is on temporary disk, not in Python memory.
    assert large["temp_disk_peak_bytes"] >= large["xml_bytes"] * 0.95

    # The first URL is produced well before a sequential parse of 40 MiB could
    # finish, i.e. parsing is incremental rather than whole-document.
    assert large["first_url_seconds"] is not None
    assert large["first_url_seconds"] < 5.0
