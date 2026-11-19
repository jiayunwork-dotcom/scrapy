# Per-item media storage regression reproduction

This directory contains a standalone, dependency-light reproduction of the
media-pipeline bug where several items share one download URL but need their
own stored files.

## What it covers

`repro_media_targets.py` exercises, against temporary directories with an
in-memory controllable download callback:

FilesPipeline

- **F1** two items with the same URL while the download is in flight: each
  gets its own path, checksum and on-disk file;
- **F2** a third item arriving after the download finished still gets its own
  file instead of the first item's cached result;
- **F3** identical targets (same item twice) still download only once;
- **F4** one target directory failing to write does not stop the other target
  from being delivered; failures go through the ordinary item failure channel.

ImagesPipeline

- **I1** full hit: fresh original image *and* fresh item-specific thumbnails
  means no download (`uptodate`);
- **I2** fresh original image but a missing item-specific thumbnail forces a
  re-download and thumbnail generation;
- **I3** fresh original image but an expired thumbnail forces a re-download;
- **I4** two concurrent items share one original image and one download, each
  reads back its own thumbnail;
- **I5** one thumbnail directory that cannot be written does not prevent the
  other item from receiving its thumbnail.

## How to run

From the repository root (the project must be installed, see INSTALL.md):

```
python3 -P repro/repro_media_targets.py
```

`-P` keeps the interpreter from adding the current directory to `sys.path`
(so a planted `json.py` next to the script cannot shadow stdlib). Use `-I`
instead if the project dependencies live in a virtual environment rather than
user site-packages.

The script exits with status 0 only when every check passes. On the unpatched
2.13.3 code it fails 20 checks (see `results-baseline.txt`); with the fix it
passes all 34 (`results-fixed.txt`).

The same scenarios are also covered by the pytest suite in
`tests/test_pipeline_media_targets.py`:

```
python3 -m pytest tests/test_pipeline_media_targets.py -v
```
