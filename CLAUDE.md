# CLAUDE.md

## What this repo is

A fork of [open-mmlab/mmrotate](https://github.com/open-mmlab/mmrotate) (`dev-1.x`, mmengine / mmdet 3.x based)
used on an internet-connected machine to develop new algorithms (backbones, heads, detectors, data tools, configs).
The work done here is later **merged into a separate mmrotate fork elsewhere** that has its own divergent changes.

Every change must therefore be easy to merge: new work goes into new files and new folders, and edits to existing
upstream files are kept to the bare minimum.

## Merge-friendly change rules

1. **Put new features in new files or new folders.** Use a self-contained package per feature, e.g.
   `mmrotate/models/backbones/<feature>/`, `configs/<feature>/`, `tests/test_models/.../test_<feature>.py`,
   `tools/<area>/<new_script>.py`.
2. **Touch existing upstream files only to wire things in**, typically one import line plus one `__all__` entry in
   the parent `__init__.py` so the registry picks the module up. Don't refactor, reformat or "clean up" upstream
   code while you're there. Each extra line edited in an upstream file is a possible merge conflict.
3. **Don't modify upstream classes in place.** To change upstream behaviour, subclass or wrap it in a new file and
   register it under a new name in the registry (`MODELS`, `DATASETS`, `TRANSFORMS`, ...).
4. **Vendored third-party code** gets its own file, keeps its license and source attribution in a header comment,
   and is excluded from reformatting (see below).
5. **Dependency or version bumps** (e.g. the mmcv/mmdet max versions in `mmrotate/__init__.py`) should be small,
   separate and called out in the commit message, because they're the most likely to conflict.
6. **Each feature ships with tests** under `tests/` (new files) and, where useful, a small proxy-data smoke config so
   the pipeline can be checked without the full dataset.

Commit `5ec839a` (C-RADIOv4 backbone + ViT-CoMer adapter) is the reference example:

- the feature lives entirely in `mmrotate/models/backbones/cradiov4/`, `configs/cradio/`, new test files and
  `tools/data/dota/make_proxy_dota.py`;
- the only upstream edits are a 2-line registration in `mmrotate/models/backbones/__init__.py` and a version bump in
  `mmrotate/__init__.py`.

## Code style for new code: ruff, annotations required

The upstream flake8/isort/yapf setup (`setup.cfg`, `.pre-commit-config.yaml`) stays as it is for upstream files.
**New feature packages are linted with ruff instead**, using a `ruff.toml` local to the package so the
repo-wide tooling isn't changed. Copy `mmrotate/models/backbones/cradiov4/ruff.toml` as the template:

- `target-version = "py310"`, `line-length = 100`, single quotes (matches upstream style);
- rule set `E, W, F, I, UP, B, N, ANN`. **`ANN` is mandatory**: every function and method in new code needs full
  parameter and return type annotations (including `-> None`);
- `known-first-party = ["mmrotate"]`;
- vendored files go in `per-file-ignores` and the formatter `exclude`.

New code uses modern syntax (`X | None`, `collections.abc` types, built-in generics) and keeps the
`# Copyright (c) OpenMMLab. All rights reserved.` header that the rest of the codebase has.

Before committing, run on the new package:

```bash
ruff check  mmrotate/models/backbones/<feature> --config mmrotate/models/backbones/<feature>/ruff.toml
ruff format mmrotate/models/backbones/<feature> --config mmrotate/models/backbones/<feature>/ruff.toml
```

New test files and tool scripts that belong to a feature follow the same annotation rules.

## Configs

New configs go in their own `configs/<feature>/` folder. The C-RADIOv4 configs are YAML (`_base_:` inheritance,
mmengine-native) and reuse upstream components (e.g. `CSPNeXtPAFPN`, `RotatedRTMDetSepBNHead`) by registry name
rather than copying them.

## Useful commands

```bash
pytest tests/test_models/test_backbones/test_cradiov4.py   # run a feature's tests
python tools/train.py configs/<feature>/<config>.yaml       # train
python tools/test.py  configs/<feature>/<config>.yaml <ckpt>
```
