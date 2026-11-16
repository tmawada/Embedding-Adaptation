"""Drop repeated questions from the dev / test splits of the existing caches, without re-encoding anything.

MIRACL-id repeats some questions under different ids, so a few dev / test queries had a train "twin" with the same
text (and usually the same relevant passage). preprocessing.dataset.drop_twins removes them; methods/prepare.py does
this when it builds a cache (--drop_twins true), and this script applies the same rule to caches built before that:

  methods/cache/splits.json                     rewritten; the old file is kept as splits_with_twins.json
  every other encoder cache (methods/encoders)  receives the same splits.json, so all encoders share one split

Embeddings are stored per query id, so they stay valid. Running the script again changes nothing.

  python3 methods/drop_twins.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from preprocessing.dataset import drop_twins  # noqa: E402
from methods.encoders import ENCODERS  # noqa: E402


def keep_original(cache_dir: str):
    """Save the pre-dedup split once, next to the new one."""
    src, dst = os.path.join(cache_dir, "splits.json"), os.path.join(cache_dir, "splits_with_twins.json")
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copyfile(src, dst)


def main():
    base = os.path.join(HERE, "cache")
    path = os.path.join(base, "splits.json")
    with open(path) as f:
        splits = json.load(f)
    if "twins_excluded" in splits:
        print(f"{os.path.relpath(path, ROOT)} already excludes {len(splits['twins_excluded'])} twins; nothing to drop")
    else:
        keep_original(base)
        new = drop_twins(splits, os.path.join(ROOT, "data"))
        for name in ("train", "dev", "test"):
            removed = sorted(set(splits[name]) - set(new[name]), key=int)
            print(f"{name}: {len(splits[name])} -> {len(new[name])}"
                  + (f", dropped {', '.join(removed)}" if removed else ""))
        with open(path, "w") as f:
            json.dump(new, f)
    for spec in ENCODERS.values():
        if os.path.abspath(spec.cache_dir) == os.path.abspath(base) or not os.path.isdir(spec.cache_dir):
            continue
        keep_original(spec.cache_dir)
        shutil.copyfile(path, os.path.join(spec.cache_dir, "splits.json"))
        print(f"-> {os.path.relpath(os.path.join(spec.cache_dir, 'splits.json'), ROOT)} (same split as the base cache)")


if __name__ == "__main__":
    main()
