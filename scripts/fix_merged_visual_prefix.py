#!/usr/bin/env python
"""Rewrite a merged Qwen3.5 VL safetensors checkpoint to fix the visual key
prefix.

Background
----------
After ``PeftModel.merge_and_unload()`` + ``save_pretrained``, the visual tower
weights end up nested as ``model.language_model.visual.*`` instead of the
correct ``model.visual.*`` that ``Qwen3_5ForConditionalGeneration`` (and
hence vLLM) expects. This script rewrites the safetensors file(s) in place,
renaming those keys.

Usage
-----
    python scripts/fix_merged_visual_prefix.py \
        --merged_dir /path/to/merged_model
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

from safetensors import safe_open
from safetensors.torch import save_file


BAD_PREFIX = "model.language_model.visual."
GOOD_PREFIX = "model.visual."


def remap_key(k: str) -> str:
    if k.startswith(BAD_PREFIX):
        return GOOD_PREFIX + k[len(BAD_PREFIX):]
    return k


def fix_file(path: str) -> int:
    """Rewrite a single safetensors file in place. Returns # keys renamed."""
    with safe_open(path, framework="pt") as f:
        keys = list(f.keys())
        meta = f.metadata() or {}
        tensors = {k: f.get_tensor(k) for k in keys}

    new_tensors = {}
    n_renamed = 0
    for k, v in tensors.items():
        nk = remap_key(k)
        if nk != k:
            n_renamed += 1
        new_tensors[nk] = v

    if n_renamed == 0:
        return 0

    tmp = path + ".tmp"
    save_file(new_tensors, tmp, metadata=meta)
    shutil.move(tmp, path)
    return n_renamed


def fix_index(index_path: str) -> int:
    with open(index_path) as f:
        idx = json.load(f)
    wm = idx.get("weight_map", {})
    new_wm = {}
    n_renamed = 0
    for k, v in wm.items():
        nk = remap_key(k)
        if nk != k:
            n_renamed += 1
        new_wm[nk] = v
    if n_renamed == 0:
        return 0
    idx["weight_map"] = new_wm
    with open(index_path, "w") as f:
        json.dump(idx, f, indent=2)
    return n_renamed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged_dir", required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.merged_dir, "*.safetensors")))
    if not files:
        raise SystemExit(f"No .safetensors files found in {args.merged_dir}")

    total_renamed = 0
    for fp in files:
        n = fix_file(fp)
        print(f"  {os.path.basename(fp)}: renamed {n} keys")
        total_renamed += n

    index_files = sorted(
        glob.glob(os.path.join(args.merged_dir, "*.safetensors.index.json"))
    )
    for ip in index_files:
        n = fix_index(ip)
        print(f"  {os.path.basename(ip)}: renamed {n} weight_map entries")

    print(f"Done. Total keys renamed in tensor files: {total_renamed}")


if __name__ == "__main__":
    main()
