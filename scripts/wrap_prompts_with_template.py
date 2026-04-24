"""Wrap user prompts in a fixed Instruction/Response template.

Reads the parquet datasets under --src-root (default: data/) and writes
templated copies under --dst-root (default: data_templated/). Only the
``content`` of the LAST ``role == "user"`` message is rewritten;
system messages and earlier turns are kept as-is so multi-turn structure
is preserved.

Usage
-----
python scripts/wrap_prompts_with_template.py \
    --src-root data --dst-root data_templated

Then in the run script:

    DATA_DIR=$PWD/data_templated bash examples/gspo_trainer/run_qwen3_5_0_8b_gspo.sh
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pyarrow.parquet as pq
import pandas as pd

DEFAULT_TEMPLATE = """
Carefully think through the problem and provide a complete and efficient solution in an appropriate format that addresses the following task:
### Instruction:
{}

### Response:
"""


def wrap_messages(messages, template: str):
    """Rewrite the last user message's content using ``template``.

    ``messages`` is the list-of-dicts that verl expects under the prompt key.
    Returns a new list (does not mutate input). Accepts numpy arrays too,
    since pandas materializes parquet ``list<struct>`` columns as ndarrays.
    """
    try:
        seq = list(messages)
    except TypeError:
        return messages
    if not seq:
        return seq

    out = [dict(m) if isinstance(m, dict) else m for m in seq]
    # find the LAST user message and wrap it
    for i in range(len(out) - 1, -1, -1):
        m = out[i]
        if isinstance(m, dict) and m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, str):
                m["content"] = template.format(content)
            break
    return out


def process_file(src: Path, dst: Path, template: str, prompt_key: str) -> int:
    table = pq.read_table(src)
    df = table.to_pandas()
    if prompt_key not in df.columns:
        print(f"[skip] {src} has no '{prompt_key}' column; copying as-is")
        dst.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dst, index=False)
        return 0

    df[prompt_key] = df[prompt_key].apply(lambda m: wrap_messages(m, template))
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst, index=False)
    return len(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", default="data")
    ap.add_argument("--dst-root", default="data_templated")
    ap.add_argument("--prompt-key", default="prompt")
    ap.add_argument(
        "--template",
        default=DEFAULT_TEMPLATE,
        help="Format string with a single {} placeholder for the user content.",
    )
    args = ap.parse_args()

    if "{}" not in args.template:
        raise SystemExit("--template must contain a single '{}' placeholder.")

    src_root = Path(args.src_root).resolve()
    dst_root = Path(args.dst_root).resolve()
    if not src_root.exists():
        raise SystemExit(f"src-root does not exist: {src_root}")

    parquets = sorted(src_root.rglob("*.parquet"))
    if not parquets:
        raise SystemExit(f"no parquet files under {src_root}")

    total = 0
    for src in parquets:
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        n = process_file(src, dst, args.template, args.prompt_key)
        print(f"[ok] {rel}: {n} rows -> {dst}")
        total += n
    print(f"Done. {total} rows across {len(parquets)} files written under {dst_root}.")


if __name__ == "__main__":
    main()
