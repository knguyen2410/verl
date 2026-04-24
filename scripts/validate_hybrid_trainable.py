"""Validate the hybrid trainable parameter layout for the GSPO run.

Loads the base model + PEFT adapter the same way ``_build_lora_module`` does
when ``lora_adapter_path`` is set, then unfreezes the full-FT layers, and
prints a per-layer breakdown so you can confirm that ONLY:

    - LoRA adapters on layers ``[first..N-last)``
    - Base weights of layers ``[0..first) + [N-last..N)``  (full FT)
    - lm_head, model.norm  (when last_layers > 0)

are trainable.

Usage::

    python scripts/validate_hybrid_trainable.py \\
        --base /fs/ess/PCON0781/data/khoa/Qwen3.5-9B-Phase4.5-step5400 \\
        --adapter /fs/ess/PCON0781/data/khoa/verl/checkpoint-5400 \\
        --first 4 --last 4
"""

import argparse
import re
from collections import defaultdict

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--first", type=int, default=4)
    ap.add_argument("--last", type=int, default=4)
    args = ap.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    from verl.utils.model import unfreeze_params_by_patterns
    from verl.workers.fsdp_workers import (
        _build_full_finetune_patterns_from_layer_ids,
        _build_hybrid_layer_selection,
        _get_num_hidden_layers,
    )

    print(f"Loading base model: {args.base}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True
    )

    num_layers = _get_num_hidden_layers(model.config)
    full_layer_ids, lora_layer_ids = _build_hybrid_layer_selection(num_layers, args.first, args.last)
    print(f"num_layers={num_layers}")
    print(f"full_layer_ids={sorted(full_layer_ids)}")
    print(f"lora_layer_ids={lora_layer_ids}")

    # 1. PEFT load (freezes base, makes LoRA trainable)
    print(f"\nLoading PEFT adapter: {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)

    # 2. Unfreeze full-FT layers (mirrors _build_lora_module)
    full_finetune_modules = _build_full_finetune_patterns_from_layer_ids(
        full_layer_ids, include_final_modules=args.last > 0
    )
    print(f"\nfull_finetune_modules patterns: {full_finetune_modules}")
    n_unfrozen, n_matched = unfreeze_params_by_patterns(model, full_finetune_modules)
    print(f"Unfroze {n_unfrozen} params (matched {n_matched} names)")

    # 3. Audit
    print("\n" + "=" * 80)
    print("TRAINABLE PARAMETER AUDIT")
    print("=" * 80)

    lora_by_layer = defaultdict(int)
    base_by_layer = defaultdict(int)
    other_trainable = []
    visual_lora = 0
    visual_base = 0
    total_trainable_numel = 0
    total_lora_numel = 0
    total_full_numel = 0

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        total_trainable_numel += p.numel()
        is_lora = "lora_" in name or "modules_to_save" in name
        if is_lora:
            total_lora_numel += p.numel()
        else:
            total_full_numel += p.numel()

        if "visual" in name:
            if is_lora:
                visual_lora += 1
            else:
                visual_base += 1
            continue

        m = re.search(r"\.layers\.(\d+)\.", name)
        if m:
            lid = int(m.group(1))
            if is_lora:
                lora_by_layer[lid] += 1
            else:
                base_by_layer[lid] += 1
        else:
            other_trainable.append((name, p.numel(), "lora" if is_lora else "full"))

    print(f"\nLoRA-trainable LM layers: {sorted(lora_by_layer.keys())}")
    print(f"Full-FT-trainable LM layers: {sorted(base_by_layer.keys())}")
    print(f"Visual LoRA params: {visual_lora}, Visual base trainable: {visual_base}")
    print(f"Other trainable (non-layer): {len(other_trainable)}")
    for n, s, k in other_trainable:
        print(f"   - [{k:4s}] {n}  ({s:,} elems)")

    print(f"\nTotal trainable: {total_trainable_numel:,} elements")
    print(f"   LoRA:    {total_lora_numel:,}")
    print(f"   Full FT: {total_full_numel:,}")

    # 4. Verdict
    expected_lora = set(lora_layer_ids)
    expected_full = full_layer_ids
    actual_lora = set(lora_by_layer.keys())
    actual_full = set(base_by_layer.keys())

    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    print(f"LoRA layers  : expected={sorted(expected_lora)}\n               actual  ={sorted(actual_lora)}\n               OK={expected_lora == actual_lora}")
    print(f"Full-FT layers: expected={sorted(expected_full)}\n                actual  ={sorted(actual_full)}\n                OK={expected_full == actual_full}")
    overlap = expected_lora & actual_full
    if overlap:
        print(f"WARNING: layers in both LoRA and full-FT trainable: {sorted(overlap)}")
    if visual_lora or visual_base:
        print(f"WARNING: {visual_lora} visual LoRA + {visual_base} visual base params are trainable")


if __name__ == "__main__":
    main()
