"""Hello-world inference from a verl GSPO hybrid checkpoint.

What this script does
---------------------
A verl FSDP checkpoint for a hybrid LoRA + full-FT run lives in two places:

  ckpt/global_step_N/actor/
    model_world_size_W_rank_R.pt   <- FSDP1 SHARDED_STATE_DICT shards (DTensors).
                                     Contain BOTH the (frozen) base weights AND
                                     the trainable LoRA adapters under keys like
                                     ``base_model.model.<...>.lora_{A,B}.default.weight``.
    trainable_parameters.bin       <- Plain torch.save dict of the unfrozen
                                     non-LoRA params (full-FT first/last layers,
                                     model.norm, lm_head, model.embed_tokens),
                                     stripped of the ``base_model.model.`` prefix.
    huggingface/                   <- config.json, tokenizer*, generation_config.

So to do inference we:
  1. Load the base HF model from --base_model_path.
  2. Apply a LoRA adapter wrapper with the SAME LoRA config used during training
     (rank/alpha/target_modules/layers_to_transform).
  3. Pull the lora_A / lora_B tensors out of the FSDP shards (gathered to full),
     and load them into the PEFT model.
  4. Load trainable_parameters.bin (full-FT layers + embed_tokens + lm_head)
     with strict=False on top.
  5. Optionally merge the LoRA into the base for faster inference.
  6. Generate.

Usage
-----
python scripts/infer_from_gspo_hybrid_ckpt.py \\
    --ckpt /fs/ess/PCON0781/data/khoa/verl/checkpoints/qwen3_5_9b_gspo_hybrid/global_step_20/actor \\
    --base_model_path /fs/ess/PCON0781/data/khoa/Qwen3.5-9B-Phase4.5-step5400 \\
    --tokenizer_path /fs/ess/PCON0781/data/khoa/tokenizerDP/Qwen3.5-9B \\
    --lora_rank 32 --lora_alpha 16 \\
    --first 4 --last 4 \\
    --prompt "Hello, world!" \\
    --merge

This script does NOT need torch.distributed or FSDP — it loads the per-rank
shard files manually and reassembles the LoRA tensors on a single GPU.
"""

import argparse
import glob
import os
import re
from typing import Dict, List

import torch
from torch.distributed._tensor import DTensor


def _to_full(t):
    """Convert a DTensor (FSDP SHARDED_STATE_DICT entry) to a full tensor."""
    if isinstance(t, DTensor):
        return t.full_tensor()
    return t


def load_full_state_dict_from_shards(actor_dir: str) -> Dict[str, torch.Tensor]:
    """Reassemble a full state_dict from FSDP1 SHARDED_STATE_DICT shard files.

    Each shard file contains the same set of keys; values are DTensors sharded
    on dim 0 across the world. Calling ``.full_tensor()`` on any one of them
    issues a collective — but with only one process, ``DTensor`` raised on us;
    so instead we manually concatenate the local shards along dim 0.
    """
    shard_paths = sorted(glob.glob(os.path.join(actor_dir, "model_world_size_*_rank_*.pt")))
    assert shard_paths, f"No shard files in {actor_dir}"
    print(f"Found {len(shard_paths)} shard files")

    # Load all shards
    shards = [torch.load(p, map_location="cpu", weights_only=False) for p in shard_paths]

    # Sort by rank (filename contains rank_R)
    def rank_of(path):
        m = re.search(r"rank_(\d+)\.pt$", path)
        return int(m.group(1))

    pairs = sorted(zip(shard_paths, shards), key=lambda x: rank_of(x[0]))
    shards = [s for _, s in pairs]

    keys = list(shards[0].keys())
    full_sd: Dict[str, torch.Tensor] = {}
    for k in keys:
        per_rank = [s[k] for s in shards]
        # Each is a DTensor; grab its local shard.
        local_tensors = [
            t._local_tensor if isinstance(t, DTensor) else t for t in per_rank
        ]
        # FSDP1 fully-shards on dim 0 (after padding). Concatenate.
        full = torch.cat(local_tensors, dim=0)
        # Trim padding: the global numel must divide cleanly into the original
        # shape stored in the DTensor metadata.
        if isinstance(per_rank[0], DTensor):
            global_shape = per_rank[0].shape
            global_numel = 1
            for d in global_shape:
                global_numel *= d
            if full.numel() > global_numel:
                full = full.flatten()[:global_numel].view(global_shape)
            else:
                full = full.view(global_shape)
        full_sd[k] = full
    return full_sd


def _fix_merged_visual_prefix(merged_dir: str) -> None:
    """Rewrite ``model.language_model.visual.*`` -> ``model.visual.*`` in
    every safetensors file (and any sharded index) under ``merged_dir``.

    PEFT's ``merge_and_unload`` + ``save_pretrained`` on
    ``Qwen3_5ForConditionalGeneration`` ends up nesting the visual tower
    under ``language_model``, but the model class (and vLLM) expect the
    flat ``model.visual.*`` layout. This function fixes the saved files in
    place so the merged checkpoint is directly loadable.
    """
    import json
    import shutil

    from safetensors import safe_open
    from safetensors.torch import save_file

    bad, good = "model.language_model.visual.", "model.visual."

    def remap(k: str) -> str:
        return good + k[len(bad):] if k.startswith(bad) else k

    total = 0
    for fp in sorted(glob.glob(os.path.join(merged_dir, "*.safetensors"))):
        with safe_open(fp, framework="pt") as f:
            keys = list(f.keys())
            meta = f.metadata() or {}
            tensors = {k: f.get_tensor(k) for k in keys}
        new_tensors = {}
        renamed = 0
        for k, v in tensors.items():
            nk = remap(k)
            if nk != k:
                renamed += 1
            new_tensors[nk] = v
        if renamed == 0:
            continue
        tmp = fp + ".tmp"
        save_file(new_tensors, tmp, metadata=meta)
        shutil.move(tmp, fp)
        total += renamed
        print(f"  fix-visual-prefix: {os.path.basename(fp)}: renamed {renamed}")

    for ip in sorted(glob.glob(os.path.join(merged_dir, "*.safetensors.index.json"))):
        with open(ip) as f:
            idx = json.load(f)
        wm = idx.get("weight_map", {})
        new_wm = {remap(k): v for k, v in wm.items()}
        n = sum(1 for k in wm if remap(k) != k)
        if n:
            idx["weight_map"] = new_wm
            with open(ip, "w") as f:
                json.dump(idx, f, indent=2)
            print(f"  fix-visual-prefix: {os.path.basename(ip)}: renamed {n} index entries")

    print(f"  fix-visual-prefix: total tensor keys renamed = {total}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to global_step_N/actor")
    ap.add_argument("--base_model_path", required=True)
    ap.add_argument("--tokenizer_path", required=True)
    ap.add_argument("--lora_rank", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--first", type=int, default=4, help="full-FT first N layers")
    ap.add_argument("--last", type=int, default=4, help="full-FT last N layers")
    ap.add_argument("--prompt", default="Hello, world!")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--merge", action="store_true", help="merge LoRA into base for inference")
    ap.add_argument(
        "--save_adapter_dir",
        default=None,
        help="If set, save the LoRA adapter (PEFT format) to this directory "
        "BEFORE any merge. Reload later with PeftModel.from_pretrained.",
    )
    ap.add_argument(
        "--save_merged_dir",
        default=None,
        help="If set, save the merged base+LoRA+full-FT model to this directory "
        "in standard HF format. Reload later with AutoModelForCausalLM.from_pretrained. "
        "Implies --merge.",
    )
    ap.add_argument(
        "--skip_generate",
        action="store_true",
        help="Skip the test generation step (useful when you only want to save).",
    )
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    if args.save_merged_dir and not args.merge:
        print("--save_merged_dir set; enabling --merge.")
        args.merge = True

    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3_5ForConditionalGeneration, AutoProcessor

    # 1. Tokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # 2. Base model in bf16 on CPU (we'll move to GPU after loading weights)
    print(f"Loading base model from {args.base_model_path} ...")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # 3. Compute hybrid layer ids exactly like training did
    _text_cfg = getattr(model.config, "text_config", model.config)
    num_layers = _text_cfg.num_hidden_layers
    full_layer_ids = list(range(args.first)) + list(range(num_layers - args.last, num_layers))
    lora_layer_ids = [i for i in range(num_layers) if i not in full_layer_ids]
    print(f"num_layers={num_layers}  full_layers={full_layer_ids}  lora_layers={lora_layer_ids[:4]}...{lora_layer_ids[-4:]}")

    # 4. Apply the same LoRA config as training.
    # PEFT forbids ``target_modules="all-linear"`` together with
    # ``layers_to_transform``, so we enumerate the unique linear module
    # *suffixes* (last component of the module name) found inside any of the
    # transformer layers we want LoRA on. This mirrors the fallback that the
    # training-side ``_build_lora_module`` performs.
    import torch.nn as nn

    target_module_names: set[str] = set()
    # Find the per-layer ModuleList. Qwen3.5 multimodal exposes
    # ``model.language_model.layers`` (LM blocks). We restrict our search to
    # those so the LoRA does not get applied to the visual encoder.
    layers_module = None
    for path in ("model.language_model.layers", "model.layers", "language_model.layers"):
        obj = model
        ok = True
        for part in path.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok:
            layers_module = obj
            print(f"[lora] using layer container: {path}  (len={len(layers_module)})")
            break
    assert layers_module is not None, "Could not find transformer layer container"

    for lid in lora_layer_ids:
        layer = layers_module[lid]
        for name, sub in layer.named_modules():
            if isinstance(sub, nn.Linear):
                target_module_names.add(name.split(".")[-1])
    target_modules = sorted(target_module_names)
    print(f"[lora] target_modules={target_modules}")

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        layers_to_transform=lora_layer_ids,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)

    # 5. Load full FSDP-merged state_dict and pull out LoRA tensors.
    # The training run wrapped the LM inside a multimodal container, so saved
    # keys look like ``base_model.model.model.language_model.layers.X.*``. The
    # text-only base model we loaded here exposes ``model.layers.X.*`` instead.
    # Remap ``language_model.`` away (and ``model.visual.`` keys are dropped
    # since this base model has no visual encoder).
    print(f"Loading FSDP shards from {args.ckpt} ...")
    full_sd = load_full_state_dict_from_shards(args.ckpt)

    def remap(k: str) -> str | None:
        if ".visual." in k:
            return None
        return k.replace("model.language_model.", "model.")

    lora_sd = {}
    for k, v in full_sd.items():
        if "lora_" not in k:
            continue
        nk = remap(k)
        if nk is None:
            continue
        lora_sd[nk] = v
    print(f"Found {len(lora_sd)} LoRA tensors after remap (dropped visual-encoder LoRA)")
    missing, unexpected = model.load_state_dict(lora_sd, strict=False)
    n_lora_missing_in_model = sum(1 for k in missing if "lora_" in k)
    print(f"LoRA load: {len(lora_sd)} attempted, {n_lora_missing_in_model} lora keys still missing in model, {len(unexpected)} unexpected")

    # 6. Load trainable_parameters.bin (full-FT layers + embed_tokens)
    tp_path = os.path.join(args.ckpt, "trainable_parameters.bin")
    if os.path.exists(tp_path):
        tp = torch.load(tp_path, map_location="cpu", weights_only=False)
        prefixed = {}
        for k, v in tp.items():
            nk = remap(k)
            if nk is None:
                continue
            if not nk.startswith("base_model.model."):
                nk = f"base_model.model.{nk}"
            prefixed[nk] = v
        m, u = model.load_state_dict(prefixed, strict=False)
        print(f"trainable_parameters.bin: loaded {len(prefixed)} tensors, {len(u)} unexpected")
    else:
        print(f"WARNING: {tp_path} not found")

    # 7a. Save the LoRA adapter BEFORE merging (merge_and_unload destroys it).
    if args.save_adapter_dir:
        os.makedirs(args.save_adapter_dir, exist_ok=True)
        print(f"Saving LoRA adapter to {args.save_adapter_dir} ...")
        model.save_pretrained(args.save_adapter_dir)
        tok.save_pretrained(args.save_adapter_dir)
        print(
            f"Adapter saved. Reload with:\n"
            f"  from peft import PeftModel\n"
            f"  base = AutoModelForCausalLM.from_pretrained('{args.base_model_path}', torch_dtype='bfloat16', trust_remote_code=True)\n"
            f"  model = PeftModel.from_pretrained(base, '{args.save_adapter_dir}')"
        )

    # 7b. Optionally merge LoRA into base for inference / saving as a plain HF model.
    if args.merge:
        print("Merging LoRA into base ...")
        model = model.merge_and_unload()

    if args.save_merged_dir:
        os.makedirs(args.save_merged_dir, exist_ok=True)
        print(f"Saving merged model to {args.save_merged_dir} ...")
        model.save_pretrained(args.save_merged_dir, safe_serialization=True)
        tok.save_pretrained(args.save_merged_dir)
        # PEFT merge nests visual tower keys under ``model.language_model.visual.*``
        # but Qwen3_5ForConditionalGeneration / vLLM expect ``model.visual.*``.
        # Rewrite saved safetensors (and any sharded index) in place.
        _fix_merged_visual_prefix(args.save_merged_dir)
        print(
            f"Merged model saved. Reload with:\n"
            f"  AutoModelForCausalLM.from_pretrained('{args.save_merged_dir}', torch_dtype='bfloat16', trust_remote_code=True)"
        )

    model = model.to(args.device).eval()

    # 8. Generate (optional)
    if args.skip_generate:
        return
    template = (
        "Carefully think through the problem and provide a complete and efficient solution "
        "in an appropriate format that addresses the following task:\n"
        "### Instruction:\n"
        "{}\n\n"
        "### Response:\n"
    )
    formatted_prompt = template.format(args.prompt)
    messages = [{"role": "user", "content": formatted_prompt}]
    chat_text = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tok(chat_text, return_tensors="pt").to(args.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    # Decode only the newly generated tokens (after the prompt)
    generated = out[0][inputs["input_ids"].shape[-1]:]
    print("=" * 60)
    print("PROMPT :", args.prompt)
    print("OUTPUT :", tok.decode(generated, skip_special_tokens=True))
    print("=" * 60)


if __name__ == "__main__":
    main()
