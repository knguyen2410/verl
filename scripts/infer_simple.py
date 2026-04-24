"""
Simple inference script for a fully-saved Qwen3.5 model directory.

Usage:
    python scripts/infer_simple.py \
        --model_path model/Qwen3.5-9B-Phase5-step1400 \
        --prompt "Prove Cauchy-Schwarz inequatlity wrong" \
        --max_new_tokens 4096

    # Multiple prompts from a file (one per line):
    python scripts/infer_simple.py \
        --model_path model/Qwen3.5-9B-step400 \
        --prompt_file prompts.txt \
        --max_new_tokens 2048

    # Sampling:
    python scripts/infer_simple.py \
        --model_path model/Qwen3.5-9B-step400 \
        --prompt "Write a haiku about AI." \
        --do_sample --temperature 0.7 --top_p 0.9
"""

import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


INSTRUCTION_TEMPLATE = (
    "Carefully think through the problem and provide a complete and efficient solution "
    "in an appropriate format that addresses the following task:\n"
    "### Instruction:\n"
    "{}\n\n"
    "### Response:\n"
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run inference on a saved Qwen3.5 model.")
    p.add_argument("--model_path", required=True, help="Path to saved model directory.")
    p.add_argument("--tokenizer_path", default=None,
                   help="Tokenizer directory. Defaults to --model_path.")
    p.add_argument("--prompt", default=None, help="Single prompt string.")
    p.add_argument("--prompt_file", default=None,
                   help="Text file with one prompt per line.")
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--do_sample", action="store_true")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no_instruction_template", action="store_true",
                   help="Skip wrapping the prompt in the instruction template.")
    return p


def load_model_and_tokenizer(model_path: str, tokenizer_path: str | None, device: str):
    tok_path = tokenizer_path or model_path
    print(f"Loading tokenizer from {tok_path} ...")
    tok = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)

    print(f"Loading model from {model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded on {device}. Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    return model, tok


def generate_one(model, tok, raw_prompt: str, args) -> str:
    if args.no_instruction_template:
        user_content = raw_prompt
    else:
        user_content = INSTRUCTION_TEMPLATE.format(raw_prompt)

    messages = [{"role": "user", "content": user_content}]
    chat_text = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tok(chat_text, return_tensors="pt").to(args.device)

    # Build a list of EOS token ids: include both <|im_end|> (chat stop) and
    # the model config's eos_token_id (e.g. <|endoftext|>).  The generation_config
    # only stores one of them, so we look up both explicitly.
    eos_ids: list[int] = []
    for tok_str in ("<|im_end|>", "<|endoftext|>"):
        tid = tok.convert_tokens_to_ids(tok_str)
        if tid is not None and tid != tok.unk_token_id:
            eos_ids.append(tid)
    if not eos_ids:
        eos_ids = [tok.eos_token_id]

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        eos_token_id=eos_ids,
        pad_token_id=eos_ids[0],
    )
    if args.do_sample:
        gen_kwargs["temperature"] = args.temperature
        gen_kwargs["top_p"] = args.top_p

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    generated_ids = out[0][inputs["input_ids"].shape[-1]:]
    return tok.decode(generated_ids, skip_special_tokens=True)


def main():
    args = build_parser().parse_args()

    if args.prompt is None and args.prompt_file is None:
        print("Error: provide --prompt or --prompt_file.", file=sys.stderr)
        sys.exit(1)

    model, tok = load_model_and_tokenizer(args.model_path, args.tokenizer_path, args.device)

    prompts: list[str] = []
    if args.prompt:
        prompts.append(args.prompt)
    if args.prompt_file:
        with open(args.prompt_file) as f:
            prompts.extend(line.rstrip("\n") for line in f if line.strip())

    for i, prompt in enumerate(prompts):
        output = generate_one(model, tok, prompt, args)
        print("=" * 60)
        print(f"[{i+1}/{len(prompts)}] PROMPT : {prompt}")
        print(f"[{i+1}/{len(prompts)}] OUTPUT :")
        print(output)
        print("=" * 60)


if __name__ == "__main__":
    main()
