"""
batch_inference_new_testing.py

Runs the LoRA adapter on every example in new_testing_questions.jsonl
and writes results to output/new_testing_inference_results.jsonl

Output schema per line:
  {
    "index": ...,
    "category": ...,
    "latest_user_message": "...",
    "existing_memory": [...],
    "expected": {...},
    "model_output": {...},
    "model_raw": "...",
    "parse_error": true/false,
    "correct": true/false
  }
"""

import argparse
import json
import os
import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

import config
from prompts import build_messages


def resolve_device():
    if torch.cuda.is_available():
        print("⚡ Using CUDA GPU")
        return "cuda", torch.bfloat16
    elif torch.backends.mps.is_available():
        print("🍎 Using Apple MPS")
        return "mps", torch.float16
    else:
        print("🐢 Using CPU")
        return "cpu", torch.float32


def load_model_and_tokenizer(adapter_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    device, dtype = resolve_device()
    base = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    if device != "cpu":
        model = model.to(device)
    return model, tokenizer


def predict(model, tokenizer, existing_memory, latest_user_message, max_new_tokens=256):
    example = {
        "existing_memory": existing_memory,
        "latest_user_message": latest_user_message,
        "expected_output": {},
    }
    messages = build_messages(example)
    messages_no_assistant = messages[:-1]

    prompt = tokenizer.apply_chat_template(
        messages_no_assistant,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    raw = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    try:
        parsed = json.loads(raw)
        return parsed, False, raw
    except json.JSONDecodeError:
        return {"raw_output": raw, "parse_error": True}, True, raw


def normalize(obj):
    """Sort keys for stable comparison."""
    try:
        return json.dumps(obj, sort_keys=True)
    except Exception:
        return str(obj)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=config.ADAPTER_SAVE_DIR)
    parser.add_argument("--input", default="new_testing_questions.jsonl")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--output", default="new_testing_inference_results.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, args.output)

    records = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if args.limit:
        records = records[:args.limit]

    total = len(records)
    print(f"📄 Loaded {total} records from {args.input}")
    print(f"Loading adapter from: {args.adapter}\n")

    model, tokenizer = load_model_and_tokenizer(args.adapter)

    parse_error_count = 0
    correct_count = 0
    category_stats = {}

    with open(out_path, "w") as out_f:
        for i, record in enumerate(records):
            idx = record.get("index", i)
            user_msg = record.get("latest_user_message", "")
            existing_mem = record.get("existing_memory", [])
            expected = record.get("expected", {})
            category = record.get("category", "unknown")

            print(f"[{i+1}/{total}] idx={idx} cat={category} | {user_msg[:55]!r}")

            model_output, parse_error, raw = predict(
                model, tokenizer, existing_mem, user_msg,
                max_new_tokens=args.max_new_tokens,
            )

            correct = normalize(model_output) == normalize(expected)

            if parse_error:
                parse_error_count += 1
            if correct:
                correct_count += 1

            # Per-category tracking
            if category not in category_stats:
                category_stats[category] = {"total": 0, "correct": 0, "parse_error": 0}
            category_stats[category]["total"] += 1
            if correct:
                category_stats[category]["correct"] += 1
            if parse_error:
                category_stats[category]["parse_error"] += 1

            result = {
                "index": idx,
                "category": category,
                "latest_user_message": user_msg,
                "existing_memory": existing_mem,
                "expected": expected,
                "model_output": model_output,
                "model_raw": raw,
                "parse_error": parse_error,
                "correct": correct,
            }
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()

            print(f"         parse_error={parse_error} | correct={correct}")

    print("\n" + "=" * 60)
    print(f"✅ Done! Results saved to: {out_path}")
    print(f"   Total          : {total}")
    print(f"   Parse errors   : {parse_error_count} ({100*parse_error_count/total:.1f}%)")
    print(f"   Correct        : {correct_count} ({100*correct_count/total:.1f}%)")
    print("\n--- Per-Category Breakdown ---")
    for cat, s in sorted(category_stats.items()):
        pct = 100 * s["correct"] / s["total"] if s["total"] else 0
        print(f"  {cat}: {s['correct']}/{s['total']} correct ({pct:.0f}%) | parse_err={s['parse_error']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
