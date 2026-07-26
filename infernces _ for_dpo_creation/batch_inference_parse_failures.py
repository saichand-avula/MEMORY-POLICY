"""
batch_inference_parse_failures.py

Runs the LoRA adapter on every example in parse_failures_excluded_from_dpo.jsonl
and writes an output JSONL with the model's prediction matched to the original input.

Output schema per line:
  {
    "index": <original metadata index>,
    "latest_user_message": "...",
    "existing_memory": [...],
    "chosen": "...",            # ground-truth from the dataset
    "model_output": {...},      # parsed JSON or {"raw_output": ..., "parse_error": true}
    "model_raw": "...",         # raw decoded string from model
    "parse_error": true/false,  # whether model output failed to parse
    "correct": true/false       # whether model output matches chosen (normalized JSON)
  }

Usage:
    python batch_inference_parse_failures.py
    python batch_inference_parse_failures.py --adapter adapter/memory-policy-lora \\
        --input parse_failures_excluded_from_dpo.jsonl \\
        --output parse_failures_inference_results.jsonl
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

# ─────────────────────────────────────────────────────────────────────────────

def resolve_device() -> tuple[str, torch.dtype]:
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
    return model, tokenizer, device


def predict(model, tokenizer, existing_memory: list, latest_user_message: str,
            max_new_tokens: int = 256) -> tuple[dict, bool, str]:
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


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch inference on parse failures")
    parser.add_argument("--adapter", default=config.ADAPTER_SAVE_DIR)
    parser.add_argument("--input", default="parse_failures_excluded_from_dpo.jsonl")
    parser.add_argument("--output", default="parse_failures_inference_results.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N records (for testing)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Load all records
    records = []
    with open(args.input, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if args.limit:
        records = records[: args.limit]

    total = len(records)
    print(f"📄 Loaded {total} records from {args.input}")
    print(f"Loading adapter from: {args.adapter}\n")

    model, tokenizer, _ = load_model_and_tokenizer(args.adapter)

    parse_error_count = 0
    correct_count = 0

    with open(args.output, "w") as out_f:
        for i, record in enumerate(records):
            meta = record.get("metadata", {})
            idx = meta.get("index", i)
            user_msg = meta.get("latest_user_message", "")
            existing_mem = meta.get("existing_memory", [])
            chosen = record.get("chosen", "")

            print(f"[{i+1}/{total}] idx={idx} | msg: {user_msg[:60]!r}")

            model_output, parse_error, raw = predict(
                model, tokenizer, existing_mem, user_msg,
                max_new_tokens=args.max_new_tokens,
            )

            # Normalize for comparison: re-serialize both sides
            try:
                chosen_parsed = json.loads(chosen)
                chosen_normalized = json.dumps(chosen_parsed, sort_keys=True)
            except Exception:
                chosen_normalized = chosen

            try:
                model_normalized = json.dumps(model_output, sort_keys=True)
            except Exception:
                model_normalized = str(model_output)

            correct = (chosen_normalized == model_normalized)

            if parse_error:
                parse_error_count += 1
            if correct:
                correct_count += 1

            result = {
                "index": idx,
                "latest_user_message": user_msg,
                "existing_memory": existing_mem,
                "chosen": chosen,
                "model_output": model_output,
                "model_raw": raw,
                "parse_error": parse_error,
                "correct": correct,
            }
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()

            print(f"         parse_error={parse_error} | correct={correct}")

    print("\n" + "=" * 60)
    print(f"✅ Done! Results saved to: {args.output}")
    print(f"   Total records     : {total}")
    print(f"   Parse errors      : {parse_error_count} ({100*parse_error_count/total:.1f}%)")
    print(f"   Correct outputs   : {correct_count} ({100*correct_count/total:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
