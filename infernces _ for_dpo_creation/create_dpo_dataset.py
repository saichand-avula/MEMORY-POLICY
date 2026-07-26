"""
create_dpo_dataset.py — Run inference on the ENTIRE training dataset,
compare model output vs expected_output, and split into:
  - dpo_matched.jsonl    : rows where model output == expected
  - dpo_not_matched.jsonl: rows where model differs (the real DPO pairs)

Each output row has the standard DPO schema:
{
  "prompt":    <the formatted user+memory prompt string>,
  "chosen":    <expected_output JSON string>,   # gold label
  "rejected":  <model_output JSON string>,      # what the model said
  "metadata": {
      "index": <int>,
      "latest_user_message": <str>,
      "existing_memory": [...],
      "parse_error": <bool>,
      "match": <bool>
  }
}

Usage:
    python create_dpo_dataset.py [--adapter adapter/memory-policy-lora]
                                 [--train-file dataset/training.json]
                                 [--out-dir dpo_dataset]
                                 [--batch-size 8]
                                 [--max-new-tokens 256]
                                 [--resume]
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

import config
from prompts import build_messages


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def resolve_device():
    if torch.cuda.is_available():
        print("⚡ CUDA GPU:", torch.cuda.get_device_name(0))
        return "cuda", torch.bfloat16
    elif torch.backends.mps.is_available():
        print("🍎 Apple MPS")
        return "mps", torch.float16
    else:
        print("🐢 CPU mode")
        return "cpu", torch.float32


def load_model_and_tokenizer(adapter_dir: str):
    device, dtype = resolve_device()
    print(f"Loading tokenizer from: {adapter_dir}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)

    # CRITICAL: left-padding required for decoder-only models in batched inference
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model: {config.MODEL_ID} ...")
    base = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto" if device == "cuda" else None,
    )
    print("Wrapping with LoRA adapter ...")
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()

    if device != "cuda":  # device_map="auto" already handles CUDA placement
        model = model.to(device)

    print(f"Model ready. device={device}  dtype={dtype}\n")
    return model, tokenizer, device


def build_prompt_string(tokenizer, example: dict) -> str:
    """Build the full prompt fed to the model (no assistant turn)."""
    messages = build_messages(example)
    messages_no_asst = messages[:-1]
    return tokenizer.apply_chat_template(
        messages_no_asst,
        tokenize=False,
        add_generation_prompt=True,
    )


def run_inference_batch(model, tokenizer, prompts: list[str],
                        device: str, max_new_tokens: int = 256) -> list[str]:
    """Batched inference with left-padding; returns list of raw decoded strings."""
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
        padding_side="left",   # explicit for safety
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out_ids = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    results = []
    for i, ids in enumerate(out_ids):
        # Decode only the newly generated tokens (skip the input prompt)
        input_len = enc["input_ids"].shape[-1]
        raw = tokenizer.decode(ids[input_len:], skip_special_tokens=True).strip()
        results.append(raw)
    return results


def normalize_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def extract_actions(obj) -> list:
    """
    Normalize model output to a list of action dicts.
    Handles two formats the model may emit:
      - {"actions": [{...}]}   (standard)
      - [{...}]                (model outputs list directly)
    """
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        return obj.get("actions", [])
    return []


def actions_match(expected: dict, predicted) -> bool:
    """Order-insensitive comparison of action lists."""
    exp_actions  = extract_actions(expected)
    pred_actions = extract_actions(predicted)
    if len(exp_actions) != len(pred_actions):
        return False
    exp_norm  = sorted(normalize_json(a) for a in exp_actions)
    pred_norm = sorted(normalize_json(a) for a in pred_actions)
    return exp_norm == pred_norm


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Create DPO dataset from training split")
    parser.add_argument("--adapter",        default=config.ADAPTER_SAVE_DIR)
    parser.add_argument("--train-file",     default=config.TRAIN_FILE)
    parser.add_argument("--out-dir",        default="dpo_dataset")
    parser.add_argument("--batch-size",     type=int, default=8,
                        help="Inference batch size (lower if OOM)")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--resume",         action="store_true",
                        help="Skip rows already written to output files")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    matched_path     = out_dir / "dpo_matched.jsonl"
    not_matched_path = out_dir / "dpo_not_matched.jsonl"

    # ── Resume support ────────────────────────────────────────────────────────
    done_indices: set[int] = set()
    if args.resume:
        for fpath in [matched_path, not_matched_path]:
            if fpath.exists():
                with open(fpath) as f:
                    for line in f:
                        try:
                            row = json.loads(line)
                            done_indices.add(row["metadata"]["index"])
                        except Exception:
                            pass
        print(f"[Resume] {len(done_indices)} rows already done — skipping.")

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"Loading training dataset: {args.train_file}")
    with open(args.train_file) as f:
        dataset = json.load(f)
    total = len(dataset)
    print(f"Total training examples: {total:,}")

    # ── Load model ────────────────────────────────────────────────────────────
    model, tokenizer, device = load_model_and_tokenizer(args.adapter)

    # ── Counters ──────────────────────────────────────────────────────────────
    n_matched     = 0
    n_not_matched = 0
    n_parse_error = 0
    start_time    = time.time()

    f_matched     = open(matched_path,     "a", encoding="utf-8")
    f_not_matched = open(not_matched_path, "a", encoding="utf-8")

    try:
        batch_examples: list = []
        batch_indices:  list = []

        def flush_batch():
            nonlocal n_matched, n_not_matched, n_parse_error
            if not batch_examples:
                return

            prompts     = [build_prompt_string(tokenizer, ex) for ex in batch_examples]
            raw_outputs = run_inference_batch(
                model, tokenizer, prompts, device, args.max_new_tokens
            )

            for idx, example, prompt, raw in zip(
                batch_indices, batch_examples, prompts, raw_outputs
            ):
                parse_error = False
                try:
                    predicted = json.loads(raw)
                except json.JSONDecodeError:
                    predicted   = {"actions": [], "raw_output": raw}
                    parse_error = True
                    n_parse_error += 1

                expected = example["expected_output"]
                match    = (not parse_error) and actions_match(expected, predicted)

                dpo_row = {
                    "prompt":   prompt,
                    "chosen":   json.dumps(expected,  ensure_ascii=False, separators=(",", ":")),
                    "rejected": json.dumps(predicted, ensure_ascii=False, separators=(",", ":")),
                    "metadata": {
                        "index":               idx,
                        "latest_user_message": example["latest_user_message"],
                        "existing_memory":     example["existing_memory"],
                        "parse_error":         parse_error,
                        "match":               match,
                    },
                }

                line = json.dumps(dpo_row, ensure_ascii=False) + "\n"
                if match:
                    f_matched.write(line)
                    n_matched += 1
                else:
                    f_not_matched.write(line)
                    n_not_matched += 1

            f_matched.flush()
            f_not_matched.flush()
            batch_examples.clear()
            batch_indices.clear()

        # ── Iterate ───────────────────────────────────────────────────────────
        for idx, example in enumerate(dataset):
            if idx in done_indices:
                continue

            batch_examples.append(example)
            batch_indices.append(idx)

            if len(batch_examples) >= args.batch_size:
                flush_batch()

                processed = n_matched + n_not_matched
                elapsed   = time.time() - start_time
                rate      = processed / elapsed if elapsed > 0 else 0
                eta_s     = (total - processed) / rate if rate > 0 else float("inf")
                print(
                    f"[{idx+1:>7}/{total}] "
                    f"matched={n_matched}  not_matched={n_not_matched}  "
                    f"parse_err={n_parse_error}  "
                    f"rate={rate:.1f} rows/s  ETA={eta_s/60:.1f}min",
                    flush=True,
                )

        flush_batch()  # last partial batch

    finally:
        f_matched.close()
        f_not_matched.close()

    # ── Final summary ─────────────────────────────────────────────────────────
    total_done  = n_matched + n_not_matched
    elapsed_min = (time.time() - start_time) / 60
    match_pct   = 100 * n_matched / total_done if total_done else 0

    print("\n" + "═" * 65)
    print("  DPO DATASET CREATION COMPLETE")
    print("═" * 65)
    print(f"  Total processed      : {total_done:,}")
    print(f"  ✅ Matched           : {n_matched:,}   ({match_pct:.1f}%)")
    print(f"  ❌ Not matched       : {n_not_matched:,}   ({100-match_pct:.1f}%)")
    print(f"  ⚠️  Parse errors      : {n_parse_error:,}")
    print(f"  ⏱️  Elapsed           : {elapsed_min:.1f} min")
    print(f"\n  📁 Matched    → {matched_path}")
    print(f"  📁 Not-matched → {not_matched_path}")
    print("═" * 65)


if __name__ == "__main__":
    main()
