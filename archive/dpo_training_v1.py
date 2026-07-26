"""
dpo_training.py — DPO fine-tuning on top of the existing SFT LoRA adapter.

Pipeline:
  1. Load tokenizer + SFT LoRA adapter (starting checkpoint)
  2. Load DPO dataset (dpo_matched + dpo_not_matched from dpo_dataset/)
  3. Attach a fresh LoRA on top for DPO stage
  4. Train with DPOTrainer (TRL)
  5. Save new adapter  → adapter/memory-policy-dpo
  6. Evaluate loss on  dataset/testing.json  (SFT-style perplexity eval)

Usage:
    python dpo_training.py
    python dpo_training.py --dataset dpo_dataset/dpo_matched.jsonl   # matched only
    python dpo_training.py --epochs 1 --beta 0.1
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import DPOTrainer, DPOConfig

import config
from prompts import build_messages

# ── Logging setup ─────────────────────────────────────────────────────────────

RUN_ID    = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR   = "logs"
LOG_FILE  = os.path.join(LOG_DIR, f"dpo_training_{RUN_ID}.log")

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="w"),
    ],
)
log = logging.getLogger(__name__)

log.info("=" * 70)
log.info("  Memory Policy — DPO Training  (run_id=%s)", RUN_ID)
log.info("  Log file: %s", LOG_FILE)
log.info("=" * 70)


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sft-adapter", default=config.ADAPTER_SAVE_DIR,
                   help="Path to the SFT LoRA adapter to start from")
    p.add_argument("--dataset", default="dataset/dpo_final_dataset.jsonl",
                   help="DPO JSONL file with prompt/chosen/rejected fields")
    p.add_argument("--output-adapter", default="adapter/memory-policy-dpo",
                   help="Where to save the DPO-trained LoRA adapter")
    p.add_argument("--output-dir", default="output/dpo_checkpoints",
                   help="Trainer checkpoint directory")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--beta", type=float, default=0.1,
                   help="DPO beta (KL penalty coefficient)")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--eval-split", type=float, default=0.05,
                   help="Fraction of DPO data held out for eval")
    p.add_argument("--test-file", default=config.TEST_FILE,
                   help="SFT testing.json for final loss evaluation")
    return p.parse_args()


# ── Dataset loading ────────────────────────────────────────────────────────────

def load_dpo_records(args) -> list[dict]:
    """Load prompt/chosen/rejected records from dpo_final_dataset.jsonl."""
    fpath = args.dataset
    if not os.path.exists(fpath):
        log.error("DPO dataset not found: %s", fpath)
        sys.exit(1)

    records = []
    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if all(k in rec for k in ("prompt", "chosen", "rejected")):
                records.append({
                    "prompt":   rec["prompt"],
                    "chosen":   rec["chosen"],
                    "rejected": rec["rejected"],
                })

    log.info("Loaded %d DPO records from %s", len(records), fpath)
    return records


def build_hf_dataset(records: list[dict], eval_split: float):
    """Split into train/eval and return HF Datasets."""
    import random
    random.seed(config.SEED)
    random.shuffle(records)

    n_eval  = max(1, int(len(records) * eval_split))
    n_train = len(records) - n_eval

    train_ds = Dataset.from_list(records[:n_train])
    eval_ds  = Dataset.from_list(records[n_train:])
    log.info("DPO split — train: %d | eval: %d", n_train, n_eval)
    return train_ds, eval_ds


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model_for_dpo(sft_adapter_dir: str):
    """
    Load base model in 4-bit, wrap with the SFT LoRA, then add a new
    trainable LoRA on top for the DPO stage.
    """
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype    = torch.bfloat16 if use_bf16 else torch.float32

    log.info("Loading tokenizer from SFT adapter: %s", sft_adapter_dir)
    tokenizer = AutoTokenizer.from_pretrained(sft_adapter_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # DPO needs left-padding

    # 4-bit quantisation
    bnb_cfg = None
    if torch.cuda.is_available():
        try:
            import bitsandbytes  # noqa
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
            log.info("4-bit NF4 quantisation enabled.")
        except ImportError:
            log.warning("bitsandbytes not found — loading in full precision.")

    log.info("Loading base model: %s", config.MODEL_ID)
    base = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        quantization_config=bnb_cfg,
        torch_dtype=dtype if bnb_cfg is None else None,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    base.config.use_cache = False

    if bnb_cfg is not None:
        base = prepare_model_for_kbit_training(base)

    # Load SFT LoRA weights
    log.info("Loading SFT LoRA from: %s", sft_adapter_dir)
    model = PeftModel.from_pretrained(base, sft_adapter_dir, is_trainable=True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    log.info("Trainable params: %s / %s (%.2f%%)",
             f"{trainable:,}", f"{total:,}", 100 * trainable / total)

    return model, tokenizer


# ── DPO training ───────────────────────────────────────────────────────────────

def run_dpo_training(model, tokenizer, train_ds, eval_ds, args):
    os.makedirs(args.output_dir, exist_ok=True)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    dpo_cfg = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        bf16=use_bf16,
        fp16=False,
        gradient_checkpointing=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        seed=config.SEED,
        beta=args.beta,
        max_length=args.max_length,
    )

    log.info("Building DPOTrainer (beta=%.3f, epochs=%d, lr=%.2e)…",
             args.beta, args.epochs, args.lr)

    trainer = DPOTrainer(
        model=model,
        ref_model=None,    # None = use implicit reference (frozen copy of starting weights)
        args=dpo_cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    log.info("Starting DPO training…")
    trainer.train()
    log.info("DPO training complete.")
    return trainer


# ── Final eval on testing.json ────────────────────────────────────────────────

def evaluate_on_test_set(model, tokenizer, test_file: str):
    """
    Compute average cross-entropy loss on dataset/testing.json
    (same SFT format: existing_memory + latest_user_message → expected_output).
    """
    log.info("=" * 70)
    log.info("  Final evaluation on: %s", test_file)
    log.info("=" * 70)

    if not os.path.exists(test_file):
        log.warning("Test file not found: %s — skipping eval.", test_file)
        return

    with open(test_file) as f:
        records = json.load(f)

    log.info("Test records: %d", len(records))

    model.eval()
    device = next(model.parameters()).device
    total_loss = 0.0
    n = 0

    for i, rec in enumerate(records):
        example = {
            "existing_memory":    rec["existing_memory"],
            "latest_user_message": rec["latest_user_message"],
            "expected_output":    rec.get("expected_output", {}),
        }
        messages = build_messages(example)
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        # Build prompt-only text to find where assistant turn starts
        prompt_text = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )

        full_ids   = tokenizer(full_text,   return_tensors="pt")["input_ids"].to(device)
        prompt_len = tokenizer(prompt_text, return_tensors="pt")["input_ids"].shape[-1]

        labels = full_ids.clone()
        labels[:, :prompt_len] = -100   # mask prompt tokens

        with torch.no_grad():
            out = model(input_ids=full_ids, labels=labels)

        total_loss += out.loss.item()
        n += 1

        if (i + 1) % 50 == 0:
            log.info("  [%d/%d] running avg loss = %.4f", i + 1, len(records),
                     total_loss / n)

    avg_loss = total_loss / n if n else float("nan")
    log.info("─" * 50)
    log.info("  Test loss (avg CE) : %.4f", avg_loss)
    log.info("  Perplexity         : %.2f", torch.exp(torch.tensor(avg_loss)).item())
    log.info("─" * 50)
    return avg_loss


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    log.info("Config summary:")
    log.info("  SFT adapter  : %s", args.sft_adapter)
    log.info("  DPO dataset  : %s", args.dataset)
    log.info("  Output dir   : %s", args.output_dir)
    log.info("  DPO adapter  : %s", args.output_adapter)
    log.info("  Epochs       : %d", args.epochs)
    log.info("  Beta         : %.3f", args.beta)
    log.info("  LR           : %.2e", args.lr)
    log.info("  Batch size   : %d (grad_accum=%d)", args.batch_size, args.grad_accum)

    # 1. Load data
    records = load_dpo_records(args)
    train_ds, eval_ds = build_hf_dataset(records, args.eval_split)

    # 2. Load model
    model, tokenizer = load_model_for_dpo(args.sft_adapter)

    # 3. DPO train
    trainer = run_dpo_training(model, tokenizer, train_ds, eval_ds, args)

    # 4. Save adapter
    os.makedirs(args.output_adapter, exist_ok=True)
    trainer.model.save_pretrained(args.output_adapter)
    tokenizer.save_pretrained(args.output_adapter)
    log.info("DPO adapter saved to: %s", args.output_adapter)

    # 5. Final eval on testing.json
    evaluate_on_test_set(trainer.model, tokenizer, args.test_file)

    log.info("=" * 70)
    log.info("  All done!  Log saved to: %s", LOG_FILE)
    log.info("=" * 70)


if __name__ == "__main__":
    main()
