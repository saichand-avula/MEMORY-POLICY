"""
train.py — Production fine-tuning entry point.

Steps:
  1. Load config
  2. Load tokenizer
  3. Load + format dataset  (chat template applied here)
  4. Load base model        (4-bit quantised for memory efficiency)
  5. Attach LoRA adapter
  6. Build SFTTrainer
  7. Train
  8. Save LoRA adapter
"""

import os
import logging

from transformers.trainer_utils import get_last_checkpoint

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

import config
from data import load_and_format_dataset

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 ─ Resolve device & precision
# ─────────────────────────────────────────────────────────────────────────────

def resolve_device_flags() -> tuple[bool, bool]:
    """
    Returns (use_bf16, use_fp16) based on what the hardware supports.
    Overrides config if the GPU doesn't support BF16.
    """
    if not torch.cuda.is_available():
        log.warning("No CUDA device found — training on CPU (very slow, debug only).")
        return False, False

    gpu = torch.cuda.get_device_name(0)
    bf16_ok = torch.cuda.is_bf16_supported()
    log.info("GPU detected: %s | BF16 supported: %s", gpu, bf16_ok)

    use_bf16 = config.BF16 and bf16_ok
    use_fp16 = config.FP16 and not use_bf16
    return use_bf16, use_fp16


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 ─ Load tokenizer
# ─────────────────────────────────────────────────────────────────────────────

def load_tokenizer() -> AutoTokenizer:
    log.info("Loading tokenizer from: %s", config.MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_ID,
        trust_remote_code=True,
    )
    # Qwen2 sets pad_token = eos_token by default; make explicit for clarity.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"   # required for causal LM training
    return tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 ─ Load base model (optionally 4-bit quantised)
# ─────────────────────────────────────────────────────────────────────────────

def load_base_model(use_bf16: bool, use_fp16: bool) -> AutoModelForCausalLM:
    """
    Load Qwen2.5-3B-Instruct.

    If bitsandbytes is available and a CUDA GPU is present, load in 4-bit
    (NF4) to keep GPU memory usage low. Otherwise load in full precision.
    """
    quantization_config = None
    load_in_4bit = False

    if torch.cuda.is_available():
        try:
            import bitsandbytes  # noqa: F401  (just check it's installed)
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            load_in_4bit = True
            log.info("4-bit (NF4) quantisation enabled via bitsandbytes.")
        except ImportError:
            log.warning(
                "bitsandbytes not installed — loading model in full precision. "
                "Install it with: pip install bitsandbytes"
            )

    log.info("Loading base model from: %s", config.MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if (use_bf16 and not load_in_4bit) else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.config.use_cache = False   # required for gradient checkpointing

    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 ─ Attach LoRA
# ─────────────────────────────────────────────────────────────────────────────

def attach_lora(model: AutoModelForCausalLM) -> AutoModelForCausalLM:
    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
        bias=config.LORA_BIAS,
        task_type=config.LORA_TASK_TYPE,
    )
    model = get_peft_model(model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    log.info(
        "LoRA attached — trainable params: %s / %s (%.2f%%)",
        f"{trainable:,}", f"{total:,}", 100 * trainable / total,
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 ─ Build TrainingArguments
# ─────────────────────────────────────────────────────────────────────────────

def build_training_args(use_bf16: bool, use_fp16: bool) -> TrainingArguments:
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    return TrainingArguments(
        output_dir=config.OUTPUT_DIR,

        # epochs / steps
        num_train_epochs=config.NUM_TRAIN_EPOCHS,

        # batching
        per_device_train_batch_size=config.PER_DEVICE_BATCH,
        per_device_eval_batch_size=config.PER_DEVICE_BATCH,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,

        # optimiser
        learning_rate=config.LEARNING_RATE,
        lr_scheduler_type=config.LR_SCHEDULER,
        warmup_ratio=config.WARMUP_RATIO,
        weight_decay=config.WEIGHT_DECAY,
        optim="paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",

        # precision
        bf16=use_bf16,
        fp16=use_fp16,

        # gradient checkpointing saves VRAM at the cost of ~20% slower forward pass
        gradient_checkpointing=True,

        # evaluation / saving
        eval_strategy="steps",
        eval_steps=config.EVAL_STEPS,
        save_strategy="steps",
        save_steps=config.SAVE_STEPS,
        save_total_limit=config.SAVE_TOTAL_LIMIT,
        load_best_model_at_end=config.LOAD_BEST_AT_END,
        metric_for_best_model=config.METRIC_FOR_BEST,

        # logging
        logging_steps=config.LOGGING_STEPS,
        report_to="none",          # set to "wandb" or "tensorboard" if desired

        # reproducibility
        seed=config.SEED,
        dataloader_num_workers=config.DATALOADER_NUM_WORKERS,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 ─ Build SFTTrainer
# ─────────────────────────────────────────────────────────────────────────────

def build_trainer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset,
    training_args: TrainingArguments,
) -> SFTTrainer:
    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        dataset_text_field="text",       # column produced by data.py
        max_seq_length=config.MAX_SEQ_LENGTH,
        packing=False,                   # packing can cause label-leakage; keep off
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("═" * 60)
    log.info("  Memory Policy LoRA — Fine-tuning Qwen2.5-3B-Instruct")
    log.info("═" * 60)

    # 1. Device flags
    use_bf16, use_fp16 = resolve_device_flags()

    # 2. Tokenizer
    tokenizer = load_tokenizer()

    # 3. Dataset (chat template applied inside load_and_format_dataset)
    log.info("Loading and formatting dataset…")
    dataset = load_and_format_dataset(tokenizer)
    log.info(
        "Dataset ready — train: %d examples, test: %d examples",
        len(dataset["train"]), len(dataset["test"]),
    )

    # 4. Base model
    model = load_base_model(use_bf16, use_fp16)

    # 5. LoRA
    model = attach_lora(model)

    # 6. Training arguments
    training_args = build_training_args(use_bf16, use_fp16)

    # 7. Trainer
    trainer = build_trainer(model, tokenizer, dataset, training_args)

    # 8. Train
    log.info("Starting training…")
    last_checkpoint = get_last_checkpoint(config.OUTPUT_DIR)

    trainer.train(resume_from_checkpoint=last_checkpoint)
    log.info("Training complete.")

    # 9. Save LoRA adapter (weights only — not the full merged model)
    os.makedirs(config.ADAPTER_SAVE_DIR, exist_ok=True)
    trainer.model.save_pretrained(config.ADAPTER_SAVE_DIR)
    tokenizer.save_pretrained(config.ADAPTER_SAVE_DIR)
    log.info("LoRA adapter saved to: %s", config.ADAPTER_SAVE_DIR)


if __name__ == "__main__":
    main()