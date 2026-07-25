"""
data.py — Dataset loading and tokenization.

Responsibilities:
  1. Load raw JSON files from disk.
  2. Apply the chat template to every example.
  3. Return HuggingFace Dataset objects ready for SFTTrainer.
"""

from datasets import load_dataset, DatasetDict
from transformers import PreTrainedTokenizer

from prompts import build_messages
import config


# ── 1. Raw loader ─────────────────────────────────────────────────────────────

def load_raw_dataset() -> DatasetDict:
    """Load train / test splits from the local JSON files."""
    return load_dataset(
        "json",
        data_files={
            "train": config.TRAIN_FILE,
            "test":  config.TEST_FILE,
        },
    )


# ── 2. Chat-template formatter ────────────────────────────────────────────────

def format_example(example: dict, tokenizer: PreTrainedTokenizer) -> dict:
    """
    Convert one dataset row into a single tokenized string using the
    tokenizer's built-in chat template.

    The full conversation (system + user + assistant) is returned as a
    single `text` field so that SFTTrainer can handle packing / truncation.
    """
    messages = build_messages(example)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,          # return a plain string; SFTTrainer tokenizes later
        add_generation_prompt=False,  # include the assistant turn we want to learn
    )
    return {"text": text}


# ── 3. Public entry point ─────────────────────────────────────────────────────

def load_and_format_dataset(tokenizer: PreTrainedTokenizer) -> DatasetDict:
    """
    Full pipeline: load JSON → apply chat template → return DatasetDict
    with `text` column on every split.
    """
    raw = load_raw_dataset()

    formatted = raw.map(
        lambda ex: format_example(ex, tokenizer),
        remove_columns=raw["train"].column_names,  # drop raw fields; keep only `text`
        desc="Applying chat template",
    )

    return formatted