"""
inference.py — Load the saved LoRA adapter and run a single prediction.

Usage:
    python inference.py
    python inference.py --adapter path/to/adapter --message "your message"
"""

import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

import config
from prompts import build_messages, SYSTEM_PROMPT


def load_model_and_tokenizer(adapter_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)

    base = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model, tokenizer


def predict(
    model,
    tokenizer,
    existing_memory: list[dict],
    latest_user_message: str,
    max_new_tokens: int = 256,
) -> dict:
    """Return the parsed JSON action dict for a single inference call."""
    example = {
        "existing_memory": existing_memory,
        "latest_user_message": latest_user_message,
        "expected_output": {},   # not used for inference
    }

    messages = build_messages(example)
    # For inference: drop the assistant turn and add generation prompt
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
            do_sample=False,        # greedy for deterministic JSON output
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    raw = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw_output": raw, "parse_error": True}


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Memory Policy LoRA inference")
    parser.add_argument(
        "--adapter", default=config.ADAPTER_SAVE_DIR,
        help="Path to saved LoRA adapter directory",
    )
    parser.add_argument(
        "--message", default="switch the nickname to Suja please",
        help="The user's latest message",
    )
    parser.add_argument(
        "--memory", default='[{"key":"preferred_name","value":"Adva"}]',
        help="Existing memory as a JSON string",
    )
    args = parser.parse_args()

    existing_memory = json.loads(args.memory)

    print(f"\nLoading adapter from: {args.adapter}")
    model, tokenizer = load_model_and_tokenizer(args.adapter)

    print(f"\nExisting memory : {existing_memory}")
    print(f"User message    : {args.message}")
    print("\nRunning inference…\n")

    result = predict(model, tokenizer, existing_memory, args.message)
    print("─" * 50)
    print(json.dumps(result, indent=2))
    print("─" * 50)


if __name__ == "__main__":
    main()
