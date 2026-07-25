"""
inference.py — Load the saved LoRA adapter and run a single prediction.

The memory store is persisted to memory.json automatically.
Each run reads the current memory, runs inference, applies the model's
CREATE / UPDATE / DELETE actions, and writes the updated memory back.

Usage:
    python inference.py
    python inference.py --adapter path/to/adapter --message "your message"
    python inference.py --message "My name is Sai" --memory-file memory.json
"""

import argparse
import json
import os
import uuid
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

import config
from prompts import build_messages, SYSTEM_PROMPT

DEFAULT_MEMORY_FILE = "memory.json"


def resolve_device() -> tuple[str, torch.dtype]:
    """Pick the best available device and matching dtype."""
    if torch.cuda.is_available():
        print("⚡ Using CUDA GPU")
        return "cuda", torch.bfloat16
    elif torch.backends.mps.is_available():
        print("🍎 Using Apple MPS (Metal GPU)")
        return "mps", torch.float16   # MPS works best with float16
    else:
        print("🐢 Using CPU (slow — consider running on the server)")
        return "cpu", torch.float32


def load_model_and_tokenizer(adapter_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)

    device, dtype = resolve_device()

    # Load base model on CPU first (avoids MPS + device_map segfault with PEFT)
    base = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    # Wrap with LoRA adapter (still on CPU)
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()

    # Now move the fully-wrapped model to MPS/CUDA
    if device != "cpu":
        model = model.to(device)

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
# Memory store helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_memory(path: str) -> list[dict]:
    """Load memory from JSON file, return empty list if file missing."""
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)


def save_memory(memory: list[dict], path: str) -> None:
    """Write memory list back to JSON file."""
    with open(path, "w") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)


def apply_actions(memory: list[dict], actions: list[dict]) -> list[dict]:
    """
    Apply CREATE / UPDATE / DELETE actions to the in-memory list.
    Each entry has: {id, key, value, updated_at}
    """
    for action in actions:
        act = action.get("action", "").upper()
        key = action.get("key", "")

        if act == "CREATE":
            memory.append({
                "id": str(uuid.uuid4()),
                "key": key,
                "value": action.get("value", ""),
                "updated_at": datetime.utcnow().isoformat(),
            })
            print(f"  ➕ CREATE  [{key}] = {action.get('value')}")

        elif act == "UPDATE":
            old_val = action.get("old_value", "")
            new_val = action.get("new_value", "")
            matched = False
            for entry in memory:
                if entry["key"] == key and entry["value"] == old_val:
                    entry["value"] = new_val
                    entry["updated_at"] = datetime.utcnow().isoformat()
                    matched = True
                    print(f"  ✏️  UPDATE  [{key}] {old_val!r} → {new_val!r}")
                    break
            if not matched:
                print(f"  ⚠️  UPDATE  [{key}] old_value {old_val!r} not found — skipped")

        elif act == "DELETE":
            val = action.get("value", "")
            before = len(memory)
            memory = [e for e in memory if not (e["key"] == key and e["value"] == val)]
            if len(memory) < before:
                print(f"  🗑️  DELETE  [{key}] = {val!r}")
            else:
                print(f"  ⚠️  DELETE  [{key}] value {val!r} not found — skipped")

        else:
            print(f"  ❓ Unknown action: {action}")

    return memory


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
        "--memory-file", default=DEFAULT_MEMORY_FILE,
        help=f"Path to memory JSON file (default: {DEFAULT_MEMORY_FILE})",
    )
    args = parser.parse_args()

    # Load persistent memory from file
    memory_file = args.memory_file
    existing_memory = load_memory(memory_file)

    print(f"\nLoading adapter from: {args.adapter}")
    model, tokenizer = load_model_and_tokenizer(args.adapter)

    print(f"\n📂 Memory file   : {memory_file}")
    print(f"🧠 Existing memory ({len(existing_memory)} entries):")
    for entry in existing_memory:
        print(f"   [{entry['key']}] = {entry['value']}")
    print(f"\n💬 User message  : {args.message}")
    print("\nRunning inference…\n")

    result = predict(model, tokenizer, existing_memory, args.message)

    print("─" * 50)
    print("Model output:")
    print(json.dumps(result, indent=2))
    print("─" * 50)

    # Apply actions to memory and save
    actions = result.get("actions", [])
    if actions:
        print(f"\n⚙️  Applying {len(actions)} action(s):")
        existing_memory = apply_actions(existing_memory, actions)
        save_memory(existing_memory, memory_file)
        print(f"\n✅ Memory updated and saved to: {memory_file}")
    else:
        print("\nℹ️  No memory changes — file unchanged.")

    print(f"\n🧠 Memory now ({len(existing_memory)} entries):")
    for entry in existing_memory:
        print(f"   [{entry['key']}] = {entry['value']}")


if __name__ == "__main__":
    main()
