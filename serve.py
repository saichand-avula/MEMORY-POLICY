"""
serve.py — Persistent model server.

Loads the LoRA model ONCE into RAM/MPS, then stays alive.
Send messages via the companion `chat.py` script or curl.

Usage:
    python3 serve.py                      # starts server on port 8765
    python3 serve.py --port 8765

Then in another terminal:
    python3 chat.py "My name is Saichand"
    python3 chat.py "I study at IITsricity college"
    python3 chat.py --show-memory          # view current memory
"""

import argparse
import json
import os
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
from prompts import build_messages

DEFAULT_MEMORY_FILE = "memory.json"
DEFAULT_PORT = 8765

# ── Global model (loaded once) ────────────────────────────────────────────────
_model = None
_tokenizer = None
_device = None


def resolve_device() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        print("⚡ Using CUDA GPU")
        return "cuda", torch.bfloat16
    elif torch.backends.mps.is_available():
        print("🍎 Using Apple MPS (Metal GPU)")
        return "mps", torch.float16
    else:
        print("🐢 Using CPU")
        return "cpu", torch.float32


def load_model(adapter_dir: str):
    global _model, _tokenizer, _device

    device, dtype = resolve_device()
    _device = device

    print(f"\n📦 Loading tokenizer from: {adapter_dir}")
    _tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)

    print(f"📦 Loading base model: {config.MODEL_ID}")
    base = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    print(f"🔗 Attaching LoRA adapter: {adapter_dir}")
    _model = PeftModel.from_pretrained(base, adapter_dir)
    _model.eval()

    if device != "cpu":
        print(f"🚀 Moving model to {device}...")
        _model = _model.to(device)

    print("\n✅ Model ready! Server is live.\n")


# ── Memory helpers ────────────────────────────────────────────────────────────

def load_memory(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def save_memory(memory: list[dict], path: str):
    with open(path, "w") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)


def apply_actions(memory: list[dict], actions: list[dict]) -> tuple[list[dict], list[str]]:
    log = []
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
            log.append(f"➕ CREATE  [{key}] = {action.get('value')}")

        elif act == "UPDATE":
            old_val, new_val = action.get("old_value", ""), action.get("new_value", "")
            matched = False
            for entry in memory:
                if entry["key"] == key and entry["value"] == old_val:
                    entry["value"] = new_val
                    entry["updated_at"] = datetime.utcnow().isoformat()
                    matched = True
                    log.append(f"✏️  UPDATE  [{key}] {old_val!r} → {new_val!r}")
                    break
            if not matched:
                log.append(f"⚠️  UPDATE  [{key}] old_value {old_val!r} not found — skipped")

        elif act == "DELETE":
            val = action.get("value", "")
            before = len(memory)
            memory = [e for e in memory if not (e["key"] == key and e["value"] == val)]
            if len(memory) < before:
                log.append(f"🗑️  DELETE  [{key}] = {val!r}")
            else:
                log.append(f"⚠️  DELETE  [{key}] value {val!r} not found — skipped")

    return memory, log


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(message: str, memory_file: str) -> dict:
    existing_memory = load_memory(memory_file)

    example = {
        "existing_memory": existing_memory,
        "latest_user_message": message,
        "expected_output": {},
    }

    messages = build_messages(example)[:-1]  # drop assistant turn
    prompt = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = _tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=_tokenizer.eos_token_id,
        )

    new_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    raw = _tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        return {"error": "JSON parse failed", "raw": raw, "memory": existing_memory}

    actions = result.get("actions", [])
    updated_memory, action_log = apply_actions(existing_memory, actions)
    save_memory(updated_memory, memory_file)

    return {
        "actions": actions,
        "action_log": action_log,
        "memory": updated_memory,
        "raw_model_output": raw,
    }


# ── HTTP Server ───────────────────────────────────────────────────────────────

memory_file_path = DEFAULT_MEMORY_FILE


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # silence default HTTP logs

    def do_GET(self):
        if self.path == "/memory":
            mem = load_memory(memory_file_path)
            self._respond(200, {"memory": mem, "count": len(mem)})
        elif self.path == "/health":
            self._respond(200, {"status": "ok", "model": config.MODEL_ID})
        else:
            self._respond(404, {"error": "Not found"})

    def do_POST(self):
        if self.path == "/chat":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            message = body.get("message", "")

            if not message:
                self._respond(400, {"error": "missing 'message' field"})
                return

            print(f"💬 [{datetime.now().strftime('%H:%M:%S')}] {message}")
            result = run_inference(message, memory_file_path)
            for line in result.get("action_log", []):
                print(f"   {line}")
            print(f"   🧠 Memory: {len(result['memory'])} entries")

            self._respond(200, result)
        else:
            self._respond(404, {"error": "Not found"})

    def _respond(self, code: int, data: dict):
        body = json.dumps(data, indent=2, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global memory_file_path

    parser = argparse.ArgumentParser(description="Memory Policy persistent server")
    parser.add_argument("--adapter", default=config.ADAPTER_SAVE_DIR)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--memory-file", default=DEFAULT_MEMORY_FILE)
    args = parser.parse_args()

    memory_file_path = args.memory_file

    print("═" * 55)
    print("  Memory Policy LoRA — Persistent Inference Server")
    print("═" * 55)

    load_model(args.adapter)

    print(f"🌐 Listening on http://localhost:{args.port}")
    print(f"📂 Memory file : {args.memory_file}")
    print("─" * 55)
    print("  In another terminal, run:")
    print(f'    python3 chat.py "your message here"')
    print(f'    python3 chat.py --show-memory')
    print("─" * 55)
    print("  Press Ctrl+C to stop the server and unload model.")
    print("─" * 55 + "\n")

    server = HTTPServer(("localhost", args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Server stopped. Model unloaded from RAM.")


if __name__ == "__main__":
    main()
