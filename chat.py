"""
chat.py — Lightweight client for the persistent serve.py server.

Model stays in RAM in serve.py. This script just sends your message.

Usage:
    python3 chat.py "My name is Saichand"
    python3 chat.py "I study at IITsricity college"
    python3 chat.py --show-memory
    python3 chat.py --clear-memory
"""

import argparse
import json
import sys
import urllib.request
import urllib.error

SERVER = "http://localhost:8765"


def chat(message: str):
    body = json.dumps({"message": message}).encode()
    req = urllib.request.Request(
        f"{SERVER}/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read())
    except urllib.error.URLError:
        print("❌ Server not running! Start it with:\n   python3 serve.py")
        sys.exit(1)

    print(f"\n💬 Message: {message}")
    print("─" * 50)

    actions = result.get("actions", [])
    if actions:
        print(f"⚙️  {len(actions)} action(s) applied:")
        for line in result.get("action_log", []):
            print(f"   {line}")
    else:
        print("ℹ️  No memory changes.")

    mem = result.get("memory", [])
    print(f"\n🧠 Memory ({len(mem)} entries):")
    if mem:
        for entry in mem:
            print(f"   [{entry['key']}] = {entry['value']}")
    else:
        print("   (empty)")
    print("─" * 50)


def show_memory():
    req = urllib.request.Request(f"{SERVER}/memory", method="GET")
    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read())
    except urllib.error.URLError:
        print("❌ Server not running! Start it with:\n   python3 serve.py")
        sys.exit(1)

    mem = result.get("memory", [])
    print(f"\n🧠 Current Memory ({len(mem)} entries):")
    print("─" * 50)
    if mem:
        for entry in mem:
            ts = entry.get("updated_at", "")[:19].replace("T", " ")
            print(f"  [{entry['key']}] = {entry['value']}  (updated: {ts})")
    else:
        print("  (empty — no memories stored yet)")
    print("─" * 50)


def clear_memory():
    import os
    mem_file = "memory.json"
    with open(mem_file, "w") as f:
        json.dump([], f)
    print("🗑️  memory.json cleared.")


def main():
    parser = argparse.ArgumentParser(description="Chat client for Memory Policy server")
    parser.add_argument("message", nargs="?", help="Message to send to the model")
    parser.add_argument("--show-memory", action="store_true", help="Show current memory store")
    parser.add_argument("--clear-memory", action="store_true", help="Clear all memories")
    parser.add_argument("--server", default=SERVER, help=f"Server URL (default: {SERVER})")
    args = parser.parse_args()

    if args.clear_memory:
        clear_memory()
    elif args.show_memory:
        show_memory()
    elif args.message:
        chat(args.message)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
