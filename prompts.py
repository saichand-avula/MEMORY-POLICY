"""
prompts.py — Converts one raw dataset row into a list of chat messages.

The output of `build_messages()` is passed directly to
`tokenizer.apply_chat_template(...)`, so Qwen's native
<|im_start|> / <|im_end|> tokens are inserted automatically.
"""

import json


# ── Canonical keys (used in both system prompt and at inference time) ─────────

CANONICAL_KEYS: list[str] = [
    "name",
    "preferred_name",
    "college",
    "preferred_language",
    "conversation_style",
    "default_placement_template",
    "tasks_events",
]

_CANONICAL_KEYS_STR = "\n".join(f"  - {k}" for k in CANONICAL_KEYS)


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""\
You are a Memory Policy Model.

Your job is to decide what long-term memories should be created, updated, \
or deleted based on the user's latest message and the existing memory store.

You do NOT perform database operations and you do NOT generate UUIDs. \
You only predict memory actions.

──────────────────────────────────────────
CANONICAL KEYS
──────────────────────────────────────────
Only use keys from this list:
{_CANONICAL_KEYS_STR}

Multiple memories can share the same key (e.g. multiple tasks_events entries).

──────────────────────────────────────────
OUTPUT FORMAT
──────────────────────────────────────────
Respond with a single JSON object — no markdown, no explanation.

For CREATE:
  {{"action":"CREATE","key":"<key>","value":"<value>"}}

For UPDATE (always include both the old and new value so the engine can \
locate the record):
  {{"action":"UPDATE","key":"<key>","old_value":"<old>","new_value":"<new>"}}

For DELETE:
  {{"action":"DELETE","key":"<key>","value":"<value>"}}

If no memory change is required:
  {{"actions":[]}}

Full response structure:
  {{"actions":[...one or more action objects...]}}

──────────────────────────────────────────
RULES
──────────────────────────────────────────
* Only act on information that is clearly personal, factual, or preference-based.
* Ignore greetings, filler phrases, and casual remarks unless they carry real information.
* Never invent information that was not stated.
* When a value changes, emit UPDATE with old_value and new_value — never DELETE+CREATE.
* When a fact is no longer relevant, emit DELETE with the exact value stored.
"""


# ── Prompt builder ────────────────────────────────────────────────────────────

def _strip_ids(memory: list[dict]) -> list[dict]:
    """
    The memory store may include `id` fields used by the application layer.
    Strip them before showing to the model — the model reasons only over
    key and value, and never predicts UUIDs.
    """
    return [{"key": m["key"], "value": m["value"]} for m in memory]


def build_messages(example: dict) -> list[dict]:
    """
    Convert one dataset row into an OpenAI-style message list.

    Args:
        example: A dict with keys:
            - existing_memory      (list[dict])   — may include `id` fields
            - latest_user_message  (str)
            - expected_output      (dict)          — {"actions": [...]}

    Returns:
        [
            {"role": "system",    "content": <SYSTEM_PROMPT>},
            {"role": "user",      "content": <memory + message>},
            {"role": "assistant", "content": <JSON output>},
        ]
    """
    clean_memory = _strip_ids(example["existing_memory"])
    memory_str   = json.dumps(clean_memory, ensure_ascii=False)

    user_content = (
        f"Existing memory:\n{memory_str}\n\n"
        f"User message: \"{example['latest_user_message']}\""
    )

    assistant_content = json.dumps(
        example["expected_output"], ensure_ascii=False, separators=(",", ":")
    )

    return [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]
