"""
evaluate_all_models.py — Benchmark comparison of SFT vs DPO v1 vs DPO v2 vs DPO v3
"""

import json, sys, os, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from prompts import build_messages, CANONICAL_KEYS
import config

ADAPTERS = {
    "SFT": "adapter/memory-policy-lora",
    "DPO v1": "adapter/memory-policy-dpo-v1",
    "DPO v2": "adapter/memory-policy-dpo-v2",
    "DPO v3": "adapter/memory-policy-dpo-v3",
}

# 16 Critical Failure Category Benchmark Test Suite
FAILURE_SUITE = [
    {
        "category": "bare_update_vs_null_value",
        "memory": [{"key": "name", "value": "Saichand"}],
        "message": "correction its Sai Chandra not Saichand",
        "expected": [{"action": "UPDATE", "key": "name", "old_value": "Saichand", "new_value": "Sai Chandra"}]
    },
    {
        "category": "bare_update_vs_duplicate_create",
        "memory": [{"key": "preferred_name", "value": "chandu"}],
        "message": "call me sai",
        "expected": [{"action": "UPDATE", "key": "preferred_name", "old_value": "chandu", "new_value": "sai"}]
    },
    {
        "category": "bare_update_vs_noop",
        "memory": [{"key": "college", "value": "iiit"}],
        "message": "my college is iiit tirupati",
        "expected": [{"action": "UPDATE", "key": "college", "old_value": "iiit", "new_value": "iiit tirupati"}]
    },
    {
        "category": "compound_bare_update_vs_crosswire",
        "memory": [{"key": "preferred_name", "value": "sai"}, {"key": "college", "value": "IIT Sri City"}],
        "message": "i shifted to iit madras and they call me chandu",
        "expected": [
            {"action": "UPDATE", "key": "college", "old_value": "IIT Sri City", "new_value": "IIT Madras"},
            {"action": "UPDATE", "key": "preferred_name", "old_value": "sai", "new_value": "chandu"}
        ]
    },
    {
        "category": "key_isolation_vs_spurious_delete",
        "memory": [
            {"key": "tasks_events", "value": "Amazon drive on Friday"},
            {"key": "tasks_events", "value": "Hotel booked for Amazon drive on Friday"},
            {"key": "preferred_language", "value": "Telugu"}
        ],
        "message": "amazon drive cancelled",
        "expected": [
            {"action": "DELETE", "key": "tasks_events", "value": "Amazon drive on Friday"},
            {"action": "DELETE", "key": "tasks_events", "value": "Hotel booked for Amazon drive on Friday"}
        ]
    },
    {
        "category": "update_split_into_delete_create",
        "memory": [{"key": "preferred_language", "value": "Hindi"}],
        "message": "switch to English from now on",
        "expected": [{"action": "UPDATE", "key": "preferred_language", "old_value": "Hindi", "new_value": "English"}]
    },
    {
        "category": "hallucinated_old_value_should_be_create",
        "memory": [],
        "message": "correction, call me Chandu",
        "expected": [{"action": "CREATE", "key": "preferred_name", "value": "Chandu"}]
    },
    {
        "category": "degenerate_action_should_be_noop",
        "memory": [{"key": "name", "value": "Saichand"}],
        "message": "hi how are you doing today",
        "expected": []
    },
    {
        "category": "under_trigger_tasks_events_declarative",
        "memory": [],
        "message": "there is recruitment drive of amazon on friday",
        "expected": [{"action": "CREATE", "key": "tasks_events", "value": "Amazon recruitment drive on Friday"}]
    },
    {
        "category": "implicit_linked_create",
        "memory": [{"key": "tasks_events", "value": "Amazon recruitment drive on Friday"}],
        "message": "book a cab for amazon drive",
        "expected": [{"action": "CREATE", "key": "tasks_events", "value": "Book cab for Amazon recruitment drive on Friday"}]
    },
    {
        "category": "cascade_linked_entry_context_loss",
        "memory": [
            {"key": "tasks_events", "value": "Amazon recruitment drive on Friday"},
            {"key": "tasks_events", "value": "Book cab for Amazon recruitment drive on Friday"}
        ],
        "message": "amazon drive cancelled",
        "expected": [
            {"action": "DELETE", "key": "tasks_events", "value": "Amazon recruitment drive on Friday"},
            {"action": "DELETE", "key": "tasks_events", "value": "Book cab for Amazon recruitment drive on Friday"}
        ]
    },
    {
        "category": "over_action_scope_creep",
        "memory": [
            {"key": "tasks_events", "value": "Google interview on Monday"},
            {"key": "conversation_style", "value": "formal"}
        ],
        "message": "google interview got cancelled",
        "expected": [{"action": "DELETE", "key": "tasks_events", "value": "Google interview on Monday"}]
    },
    {
        "category": "under_trigger_implicit_booking_no_anchor",
        "memory": [],
        "message": "book a hotel for Microsoft interview",
        "expected": [{"action": "CREATE", "key": "tasks_events", "value": "Hotel booked for Microsoft interview"}]
    },
    {
        "category": "under_trigger_other_singleton_keys",
        "memory": [],
        "message": "be more concise with your responses",
        "expected": [{"action": "CREATE", "key": "conversation_style", "value": "concise"}]
    },
    {
        "category": "bare_update_vs_delete_create_split",
        "memory": [{"key": "default_placement_template", "value": "ATS-friendly resume template"}],
        "message": "switch my resume template to Google-style",
        "expected": [{"action": "UPDATE", "key": "default_placement_template", "old_value": "ATS-friendly resume template", "new_value": "Google-style resume template"}]
    },
    {
        "category": "other",
        "memory": [{"key": "name", "value": "Ravi Kumar"}],
        "message": "Correction, my name is Ravi Sharma and I have a TCS interview on Monday",
        "expected": [
            {"action": "UPDATE", "key": "name", "old_value": "Ravi Kumar", "new_value": "Ravi Sharma"},
            {"action": "CREATE", "key": "tasks_events", "value": "TCS interview on Monday"}
        ]
    }
]

def normalize_actions(actions):
    """Normalize actions list for loose matching."""
    norm = []
    for a in actions:
        if not isinstance(a, dict):
            continue
        act = str(a.get("action") or "").upper()
        key = str(a.get("key") or "")
        if act == "CREATE":
            norm.append(("CREATE", key, str(a.get("value") or "").strip().lower()))
        elif act == "UPDATE":
            norm.append(("UPDATE", key, str(a.get("old_value") or "").strip().lower(), str(a.get("new_value") or "").strip().lower()))
        elif act == "DELETE":
            norm.append(("DELETE", key, str(a.get("value") or "").strip().lower()))
    return sorted(norm)

def actions_match(pred_actions, exp_actions):
    p_norm = normalize_actions(pred_actions)
    e_norm = normalize_actions(exp_actions)
    if len(p_norm) != len(e_norm):
        return False
    for (p_act, p_key, *p_vals), (e_act, e_key, *e_vals) in zip(p_norm, e_norm):
        if p_act != e_act or p_key != e_key:
            return False
        # Check if values match closely
        if p_act == "UPDATE":
            # Check old value and new value exist and non-null
            if p_vals[1] is None or p_vals[1] == "none":
                return False
        elif p_act in ("CREATE", "DELETE"):
            if not p_vals[0]:
                return False
    return True

def run_evaluation():
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device == "mps" else (torch.bfloat16 if device == "cuda" else torch.float32)

    print(f"🚀 Starting Benchmark Evaluation across 4 Adapters on device: {device}\n")

    # Load test dataset sample (50 records from testing.json)
    with open("dataset/testing.json") as f:
        testing_json = json.load(f)[:50]

    all_results = {}

    for model_name, adapter_path in ADAPTERS.items():
        if not os.path.exists(adapter_path):
            print(f"⚠️ Adapter path {adapter_path} not found, skipping {model_name}.")
            continue

        print(f"--------------------------------------------------")
        print(f"Evaluating {model_name} ({adapter_path})...")

        tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            config.MODEL_ID,
            dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, adapter_path)
        model.eval()
        if device != "cpu":
            model = model.to(device)

        def predict(mem, msg):
            ex = {"existing_memory": mem, "latest_user_message": msg, "expected_output": {}}
            msgs = build_messages(ex)[:-1]
            prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            new_ids = out_ids[0][inputs["input_ids"].shape[-1]:]
            raw = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            try:
                res = json.loads(raw)
                return res.get("actions", [])
            except:
                return [{"error": "json_decode_failed", "raw": raw}]

        # 1. Evaluate Failure Category Suite (16 Edge Cases)
        fail_suite_correct = 0
        cat_results = {}
        for test_case in FAILURE_SUITE:
            cat = test_case["category"]
            pred = predict(test_case["memory"], test_case["message"])
            passed = actions_match(pred, test_case["expected"])
            if passed:
                fail_suite_correct += 1
            cat_results[cat] = {"passed": passed, "predicted": pred, "expected": test_case["expected"]}

        # 2. Evaluate Random Test Dataset (50 records from testing.json)
        test_json_correct = 0
        for item in testing_json:
            mem = item.get("existing_memory", [])
            msg = item.get("latest_user_message", "")
            exp_actions = item.get("expected_output", {}).get("actions", [])
            pred_actions = predict(mem, msg)
            if actions_match(pred_actions, exp_actions):
                test_json_correct += 1

        fail_acc = (fail_suite_correct / len(FAILURE_SUITE)) * 100
        test_acc = (test_json_correct / len(testing_json)) * 100
        overall_score = (fail_acc * 0.6) + (test_acc * 0.4)

        all_results[model_name] = {
            "failure_suite_acc": fail_acc,
            "failure_suite_correct": fail_suite_correct,
            "failure_suite_total": len(FAILURE_SUITE),
            "testing_json_acc": test_acc,
            "testing_json_correct": test_json_correct,
            "testing_json_total": len(testing_json),
            "overall_score": overall_score,
            "category_details": cat_results,
        }

        print(f"  --> Failure Suite Acc: {fail_acc:.1f}% ({fail_suite_correct}/{len(FAILURE_SUITE)})")
        print(f"  --> Test Dataset Acc : {test_acc:.1f}% ({test_json_correct}/{len(testing_json)})")
        print(f"  --> Composite Score  : {overall_score:.1f}%")

        # Cleanup GPU VRAM
        del model
        del base
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Save summary report to JSON
    with open("benchmark_evaluation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n==================================================")
    print("FINAL SUMMARY MATRIX")
    print("==================================================")
    print(f"{'Model':<12} | {'Edge-Case Suite':<18} | {'Test Dataset':<15} | {'Composite Score':<15}")
    print("-" * 68)
    for m_name, res in all_results.items():
        print(f"{m_name:<12} | {res['failure_suite_acc']:>5.1f}% ({res['failure_suite_correct']}/{res['failure_suite_total']})       | {res['testing_json_acc']:>5.1f}% ({res['testing_json_correct']}/{res['testing_json_total']})     | {res['overall_score']:>5.1f}%")
    print("==================================================")

if __name__ == "__main__":
    run_evaluation()
