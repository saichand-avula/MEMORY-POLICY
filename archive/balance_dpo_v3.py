"""
balance_dpo_v3.py — Generate synthetic DPO records for underrepresented categories
and merge into a balanced dpo_v3_final.jsonl dataset.

Underrepresented categories to boost:
  1. update_split_into_delete_create: 1 → need 49 more
  2. under_trigger_other_singleton_keys: 16 → need 34 more  
  3. under_trigger_implicit_booking_no_anchor: 25 → need 25 more
"""

import json
import random

random.seed(42)

SYSTEM_PROMPT = """\
<|im_start|>system
You are a Memory Policy Model.

Your job is to decide what long-term memories should be created, updated, or deleted based on the user's latest message and the existing memory store.

You do NOT perform database operations and you do NOT generate UUIDs. You only predict memory actions.

──────────────────────────────────────────
CANONICAL KEYS
──────────────────────────────────────────
Only use keys from this list:
  - name
  - preferred_name
  - college
  - preferred_language
  - conversation_style
  - default_placement_template
  - tasks_events

Multiple memories can share the same key (e.g. multiple tasks_events entries).

──────────────────────────────────────────
OUTPUT FORMAT
──────────────────────────────────────────
Respond with a single JSON object — no markdown, no explanation.

For CREATE:
  {"action":"CREATE","key":"<key>","value":"<value>"}

For UPDATE (always include both the old and new value so the engine can locate the record):
  {"action":"UPDATE","key":"<key>","old_value":"<old>","new_value":"<new>"}

For DELETE:
  {"action":"DELETE","key":"<key>","value":"<value>"}

If no memory change is required:
  {"actions":[]}

Full response structure:
  {"actions":[...one or more action objects...]}

──────────────────────────────────────────
RULES
──────────────────────────────────────────
* Only act on information that is clearly personal, factual, or preference-based.
* Ignore greetings, filler phrases, and casual remarks unless they carry real information.
* Never invent information that was not stated.
* When a value changes, emit UPDATE with old_value and new_value — never DELETE+CREATE.
* When a fact is no longer relevant, emit DELETE with the exact value stored.
<|im_end|>"""

def make_prompt(memory, message):
    mem_str = json.dumps(memory)
    return (
        f"{SYSTEM_PROMPT}\n"
        f"<|im_start|>user\n"
        f"Existing memory:\n{mem_str}\n\n"
        f"User message: {json.dumps(message)}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ─── Category 1: update_split_into_delete_create ───────────────────────────
# Chosen: UPDATE old→new
# Rejected: DELETE old + CREATE new (wrong — should be UPDATE)

COLLEGES = [
    "IIT Bombay","IIT Delhi","IIT Madras","IIT Kanpur","IIT Kharagpur",
    "NIT Trichy","NIT Warangal","NIT Surathkal","BITS Pilani","BITS Goa",
    "VIT Vellore","VIT Chennai","IIIT Hyderabad","IIIT Bangalore","IIIT Lucknow",
    "Manipal Institute of Technology","SRM University","Amity University",
    "JNTU Kakinada","JNTU Hyderabad","Andhra University","Osmania University",
    "Bangalore University","Anna University","Pune University",
    "Jadavpur University","Calcutta University","Delhi University",
    "Aligarh Muslim University","Banaras Hindu University",
    "PSG College of Technology","Coimbatore Institute of Technology",
    "RV College of Engineering","PES University","Dayananda Sagar College",
]

NAMES = [
    "Aditya Sharma","Priya Nair","Ravi Kumar","Sneha Reddy","Arjun Mehta",
    "Kavya Singh","Rahul Gupta","Pooja Verma","Vikram Rao","Ananya Das",
    "Rohan Mishra","Divya Pillai","Kiran Joshi","Meera Patel","Suresh Babu",
    "Lakshmi Devi","Ganesh Murthy","Radha Krishna","Sunita Roy","Manoj Tiwari",
    "Fathima Begum","Mohammed Ali","Harpreet Kaur","Gurpreet Singh","Zara Khan",
    "Aryan Thakur","Ritu Saxena","Amit Srivastava","Nisha Chauhan","Deepak Yadav",
]

PREFERRED_NAMES = [
    "Adi","Pri","Ravi","Sneh","Arju","Kavy","Rahu","Pooj","Vik","Anan",
    "Roh","Div","Kir","Meer","Sur","Lak","Gan","Rad","Sun","Man",
    "Fath","Mo","Harp","Gur","Zar","Ary","Ritu","Amit","Nish","Deep",
    "Buddy","Chief","Boss","Champ","Ace","Star","Pro","Cap","Zen","Max",
]

LANGUAGES = [
    "Telugu","Tamil","Kannada","Malayalam","Hindi","Marathi","Gujarati",
    "Bengali","Odia","Punjabi","Assamese","Urdu","Sanskrit","English","French",
]

TEMPLATES = [
    "ATS-friendly resume template","Google-style resume template",
    "one-page compact template","two-page detailed template",
    "Harvard Business School template","Creative portfolio template",
    "Minimal clean template","Tech-focused resume template",
    "Academic CV template","Executive resume template",
]

CONVERSATION_STYLES = [
    "casual","formal","friendly","professional","concise","detailed",
    "encouraging","straightforward","humorous","empathetic",
]

def gen_update_split_into_delete_create(n=49):
    """UPDATE should be used, model wrongly does DELETE+CREATE."""
    records = []
    keys = ["college", "preferred_language", "preferred_name", "name",
            "conversation_style", "default_placement_template"]
    
    for i in range(n):
        key = keys[i % len(keys)]
        
        if key == "college":
            old_val = random.choice(COLLEGES)
            new_val = random.choice([c for c in COLLEGES if c != old_val])
            messages = [
                f"I moved to {new_val} instead of {old_val}",
                f"Actually I'm at {new_val} now, not {old_val}",
                f"Correction, my college is {new_val} not {old_val}",
                f"switched to {new_val}",
                f"I study at {new_val}",
            ]
        elif key == "preferred_language":
            old_val = random.choice(LANGUAGES)
            new_val = random.choice([l for l in LANGUAGES if l != old_val])
            messages = [
                f"Actually reply in {new_val}, not {old_val}",
                f"Switch to {new_val} please",
                f"Use {new_val} from now on",
                f"Prefer {new_val} going forward",
            ]
        elif key == "preferred_name":
            old_val = random.choice(PREFERRED_NAMES)
            new_val = random.choice([p for p in PREFERRED_NAMES if p != old_val])
            messages = [
                f"call me {new_val} not {old_val}",
                f"Actually {new_val} is better, drop {old_val}",
                f"Stop calling me {old_val}, I go by {new_val}",
                f"Correction — call me {new_val}",
            ]
        elif key == "name":
            old_val = random.choice(NAMES)
            new_val = random.choice([n for n in NAMES if n != old_val])
            messages = [
                f"It's {new_val}, not {old_val}",
                f"Correction, my name is {new_val} not {old_val}",
                f"I'm {new_val}, not {old_val}",
            ]
        elif key == "conversation_style":
            old_val = random.choice(CONVERSATION_STYLES)
            new_val = random.choice([c for c in CONVERSATION_STYLES if c != old_val])
            messages = [
                f"Drop the {old_val} tone, be more {new_val}",
                f"Switch from {old_val} to {new_val} style",
                f"Be {new_val} instead of {old_val}",
            ]
        else:  # default_placement_template
            old_val = random.choice(TEMPLATES)
            new_val = random.choice([t for t in TEMPLATES if t != old_val])
            messages = [
                f"Use {new_val} instead of {old_val}",
                f"Switch my default to {new_val}",
                f"Change my resume template from {old_val} to {new_val}",
            ]
        
        memory = [{"key": key, "value": old_val}]
        msg = random.choice(messages)
        
        chosen = json.dumps({"actions": [{"action": "UPDATE", "key": key, "old_value": old_val, "new_value": new_val}]})
        rejected = json.dumps({"actions": [{"action": "DELETE", "key": key, "value": old_val}, {"action": "CREATE", "key": key, "value": new_val}]})
        
        records.append({
            "prompt": make_prompt(memory, msg),
            "chosen": chosen,
            "rejected": rejected,
            "metadata": {
                "failure_category": "update_split_into_delete_create",
                "existing_memory": memory,
                "latest_user_message": msg,
                "synthetic": True,
            }
        })
    return records


# ─── Category 2: under_trigger_other_singleton_keys ────────────────────────
# Chosen: CREATE the right key
# Rejected: {} (model fails to trigger)

SINGLETON_SAMPLES = [
    # (key, value, messages)
    ("preferred_language", "Telugu", [
        "please reply in Telugu from now",
        "Talk to me in Telugu",
        "Use Telugu going forward",
        "I prefer Telugu",
        "Respond in Telugu please",
    ]),
    ("preferred_language", "Kannada", [
        "reply in Kannada",
        "Kannada please",
        "speak Kannada with me",
    ]),
    ("preferred_language", "Tamil", [
        "reply in Tamil from now",
        "use Tamil",
        "I'd like responses in Tamil",
    ]),
    ("preferred_language", "Hindi", [
        "speak Hindi with me",
        "use Hindi",
        "please reply in Hindi",
    ]),
    ("preferred_language", "English", [
        "reply in English from now on",
        "English only please",
        "use English going forward",
    ]),
    ("default_placement_template", "ATS-friendly resume template", [
        "use ATS-friendly template for my resume",
        "set ATS-friendly as my default resume format",
        "I want ATS-friendly template by default",
    ]),
    ("default_placement_template", "Google-style resume template", [
        "use Google-style resume as my default",
        "set Google-style template as default",
        "I prefer Google-style resume format",
    ]),
    ("default_placement_template", "one-page compact template", [
        "use one-page compact template by default",
        "set one-page compact as my resume default",
    ]),
    ("default_placement_template", "two-page detailed template", [
        "use two-page detailed template for my resumes",
        "set two-page detailed as default",
    ]),
    ("conversation_style", "encouraging", [
        "be more encouraging with me",
        "I like encouraging responses",
        "stay encouraging going forward",
    ]),
    ("conversation_style", "concise", [
        "keep it concise going forward",
        "be more concise please",
        "I prefer concise replies",
    ]),
    ("conversation_style", "casual", [
        "be more casual with me",
        "drop the formality, be casual",
        "keep it casual",
    ]),
    ("conversation_style", "professional", [
        "be more professional",
        "I prefer professional tone",
        "keep it professional",
    ]),
    ("preferred_name", "buddy", [
        "just call me buddy",
        "call me buddy",
    ]),
    ("preferred_name", "boss", [
        "call me boss",
        "just call me boss from now",
    ]),
]

def gen_under_trigger_other_singleton_keys(n=34):
    """Model should CREATE but outputs no action."""
    records = []
    for i in range(n):
        sample = SINGLETON_SAMPLES[i % len(SINGLETON_SAMPLES)]
        key, value, msgs = sample
        msg = random.choice(msgs)
        
        # Empty memory or unrelated memory
        mem_options = [
            [],
            [{"key": "name", "value": random.choice(NAMES)}],
            [{"key": "college", "value": random.choice(COLLEGES)}],
        ]
        memory = random.choice(mem_options)
        
        chosen = json.dumps({"actions": [{"action": "CREATE", "key": key, "value": value}]})
        rejected = json.dumps({"actions": []})
        
        records.append({
            "prompt": make_prompt(memory, msg),
            "chosen": chosen,
            "rejected": rejected,
            "metadata": {
                "failure_category": "under_trigger_other_singleton_keys",
                "existing_memory": memory,
                "latest_user_message": msg,
                "synthetic": True,
            }
        })
    return records


# ─── Category 3: under_trigger_implicit_booking_no_anchor ──────────────────
# Chosen: CREATE tasks_events with the booking
# Rejected: {} (model fails to infer/create)

COMPANIES = [
    "Amazon","Google","Microsoft","Infosys","TCS","Wipro","Accenture",
    "Deloitte","HCLTech","Cognizant","Capgemini","IBM","Oracle","SAP",
    "Salesforce","Adobe","Atlassian","Freshworks","Zoho","PhonePe",
    "Flipkart","Swiggy","Zomato","Ola","BYJU'S","Paytm","CRED","Meesho",
]

BOOKING_TEMPLATES = [
    ("book a cab for {company} {event_type}", "Cab booked for {company} {event_type}"),
    ("set a reminder for {company} {event_type}", "Reminder set for {company} {event_type}"),
    ("book a hotel for {company} {event_type}", "Hotel booked for {company} {event_type}"),
    ("book flight for {company} {event_type}", "Flight booked for {company} {event_type}"),
    ("set alarm for {company} {event_type}", "Reminder set for {company} {event_type}"),
    ("need cab for {company} {event_type}", "Cab booked for {company} {event_type}"),
    ("arrange hotel for {company} {event_type}", "Hotel booked for {company} {event_type}"),
]

EVENT_TYPES = [
    "recruitment drive","campus drive","interview","HR round","technical round",
    "coding test","group discussion","placement drive","final round","onsite round",
]

def gen_under_trigger_implicit_booking_no_anchor(n=25):
    """Model should CREATE booking task but outputs no action."""
    records = []
    for i in range(n):
        company = random.choice(COMPANIES)
        event_type = random.choice(EVENT_TYPES)
        tmpl_msg, tmpl_val = BOOKING_TEMPLATES[i % len(BOOKING_TEMPLATES)]
        
        msg = tmpl_msg.format(company=company, event_type=event_type)
        value = tmpl_val.format(company=company, event_type=event_type)
        
        # Unrelated memory (no anchor event in memory — just name/college)
        memory = [{"key": "name", "value": random.choice(NAMES)}]
        if random.random() > 0.5:
            memory.append({"key": "college", "value": random.choice(COLLEGES)})
        
        chosen = json.dumps({"actions": [{"action": "CREATE", "key": "tasks_events", "value": value}]})
        rejected = json.dumps({"actions": []})
        
        records.append({
            "prompt": make_prompt(memory, msg),
            "chosen": chosen,
            "rejected": rejected,
            "metadata": {
                "failure_category": "under_trigger_implicit_booking_no_anchor",
                "existing_memory": memory,
                "latest_user_message": msg,
                "synthetic": True,
            }
        })
    return records


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    # Load existing v3
    with open("dataset/dpo_v3_dataset (1).jsonl") as f:
        v3 = [json.loads(l) for l in f if l.strip()]
    
    print(f"Original v3: {len(v3)} records")
    
    # Generate missing records
    new1 = gen_update_split_into_delete_create(49)
    new2 = gen_under_trigger_other_singleton_keys(34)
    new3 = gen_under_trigger_implicit_booking_no_anchor(25)
    
    print(f"Generated update_split_into_delete_create: {len(new1)}")
    print(f"Generated under_trigger_other_singleton_keys: {len(new2)}")
    print(f"Generated under_trigger_implicit_booking_no_anchor: {len(new3)}")
    
    all_records = v3 + new1 + new2 + new3
    random.shuffle(all_records)
    
    print(f"\nFinal v3 balanced: {len(all_records)} records")
    
    # Final category check
    cats = {}
    for r in all_records:
        cat = r.get("metadata", {}).get("failure_category", "unknown")
        cats[cat] = cats.get(cat, 0) + 1
    
    print("\nFinal category distribution:")
    for cat, n in sorted(cats.items(), key=lambda x: -x[1]):
        bar = "█" * int(n / len(all_records) * 40)
        flag = " ⚠️" if n < 30 else ""
        print(f"  {cat:<52} {n:>4}{flag}")
    
    # Save
    out_path = "dataset/dpo_v3_final.jsonl"
    with open(out_path, "w") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")
    
    print(f"\n✅ Saved to {out_path}")


if __name__ == "__main__":
    main()
