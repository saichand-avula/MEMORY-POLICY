"""
build_dpo_v3_balanced.py
Produces dpo_v3_final.jsonl with exactly TARGET=60 records per category.
  - Categories above 60: random sample down.
  - Categories below 60: generate synthetic records to fill.
"""

import json, random
random.seed(99)

TARGET = 60

# ── System prompt shared by all records ───────────────────────────────────────
SYS = """\
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

def prompt(memory, message):
    return (f"{SYS}\n<|im_start|>user\nExisting memory:\n{json.dumps(memory)}\n\n"
            f"User message: {json.dumps(message)}<|im_end|>\n<|im_start|>assistant\n")

def rec(cat, memory, msg, chosen_actions, rejected_actions, synthetic=True):
    return {
        "prompt": prompt(memory, msg),
        "chosen": json.dumps({"actions": chosen_actions}),
        "rejected": json.dumps({"actions": rejected_actions}),
        "metadata": {
            "failure_category": cat,
            "existing_memory": memory,
            "latest_user_message": msg,
            "synthetic": synthetic,
        }
    }

# ── Vocabulary ─────────────────────────────────────────────────────────────────
COLLEGES = [
    "IIT Bombay","IIT Delhi","IIT Madras","IIT Kanpur","IIT Roorkee","IIT Guwahati",
    "NIT Trichy","NIT Warangal","NIT Surathkal","BITS Pilani","BITS Goa",
    "VIT Vellore","VIT Chennai","IIIT Hyderabad","IIIT Bangalore","IIIT Lucknow",
    "Manipal Institute of Technology","SRM University","Amity University",
    "JNTU Kakinada","JNTU Hyderabad","Andhra University","Osmania University",
    "Bangalore University","Anna University","Pune University",
    "Jadavpur University","PSG College of Technology","RV College of Engineering",
    "PES University","Dayananda Sagar College","Christ University",
    "Symbiosis Institute of Technology","KIIT University","LPU",
]
NAMES = [
    "Aditya Sharma","Priya Nair","Ravi Kumar","Sneha Reddy","Arjun Mehta",
    "Kavya Singh","Rahul Gupta","Pooja Verma","Vikram Rao","Ananya Das",
    "Rohan Mishra","Divya Pillai","Kiran Joshi","Meera Patel","Suresh Babu",
    "Fathima Begum","Mohammed Ali","Harpreet Kaur","Zara Khan","Aryan Thakur",
    "Ritu Saxena","Amit Srivastava","Nisha Chauhan","Deepak Yadav","Sanjay Iyer",
    "Lakshmi Devi","Ganesh Murthy","Sunita Roy","Manoj Tiwari","Geeta Bhatt",
]
PNAMES = [
    "Adi","Pri","Ravi","Sneh","Arju","Kavy","Rahu","Vik","Anan","Roh",
    "Div","Kir","Meer","Sur","Lak","Gan","Sun","Man","Fath","Mo",
    "Harp","Zar","Ary","Ritu","Amit","Nish","Deep","Sanj","Geet","buddy",
    "boss","champ","ace","star","pro","cap","zen","max","rex","neo",
]
LANGS = ["Telugu","Tamil","Kannada","Malayalam","Hindi","Marathi","Gujarati",
         "Bengali","Odia","Punjabi","Assamese","Urdu","English","French","Japanese"]
STYLES = ["casual","formal","friendly","professional","concise","detailed",
          "encouraging","straightforward","humorous","empathetic","brief","witty"]
TEMPLATES = [
    "ATS-friendly resume template","Google-style resume template",
    "one-page compact template","two-page detailed template",
    "Harvard Business School template","Creative portfolio template",
    "Minimal clean template","Tech-focused resume template",
    "Academic CV template","Executive resume template",
]
COMPANIES = [
    "Amazon","Google","Microsoft","Infosys","TCS","Wipro","Accenture",
    "Deloitte","HCLTech","Cognizant","Capgemini","IBM","Oracle","SAP",
    "Salesforce","Adobe","Atlassian","Freshworks","Zoho","PhonePe",
    "Flipkart","Swiggy","Zomato","Ola","Paytm","CRED","Meesho","Razorpay",
]
EVENTS = ["recruitment drive","campus drive","interview","HR round",
          "technical round","coding test","group discussion","placement drive",
          "final round","onsite round","panel interview","managerial round"]
BOOKINGS = [("book a cab","Cab booked"),("set a reminder","Reminder set"),
            ("book a hotel","Hotel booked"),("book a flight","Flight booked"),
            ("set an alarm","Reminder set"),("arrange a cab","Cab arranged")]

def pair(lst, i): return lst[i % len(lst)]
def diff(lst, v): return next(x for x in lst if x != v)
def rnd(lst): return random.choice(lst)


# ── Generators for each category ───────────────────────────────────────────────

def gen_bare_update_vs_null_value(n):
    """Chosen: UPDATE old→new. Rejected: UPDATE old→null."""
    out = []
    keys = ["name","preferred_name","college","preferred_language",
            "conversation_style","default_placement_template"] * 10
    for i in range(n):
        k = keys[i]
        if k=="name":
            old,new = rnd(NAMES), rnd(NAMES)
            while old==new: new=rnd(NAMES)
            msgs = [f"It's {new}, not {old}", f"Correction, I'm {new} not {old}", f"My name is {new} actually"]
        elif k=="preferred_name":
            old,new = rnd(PNAMES), rnd(PNAMES)
            while old==new: new=rnd(PNAMES)
            msgs = [f"call me {new} not {old}", f"Actually {new}, drop {old}", f"Stop calling me {old}, call me {new}"]
        elif k=="college":
            old,new = rnd(COLLEGES), rnd(COLLEGES)
            while old==new: new=rnd(COLLEGES)
            msgs = [f"I moved to {new}", f"I'm at {new} now", f"switched to {new}"]
        elif k=="preferred_language":
            old,new = rnd(LANGS), rnd(LANGS)
            while old==new: new=rnd(LANGS)
            msgs = [f"reply in {new}", f"use {new} from now", f"switch to {new}"]
        elif k=="conversation_style":
            old,new = rnd(STYLES), rnd(STYLES)
            while old==new: new=rnd(STYLES)
            msgs = [f"be more {new}", f"drop the {old} tone, be {new}", f"switch to {new} style"]
        else:
            old,new = rnd(TEMPLATES), rnd(TEMPLATES)
            while old==new: new=rnd(TEMPLATES)
            msgs = [f"use {new} instead of {old}", f"switch to {new}"]
        msg = rnd(msgs)
        mem = [{"key":k,"value":old}]
        out.append(rec("bare_update_vs_null_value", mem, msg,
            [{"action":"UPDATE","key":k,"old_value":old,"new_value":new}],
            [{"action":"UPDATE","key":k,"old_value":old,"new_value":None}]))
    return out

def gen_bare_update_vs_duplicate_create(n):
    """Chosen: UPDATE. Rejected: CREATE duplicate."""
    out = []
    keys = ["name","preferred_name","college","preferred_language",
            "conversation_style","default_placement_template"] * 10
    for i in range(n):
        k = keys[i]
        if k=="name":
            old,new = rnd(NAMES), rnd(NAMES)
            while old==new: new=rnd(NAMES)
            msgs = [f"I'm {new}", f"Actually I'm {new}", f"My name is {new}"]
        elif k=="preferred_name":
            old,new = rnd(PNAMES), rnd(PNAMES)
            while old==new: new=rnd(PNAMES)
            msgs = [f"call me {new}", f"go by {new} please", f"I prefer {new}"]
        elif k=="college":
            old,new = rnd(COLLEGES), rnd(COLLEGES)
            while old==new: new=rnd(COLLEGES)
            msgs = [f"I study at {new}", f"my college is {new}", f"I'm at {new} now"]
        elif k=="preferred_language":
            old,new = rnd(LANGS), rnd(LANGS)
            while old==new: new=rnd(LANGS)
            msgs = [f"{new} please", f"reply in {new}", f"use {new}"]
        elif k=="conversation_style":
            old,new = rnd(STYLES), rnd(STYLES)
            while old==new: new=rnd(STYLES)
            msgs = [f"be {new}", f"keep it {new}", f"I prefer {new} responses"]
        else:
            old,new = rnd(TEMPLATES), rnd(TEMPLATES)
            while old==new: new=rnd(TEMPLATES)
            msgs = [f"use {new}", f"switch to {new} for my resume"]
        msg = rnd(msgs)
        mem = [{"key":k,"value":old}]
        out.append(rec("bare_update_vs_duplicate_create", mem, msg,
            [{"action":"UPDATE","key":k,"old_value":old,"new_value":new}],
            [{"action":"CREATE","key":k,"value":new}]))
    return out

def gen_bare_update_vs_noop(n):
    """Chosen: UPDATE. Rejected: empty actions."""
    out = []
    keys = ["name","preferred_name","college","preferred_language",
            "conversation_style","default_placement_template"] * 10
    for i in range(n):
        k = keys[i]
        if k=="name":
            old,new = rnd(NAMES), rnd(NAMES)
            while old==new: new=rnd(NAMES)
            msgs = [f"Actually it's {new}", f"Correction: {new} not {old}"]
        elif k=="preferred_name":
            old,new = rnd(PNAMES), rnd(PNAMES)
            while old==new: new=rnd(PNAMES)
            msgs = [f"I go by {new} now", f"call me {new}"]
        elif k=="college":
            old,new = rnd(COLLEGES), rnd(COLLEGES)
            while old==new: new=rnd(COLLEGES)
            msgs = [f"I enrolled at {new}", f"my uni is {new}"]
        elif k=="preferred_language":
            old,new = rnd(LANGS), rnd(LANGS)
            while old==new: new=rnd(LANGS)
            msgs = [f"prefer {new} going forward", f"talk to me in {new}"]
        elif k=="conversation_style":
            old,new = rnd(STYLES), rnd(STYLES)
            while old==new: new=rnd(STYLES)
            msgs = [f"I'd prefer {new} responses", f"be more {new} please"]
        else:
            old,new = rnd(TEMPLATES), rnd(TEMPLATES)
            while old==new: new=rnd(TEMPLATES)
            msgs = [f"I want {new} by default", f"set {new} as default"]
        msg = rnd(msgs)
        mem = [{"key":k,"value":old}]
        out.append(rec("bare_update_vs_noop", mem, msg,
            [{"action":"UPDATE","key":k,"old_value":old,"new_value":new}],
            []))
    return out

def gen_bare_update_vs_delete_create_split(n):
    """Chosen: UPDATE. Rejected: DELETE + CREATE."""
    out = []
    keys = ["name","preferred_name","college","preferred_language",
            "conversation_style","default_placement_template"] * 10
    for i in range(n):
        k = keys[i]
        if k=="college":
            old,new = rnd(COLLEGES), rnd(COLLEGES)
            while old==new: new=rnd(COLLEGES)
            msgs = [f"studying at {new}", f"I moved to {new}", f"now at {new}"]
        elif k=="preferred_language":
            old,new = rnd(LANGS), rnd(LANGS)
            while old==new: new=rnd(LANGS)
            msgs = [f"switch to {new}", f"use {new} now", f"{new} from now on"]
        elif k=="preferred_name":
            old,new = rnd(PNAMES), rnd(PNAMES)
            while old==new: new=rnd(PNAMES)
            msgs = [f"call me {new}", f"I go by {new}", f"prefer {new}"]
        elif k=="name":
            old,new = rnd(NAMES), rnd(NAMES)
            while old==new: new=rnd(NAMES)
            msgs = [f"my name is {new}", f"I'm {new}"]
        elif k=="conversation_style":
            old,new = rnd(STYLES), rnd(STYLES)
            while old==new: new=rnd(STYLES)
            msgs = [f"be {new} with me", f"I prefer {new} style"]
        else:
            old,new = rnd(TEMPLATES), rnd(TEMPLATES)
            while old==new: new=rnd(TEMPLATES)
            msgs = [f"use {new} for resumes", f"default to {new}"]
        msg = rnd(msgs)
        mem = [{"key":k,"value":old}]
        out.append(rec("bare_update_vs_delete_create_split", mem, msg,
            [{"action":"UPDATE","key":k,"old_value":old,"new_value":new}],
            [{"action":"DELETE","key":k,"value":old},{"action":"CREATE","key":k,"value":new}]))
    return out

def gen_compound_bare_update_vs_crosswire(n):
    """Chosen: two UPDATEs. Rejected: only one UPDATE (misses second)."""
    out = []
    combos = [("college","preferred_name"),("college","preferred_language"),
              ("name","college"),("preferred_name","preferred_language"),
              ("name","preferred_name"),("college","conversation_style")] * 10
    for i in range(n):
        k1,k2 = combos[i]
        def val(k):
            if k=="college": return rnd(COLLEGES)
            if k=="name": return rnd(NAMES)
            if k=="preferred_name": return rnd(PNAMES)
            if k=="preferred_language": return rnd(LANGS)
            if k=="conversation_style": return rnd(STYLES)
            return rnd(TEMPLATES)
        o1,n1 = val(k1), val(k1)
        while o1==n1: n1=val(k1)
        o2,n2 = val(k2), val(k2)
        while o2==n2: n2=val(k2)
        
        # Build message combining both
        def phrase(k,new_val):
            if k=="college": return f"I'm at {new_val} now"
            if k=="name": return f"my name is {new_val}"
            if k=="preferred_name": return f"call me {new_val}"
            if k=="preferred_language": return f"reply in {new_val}"
            if k=="conversation_style": return f"be {new_val}"
            return f"use {new_val} template"
        msg = phrase(k1,n1) + ". " + phrase(k2,n2)
        mem = [{"key":k1,"value":o1},{"key":k2,"value":o2}]
        out.append(rec("compound_bare_update_vs_crosswire", mem, msg,
            [{"action":"UPDATE","key":k1,"old_value":o1,"new_value":n1},
             {"action":"UPDATE","key":k2,"old_value":o2,"new_value":n2}],
            [{"action":"UPDATE","key":k1,"old_value":o1,"new_value":n1}]))
    return out

def gen_update_split_into_delete_create(n):
    """Chosen: UPDATE. Rejected: DELETE+CREATE for a single-key change."""
    out = []
    keys = ["college","preferred_language","preferred_name","name",
            "conversation_style","default_placement_template"] * 10
    for i in range(n):
        k = keys[i]
        if k=="college":
            old,new = rnd(COLLEGES), rnd(COLLEGES)
            while old==new: new=rnd(COLLEGES)
            msgs = [f"I moved to {new} instead of {old}",f"Actually I'm at {new} now, not {old}"]
        elif k=="preferred_language":
            old,new = rnd(LANGS), rnd(LANGS)
            while old==new: new=rnd(LANGS)
            msgs = [f"Actually reply in {new}, not {old}", f"switch from {old} to {new}"]
        elif k=="preferred_name":
            old,new = rnd(PNAMES), rnd(PNAMES)
            while old==new: new=rnd(PNAMES)
            msgs = [f"call me {new} not {old}", f"Actually {new} is better, drop {old}"]
        elif k=="name":
            old,new = rnd(NAMES), rnd(NAMES)
            while old==new: new=rnd(NAMES)
            msgs = [f"It's {new}, not {old}", f"Correction, my name is {new} not {old}"]
        elif k=="conversation_style":
            old,new = rnd(STYLES), rnd(STYLES)
            while old==new: new=rnd(STYLES)
            msgs = [f"Drop the {old} tone, be more {new}", f"switch from {old} to {new}"]
        else:
            old,new = rnd(TEMPLATES), rnd(TEMPLATES)
            while old==new: new=rnd(TEMPLATES)
            msgs = [f"Use {new} instead of {old}", f"Switch my default to {new}"]
        msg = rnd(msgs)
        mem = [{"key":k,"value":old}]
        out.append(rec("update_split_into_delete_create", mem, msg,
            [{"action":"UPDATE","key":k,"old_value":old,"new_value":new}],
            [{"action":"DELETE","key":k,"value":old},{"action":"CREATE","key":k,"value":new}]))
    return out

def gen_hallucinated_old_value_should_be_create(n):
    """Chosen: CREATE (empty memory). Rejected: UPDATE with hallucinated old_value."""
    out = []
    keys = ["name","preferred_name","college","preferred_language",
            "conversation_style","default_placement_template"] * 10
    for i in range(n):
        k = keys[i]
        if k=="name":
            v = rnd(NAMES); fake = rnd(NAMES)
            while fake==v: fake=rnd(NAMES)
            msgs = [f"I'm {v}", f"My name is {v}", f"Correction, I'm {v}"]
        elif k=="preferred_name":
            v = rnd(PNAMES); fake = rnd(PNAMES)
            while fake==v: fake=rnd(PNAMES)
            msgs = [f"call me {v}", f"Correction, call me {v}"]
        elif k=="college":
            v = rnd(COLLEGES); fake = rnd(COLLEGES)
            while fake==v: fake=rnd(COLLEGES)
            msgs = [f"I study at {v}", f"my college is {v}"]
        elif k=="preferred_language":
            v = rnd(LANGS); fake = rnd(LANGS)
            while fake==v: fake=rnd(LANGS)
            msgs = [f"reply in {v}", f"use {v} from now"]
        elif k=="conversation_style":
            v = rnd(STYLES); fake = rnd(STYLES)
            while fake==v: fake=rnd(STYLES)
            msgs = [f"be {v}", f"keep it {v}"]
        else:
            v = rnd(TEMPLATES); fake = rnd(TEMPLATES)
            while fake==v: fake=rnd(TEMPLATES)
            msgs = [f"use {v} by default"]
        msg = rnd(msgs)
        out.append(rec("hallucinated_old_value_should_be_create", [], msg,
            [{"action":"CREATE","key":k,"value":v}],
            [{"action":"UPDATE","key":k,"old_value":fake,"new_value":v}]))
    return out

def gen_key_isolation_vs_spurious_delete(n):
    """Chosen: one UPDATE. Rejected: UPDATE + spurious DELETE of unrelated key."""
    out = []
    singleton_keys = ["name","preferred_name","college","preferred_language",
                      "conversation_style","default_placement_template"]
    for i in range(n):
        k = pair(singleton_keys, i)
        if k=="college":
            old,new = rnd(COLLEGES), rnd(COLLEGES)
            while old==new: new=rnd(COLLEGES)
            msgs = [f"I go to {new}", f"my college is {new}"]
        elif k=="preferred_name":
            old,new = rnd(PNAMES), rnd(PNAMES)
            while old==new: new=rnd(PNAMES)
            msgs = [f"call me {new}", f"I go by {new}"]
        elif k=="preferred_language":
            old,new = rnd(LANGS), rnd(LANGS)
            while old==new: new=rnd(LANGS)
            msgs = [f"reply in {new}", f"use {new}"]
        elif k=="name":
            old,new = rnd(NAMES), rnd(NAMES)
            while old==new: new=rnd(NAMES)
            msgs = [f"I'm {new}", f"my name is {new}"]
        elif k=="conversation_style":
            old,new = rnd(STYLES), rnd(STYLES)
            while old==new: new=rnd(STYLES)
            msgs = [f"be {new}", f"I prefer {new} style"]
        else:
            old,new = rnd(TEMPLATES), rnd(TEMPLATES)
            while old==new: new=rnd(TEMPLATES)
            msgs = [f"use {new} template"]
        msg = rnd(msgs)
        unrelated_k = pair([x for x in singleton_keys if x!=k], i)
        unrelated_v = rnd(LANGS) if unrelated_k=="preferred_language" else \
                      rnd(STYLES) if unrelated_k=="conversation_style" else \
                      rnd(NAMES) if unrelated_k=="name" else \
                      rnd(PNAMES) if unrelated_k=="preferred_name" else \
                      rnd(TEMPLATES) if unrelated_k=="default_placement_template" else rnd(COLLEGES)
        mem = [{"key":k,"value":old},{"key":unrelated_k,"value":unrelated_v}]
        out.append(rec("key_isolation_vs_spurious_delete", mem, msg,
            [{"action":"UPDATE","key":k,"old_value":old,"new_value":new}],
            [{"action":"UPDATE","key":k,"old_value":old,"new_value":new},
             {"action":"DELETE","key":unrelated_k,"value":unrelated_v}]))
    return out

def gen_degenerate_action_should_be_noop(n):
    """Chosen: empty actions. Rejected: spurious CREATE/UPDATE."""
    greetings = ["hi","hey there","hello","thanks","ok","sure","got it",
                 "bye","see you","thanks a lot","alright","noted","ok thanks",
                 "sounds good","perfect","great","cool","nice","awesome","ok bye",
                 "makes sense","understood","will do","ok got it","yep ok"]
    keys = ["name","preferred_name","college","preferred_language"]
    for_keys = lambda k: rnd(NAMES) if k=="name" else rnd(PNAMES) if k=="preferred_name" \
                else rnd(COLLEGES) if k=="college" else rnd(LANGS)
    out = []
    mems = [[],
            [{"key":"name","value":rnd(NAMES)}],
            [{"key":"college","value":rnd(COLLEGES)}],
            [{"key":"tasks_events","value":f"{rnd(COMPANIES)} interview"}]]
    for i in range(n):
        msg = rnd(greetings)
        mem = mems[i%len(mems)]
        k = pair(keys, i)
        out.append(rec("degenerate_action_should_be_noop", mem, msg,
            [],
            [{"action":"CREATE","key":k,"value":for_keys(k)}]))
    return out

def gen_under_trigger_tasks_events_declarative(n):
    """Chosen: CREATE tasks_events. Rejected: empty."""
    out = []
    for i in range(n):
        co = pair(COMPANIES, i); ev = pair(EVENTS, i)
        msgs = [
            f"there is a {ev} of {co}",
            f"{co} {ev} is coming up",
            f"I have a {ev} at {co}",
            f"{co} has a {ev}",
            f"got a {ev} scheduled at {co}",
        ]
        msg = rnd(msgs)
        val = f"{co} {ev}"
        mem = rnd([[],[{"key":"name","value":rnd(NAMES)}]])
        out.append(rec("under_trigger_tasks_events_declarative", mem, msg,
            [{"action":"CREATE","key":"tasks_events","value":val}],
            []))
    return out

def gen_implicit_linked_create(n):
    """Chosen: CREATE linked task (booking). Rejected: empty."""
    out = []
    for i in range(n):
        co = pair(COMPANIES, i); ev = pair(EVENTS, i)
        book_msg, book_val = pair(BOOKINGS, i)
        anchor = f"{co} {ev}"
        msg = f"{book_msg} for {co} {ev}"
        val = f"{book_val} for {co} {ev}"
        mem = [{"key":"tasks_events","value":anchor}]
        out.append(rec("implicit_linked_create", mem, msg,
            [{"action":"CREATE","key":"tasks_events","value":val}],
            []))
    return out

def gen_cascade_linked_entry_context_loss(n):
    """Chosen: DELETE all linked entries. Rejected: DELETE only first."""
    out = []
    for i in range(n):
        co = pair(COMPANIES, i); ev = pair(EVENTS, i)
        anchor = f"{co} {ev}"
        book_msg, book_val = pair(BOOKINGS, i)
        book_entry = f"{book_val} for {co} {ev}"
        msgs = [f"{co} {ev} got cancelled, cancel everything related",
                f"{co} {ev} is cancelled, remove all",
                f"cancel {co} {ev} and anything related"]
        msg = rnd(msgs)
        mem = [{"key":"tasks_events","value":anchor},
               {"key":"tasks_events","value":book_entry}]
        out.append(rec("cascade_linked_entry_context_loss", mem, msg,
            [{"action":"DELETE","key":"tasks_events","value":anchor},
             {"action":"DELETE","key":"tasks_events","value":book_entry}],
            [{"action":"DELETE","key":"tasks_events","value":anchor}]))
    return out

def gen_over_action_scope_creep(n):
    """Chosen: DELETE only relevant. Rejected: also deletes unrelated."""
    out = []
    for i in range(n):
        co = pair(COMPANIES, i); ev = pair(EVENTS, i)
        anchor = f"{co} {ev}"
        unrelated_k = pair(["preferred_language","conversation_style","preferred_name"], i)
        unrelated_v = rnd(LANGS) if unrelated_k=="preferred_language" else \
                      rnd(STYLES) if unrelated_k=="conversation_style" else rnd(PNAMES)
        msgs = [f"{co} {ev} got cancelled",f"{co} {ev} is off",f"cancel {co} {ev}"]
        msg = rnd(msgs)
        mem = [{"key":"tasks_events","value":anchor},
               {"key":unrelated_k,"value":unrelated_v}]
        out.append(rec("over_action_scope_creep", mem, msg,
            [{"action":"DELETE","key":"tasks_events","value":anchor}],
            [{"action":"DELETE","key":"tasks_events","value":anchor},
             {"action":"DELETE","key":unrelated_k,"value":unrelated_v}]))
    return out

def gen_under_trigger_implicit_booking_no_anchor(n):
    """Chosen: CREATE booking task. Rejected: empty (no anchor in memory)."""
    out = []
    for i in range(n):
        co = pair(COMPANIES, i); ev = pair(EVENTS, i)
        book_msg, book_val = pair(BOOKINGS, i)
        msg = f"{book_msg} for {co} {ev}"
        val = f"{book_val} for {co} {ev}"
        mem = rnd([[{"key":"name","value":rnd(NAMES)}],[]])
        out.append(rec("under_trigger_implicit_booking_no_anchor", mem, msg,
            [{"action":"CREATE","key":"tasks_events","value":val}],
            []))
    return out

def gen_under_trigger_other_singleton_keys(n):
    """Chosen: CREATE singleton key. Rejected: empty."""
    items = [
        ("preferred_language","Telugu",["reply in Telugu","use Telugu from now","talk to me in Telugu"]),
        ("preferred_language","Tamil",["reply in Tamil","use Tamil","respond in Tamil"]),
        ("preferred_language","Kannada",["reply in Kannada","Kannada please"]),
        ("preferred_language","Hindi",["speak Hindi with me","use Hindi","reply in Hindi"]),
        ("preferred_language","English",["reply in English","English only please"]),
        ("preferred_language","Marathi",["reply in Marathi","use Marathi"]),
        ("preferred_language","Bengali",["reply in Bengali","Bengali please"]),
        ("default_placement_template","ATS-friendly resume template",
         ["use ATS-friendly template","set ATS-friendly as default"]),
        ("default_placement_template","Google-style resume template",
         ["use Google-style resume","Google-style template please"]),
        ("default_placement_template","one-page compact template",
         ["use one-page compact template","one-page compact by default"]),
        ("default_placement_template","two-page detailed template",
         ["use two-page detailed template","set two-page detailed as default"]),
        ("conversation_style","encouraging",
         ["be more encouraging","I like encouraging responses"]),
        ("conversation_style","concise",
         ["keep it concise","be more concise please","I prefer concise"]),
        ("conversation_style","casual",
         ["be more casual","drop the formality","keep it casual"]),
        ("conversation_style","professional",
         ["be professional","I prefer professional tone"]),
        ("preferred_name","buddy",["call me buddy","just call me buddy"]),
        ("preferred_name","boss",["call me boss"]),
        ("preferred_name","champ",["call me champ"]),
    ]
    out = []
    mems = [[],[{"key":"name","value":rnd(NAMES)}],[{"key":"college","value":rnd(COLLEGES)}]]
    for i in range(n):
        k,v,msgs = items[i%len(items)]
        msg = rnd(msgs)
        mem = mems[i%len(mems)]
        out.append(rec("under_trigger_other_singleton_keys", mem, msg,
            [{"action":"CREATE","key":k,"value":v}],
            []))
    return out

def gen_other(n):
    """Mixed compound scenarios: CREATE + DELETE different keys."""
    out = []
    singleton_keys = ["preferred_language","conversation_style","preferred_name","college"]
    for i in range(n):
        # Pattern: user changes name AND adds a task
        co = pair(COMPANIES, i); ev = pair(EVENTS, i)
        new_name = rnd(NAMES); old_name = rnd(NAMES)
        while old_name==new_name: old_name=rnd(NAMES)
        
        if i%3==0:
            # UPDATE name + CREATE task
            msg = f"Correction, its {new_name} not {old_name}. Also {co} {ev} is scheduled"
            mem = [{"key":"name","value":old_name}]
            chosen = [{"action":"UPDATE","key":"name","old_value":old_name,"new_value":new_name},
                      {"action":"CREATE","key":"tasks_events","value":f"{co} {ev}"}]
            rejected = [{"action":"UPDATE","key":"name","old_value":old_name,"new_value":new_name}]
        elif i%3==1:
            # DELETE task + UPDATE preference
            k = pair(singleton_keys, i)
            old_v = rnd(LANGS) if k=="preferred_language" else rnd(STYLES) if k=="conversation_style" \
                    else rnd(PNAMES) if k=="preferred_name" else rnd(COLLEGES)
            new_v = rnd(LANGS) if k=="preferred_language" else rnd(STYLES) if k=="conversation_style" \
                    else rnd(PNAMES) if k=="preferred_name" else rnd(COLLEGES)
            while old_v==new_v: new_v = rnd(LANGS)
            anchor = f"{co} {ev}"
            msg = f"{co} {ev} cancelled. Also reply in {new_v} from now" if k=="preferred_language" else \
                  f"{co} {ev} cancelled. Call me {new_v}"
            mem = [{"key":"tasks_events","value":anchor},{"key":k,"value":old_v}]
            chosen = [{"action":"DELETE","key":"tasks_events","value":anchor},
                      {"action":"UPDATE","key":k,"old_value":old_v,"new_value":new_v}]
            rejected = [{"action":"DELETE","key":"tasks_events","value":anchor}]
        else:
            # CREATE task + DELETE old preference
            k = "conversation_style"
            old_style = rnd(STYLES)
            msg = f"{co} {ev} tomorrow. Also drop the {old_style} tone"
            mem = [{"key":k,"value":old_style}]
            chosen = [{"action":"CREATE","key":"tasks_events","value":f"{co} {ev}"},
                      {"action":"DELETE","key":k,"value":old_style}]
            rejected = [{"action":"CREATE","key":"tasks_events","value":f"{co} {ev}"}]
        out.append(rec("other", mem, msg, chosen, rejected))
    return out


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Load existing v3_final
    with open("dataset/dpo_v3_final.jsonl") as f:
        v3 = [json.loads(l) for l in f if l.strip()]

    # Group by category
    by_cat = {}
    for r in v3:
        cat = r.get("metadata",{}).get("failure_category","unknown")
        by_cat.setdefault(cat, []).append(r)

    # All generators: cat → function
    generators = {
        "bare_update_vs_null_value":          gen_bare_update_vs_null_value,
        "bare_update_vs_duplicate_create":    gen_bare_update_vs_duplicate_create,
        "bare_update_vs_noop":                gen_bare_update_vs_noop,
        "bare_update_vs_delete_create_split": gen_bare_update_vs_delete_create_split,
        "compound_bare_update_vs_crosswire":  gen_compound_bare_update_vs_crosswire,
        "update_split_into_delete_create":    gen_update_split_into_delete_create,
        "hallucinated_old_value_should_be_create": gen_hallucinated_old_value_should_be_create,
        "key_isolation_vs_spurious_delete":   gen_key_isolation_vs_spurious_delete,
        "degenerate_action_should_be_noop":   gen_degenerate_action_should_be_noop,
        "under_trigger_tasks_events_declarative": gen_under_trigger_tasks_events_declarative,
        "implicit_linked_create":             gen_implicit_linked_create,
        "cascade_linked_entry_context_loss":  gen_cascade_linked_entry_context_loss,
        "over_action_scope_creep":            gen_over_action_scope_creep,
        "under_trigger_implicit_booking_no_anchor": gen_under_trigger_implicit_booking_no_anchor,
        "under_trigger_other_singleton_keys": gen_under_trigger_other_singleton_keys,
        "other":                              gen_other,
    }

    final = []
    print(f"{'Category':<52} {'Have':>5} {'Need':>6} {'Final':>6}")
    print("-"*72)

    for cat, gen_fn in generators.items():
        existing = by_cat.get(cat, [])
        have = len(existing)
        need = max(0, TARGET - have)
        
        if have >= TARGET:
            # Trim: randomly sample down
            sampled = random.sample(existing, TARGET)
        else:
            # Use all existing + generate the gap
            new_recs = gen_fn(need)
            sampled = existing + new_recs

        final.extend(sampled)
        print(f"{cat:<52} {have:>5} {need:>6} {len(sampled):>6}")

    random.shuffle(final)

    print("-"*72)
    print(f"{'TOTAL':<52} {'':>5} {'':>6} {len(final):>6}")
    print()

    # Final validation
    cats = {}
    issues = []
    policy_keys = {'name','preferred_name','college','preferred_language',
                   'conversation_style','default_placement_template','tasks_events'}
    for i,r in enumerate(final):
        cat = r.get("metadata",{}).get("failure_category","unknown")
        cats[cat] = cats.get(cat,0)+1
        try:
            c = json.loads(r["chosen"])
            for a in c.get("actions",[]):
                act = a.get("action","").upper()
                k = a.get("key","")
                if k not in policy_keys:
                    issues.append(f"Row {i}: invalid key {k!r}")
                if act=="UPDATE" and (a.get("new_value") is None or a.get("old_value") is None):
                    issues.append(f"Row {i}: UPDATE null value")
                if act=="DELETE" and not a.get("value"):
                    issues.append(f"Row {i}: DELETE missing value")
                if act=="CREATE" and not a.get("value"):
                    issues.append(f"Row {i}: CREATE missing value")
        except Exception as e:
            issues.append(f"Row {i}: {e}")

    print(f"Policy violations: {len(issues)}")
    for iss in issues[:5]:
        print(f"  {iss}")
    
    balanced = all(cats.get(c,0) == TARGET for c in generators)
    print(f"Perfectly balanced (all == {TARGET}): {balanced}")
    if not balanced:
        for c,n in cats.items():
            if n != TARGET:
                print(f"  {c}: {n}")

    out_path = "dataset/dpo_v3_final.jsonl"
    with open(out_path,"w") as f:
        for r in final:
            f.write(json.dumps(r)+"\n")
    print(f"\n✅ Saved {len(final)} balanced records → {out_path}")


if __name__ == "__main__":
    main()
