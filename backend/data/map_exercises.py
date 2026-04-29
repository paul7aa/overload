"""
Maps our messy 3213 training exercise names to the 1324 canonical exercises
in exercises_dataset/data/exercises.json, then applies the mapping so the
canonical dataset becomes the single source of truth.

Strategy:
  1. Fuzzy match each training name against all 1324 canonical names.
  2. Score >= AUTO_ACCEPT  → accept automatically (string is close enough)
  3. AUTO_ACCEPT > score >= LLM_THRESHOLD → ask Claude to confirm (yes/no)
  4. score < LLM_THRESHOLD → "no_match" (exercise not in our dataset)

Outputs:
  exercise_mapping.json        { "training_name": "canonical_name" | null }
  programs_detailed_canonical.csv  cleaned CSV using canonical names only
  exercise_map.json            { "canonical_name": integer_id } for the model
"""

import json
import os
import re
import time
import pandas as pd
from rapidfuzz import fuzz, process
import anthropic
from dotenv import load_dotenv

load_dotenv()

def clean_canonical(name: str) -> str:
    """
    Light clean applied to canonical names in the output — fixes real errors
    without destroying meaningful structure like hyphens or parentheses.
    """
    # Fix encoding artifact: cyrillic 'в' before degree sign
    name = name.replace("в°", "°")
    # Straighten curly apostrophes
    name = name.replace("‘", "'").replace("’", "'")
    return name.strip()


def normalize(name: str) -> str:
    """
    Clean a name purely for fuzzy comparison — the original is always preserved
    in the output. Applied identically to both training and canonical names so
    the same noise on either side doesn't create artificial mismatches.
    """
    s = name.lower()
    # Fix encoding artifact in canonical dataset: cyrillic 'в' before degree sign
    s = s.replace("в°", "°")
    # Expand common shorthands
    s = s.replace("w/", "with ")
    s = s.replace("w /", "with ")
    # Degree symbol → word (so "45°" and "45 degree" both become "45 degrees")
    s = s.replace("°", " degrees ")
    # Strip version suffixes from canonical names ("v. 2", "v. 3")
    s = re.sub(r'\bv\.\s*\d+\b', '', s)
    # Curly apostrophes → straight (farmer's → farmer's)
    s = s.replace("’", "'").replace("‘", "'")
    # Remove emoji and other non-latin characters
    s = s.encode("ascii", errors="ignore").decode()
    # Collapse remaining punctuation to spaces (keeps alphanumeric and spaces)
    s = re.sub(r'[^\w\s]', ' ', s)
    # Collapse multiple spaces
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# ── Thresholds ────────────────────────────────────────────────────────────────
AUTO_ACCEPT    = 95  # string so close we trust it without LLM
LLM_THRESHOLD  = 60   # below this → almost certainly a different exercise
BATCH_SIZE     = 32   # how many pairs to send to Claude per API call

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv("data/programs_detailed.csv")
training_names = df["exercise_name"].unique().tolist()

with open("data/exercises_dataset/data/exercises.json") as f:
    dataset = json.load(f)

# We'll match against the canonical names from the clean dataset.
# We lowercase both sides for matching only — the original canonical name is
# preserved in the output so the mapping stays correct.
canonical_names = [clean_canonical(ex["name"]) for ex in dataset]
canonical_lower = [normalize(n) for n in canonical_names]

print(f"Training exercises : {len(training_names)}")
print(f"Canonical exercises: {len(canonical_names)}")

# ── Step 1: Fuzzy matching ─────────────────────────────────────────────────────
# For each training name, find the single best match in the canonical list.
# token_sort_ratio ignores word order ("Leg Press Machine" == "Machine Leg Press").
# We compare lowercased strings so case differences don't hurt the score.

auto_mapped   = {}   # training_name → canonical_name  (high confidence)
needs_llm     = []   # list of (training_name, canonical_name, score) tuples
no_match      = []   # training_names with no close candidate

print("\nRunning fuzzy matching...")
for name in training_names:
    result = process.extractOne(normalize(name), canonical_lower, scorer=fuzz.token_sort_ratio)
    if result is None:
        no_match.append(name)
        continue

    _, score, idx = result
    canonical = canonical_names[idx]
    # If we matched a "v. 2" variant but the base version also exists, prefer the base.
    base = re.sub(r'\s*v\.\s*\d+', '', canonical).strip()
    if base != canonical and base in canonical_names:
        canonical = base
    if score >= AUTO_ACCEPT:
        auto_mapped[name] = canonical
    elif score >= LLM_THRESHOLD:
        needs_llm.append((name, canonical, score))
    else:
        no_match.append(name)

print(f"  Auto-accepted : {len(auto_mapped)}")
print(f"  Needs LLM     : {len(needs_llm)}")
print(f"  No match (<{LLM_THRESHOLD}%): {len(no_match)}")

auto_accepted_list = sorted(
    ({"training": k, "canonical": v} for k, v in auto_mapped.items()),
    key=lambda x: x["training"]
)
with open("data/exercise_auto_accepted.json", "w") as f:
    json.dump(auto_accepted_list, f, indent=2)
# ── Step 2: LLM classification ─────────────────────────────────────────────────
# We send Claude a batch of pairs and ask it to return JSON.
# Each item in the batch: { "training": "...", "canonical": "..." }
# Claude returns a parallel list of true/false decisions.
#
# Why batch? Each API call has latency overhead. Sending 40 pairs at once
# is ~40x more efficient than one call per pair.

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

def classify_batch(pairs: list[tuple[str, str, float]]) -> list[bool]:
    """
    Ask Claude whether each (training_name, canonical_name) pair refers to the
    same physical exercise. Returns a list of booleans, one per pair.
    """
    items = [
        {"i": i, "training": t, "canonical": c}
        for i, (t, c, _) in enumerate(pairs)
    ]

    prompt = f"""You are classifying exercise name pairs.
For each pair, decide whether the training name and the canonical name refer to
the same physical exercise (even if worded differently).

Rules:
- "Barbell Squat" and "Squat (Barbell)" → SAME
- "Dumbbell Curl" and "Barbell Curl" → DIFFERENT (different equipment)
- "Push-up" and "Push Up" → SAME
- "Leg Press" and "Hack Squat" → DIFFERENT (different movement)
- Minor spelling/punctuation differences → SAME

Return ONLY a JSON array of objects with keys "i" and "same" (boolean).
No explanation, no markdown fences — raw JSON only.

Pairs:
{json.dumps(items, indent=2)}"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()
    results = json.loads(raw)
    # Sort by index in case Claude reorders them
    results.sort(key=lambda x: x["i"])
    return [r["same"] for r in results]


llm_mapped  = {}   # training_name → canonical_name  (LLM confirmed)
llm_rejected = []  # training_names the LLM said are different exercises

if needs_llm:
    print(f"\nSending {len(needs_llm)} pairs to Claude in batches of {BATCH_SIZE}...")
    total_batches = (len(needs_llm) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_i in range(total_batches):
        batch = needs_llm[batch_i * BATCH_SIZE : (batch_i + 1) * BATCH_SIZE]
        print(f"  Batch {batch_i + 1}/{total_batches} ({len(batch)} pairs)...", end=" ", flush=True)

        decisions = classify_batch(batch)
        for (training, canonical, score), is_same in zip(batch, decisions):
            if is_same:
                llm_mapped[training] = canonical
            else:
                llm_rejected.append(training)

        print(f"done ({sum(decisions)} matched)")
        # Be polite to the API — small pause between batches
        if batch_i < total_batches - 1:
            time.sleep(0.5)

# ── Step 3: Combine and save ───────────────────────────────────────────────────
final_mapping = {}
final_mapping.update(auto_mapped)
final_mapping.update(llm_mapped)
for name in no_match + llm_rejected:
    final_mapping[name] = None   # null = no canonical match found

with open("data/exercise_mapping.json", "w") as f:
    json.dump(final_mapping, f, indent=2)

matched   = sum(1 for v in final_mapping.values() if v is not None)
unmatched = sum(1 for v in final_mapping.values() if v is None)

unmatched_names = sorted(k for k, v in final_mapping.items() if v is None)
with open("data/exercise_no_match.json", "w") as f:
    json.dump(unmatched_names, f, indent=2)

# Save auto-accepted pairs so you can spot-check that fuzzy matching didn't
# make obvious mistakes at the 95% threshold.

print(f"\nMapping saved to data/exercise_mapping.json")
print(f"  Matched  : {matched}/{len(training_names)}")
print(f"  Unmatched: {unmatched}/{len(training_names)} — see data/exercise_no_match.json")
print(f"  Auto-accepted: {len(auto_mapped)} — see data/exercise_auto_accepted.json")

# ── Step 4: Apply mapping to training CSV ──────────────────────────────────────
# Replace every messy exercise_name with its canonical equivalent.
# Drop rows whose exercise had no match — those exercises won't exist in the app,
# so training on them would be noise.

print("\nApplying mapping to programs_detailed.csv...")
df["exercise_name"] = df["exercise_name"].map(final_mapping)

before = len(df)
df = df.dropna(subset=["exercise_name"])
after  = len(df)

df.to_csv("data/programs_detailed_canonical.csv", index=False)
print(f"  Rows before: {before:,}")
print(f"  Rows after : {after:,}  ({before - after:,} dropped — unmatched exercises)")
print(f"  Saved to   : data/programs_detailed_canonical.csv")

# ── Step 5: Rebuild exercise_map.json from canonical dataset ───────────────────
# The model needs to encode exercise names as integers (a lookup table).
# Previously this was built from the messy training data; now we build it
# from the canonical dataset so the IDs are stable and match the app's exercises.
#
# We only include exercises that actually appear in the filtered training data —
# no point encoding an exercise the model has never seen.

exercises_in_training = set(df["exercise_name"].unique())
exercise_map = {
    ex["name"]: idx
    for idx, ex in enumerate(dataset)
    if ex["name"] in exercises_in_training
}

with open("data/exercise_map.json", "w") as f:
    json.dump(exercise_map, f, indent=2)

print(f"\nRebuilt exercise_map.json")
print(f"  Canonical exercises in training data: {len(exercise_map)}/{len(dataset)}")
print(f"  Saved to: data/exercise_map.json")
print(f"\nAll done! Re-run preprocess/train with programs_detailed_canonical.csv.")
