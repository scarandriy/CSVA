import json
from collections import Counter
from statsmodels.stats.contingency_tables import mcnemar
import re

def extract_scam_probability(response_text):
    # 1. Strip leading/trailing whitespace
    cleaned = response_text.strip()

    # 2. If it starts with or json, remove exactly that (but keep everything inside).
    #    We allow an optional "json" (case‐insensitive) right after the opening backticks.
    cleaned = re.sub(r"^(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)

    # 3. If it ends with a closing, remove exactly that.
    cleaned = re.sub(r"\s*$", "", cleaned)

    # 4. Remove any backslashes (in case the JSON was double‐escaped)
    cleaned = cleaned.replace("\\", "")

    cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned)

    cleaned = cleaned.replace("_", " ")


    # 5. Collapse multiple whitespace into single spaces (so we can reliably search)
    cleaned = re.sub(r"\s+", " ", cleaned)

    # 6. Now search for "Scam Probability": <integer>
    pattern = re.compile(r"""
        ["']?Scam\ Probability["']?     # optional quotes around Scam Probability
        \s*:\s*                         # colon with optional spaces
        (\d+)                           # capture one or more digits
        """, re.IGNORECASE | re.VERBOSE)

    match = pattern.search(cleaned)
    if match:
        return int(match.group(1))
    return None

def load_correct_flags(log_path):
    """
    Returns a list of 1/0 flags: 1 if that record was classified correctly.
    Assumes each record has 'image' and 'response', and that you can
    parse the integer 'Scam Probability' and compare to ground truth folder.
    """
    flags = []
    data = json.load(open(log_path, 'r', encoding='utf-8'))
    for rec in data:
        path = rec['image'].lower()
        prob = extract_scam_probability(rec['response'])
        # ground truth: folder name
        is_scam = 'scam2' in path
        # prediction: prob ≥ 3 → scam
        pred = (prob is not None) and (prob >= 3)
        correct = int(pred == is_scam)
        flags.append(correct)
    return flags

# Load flags for both runs
flags_A = load_correct_flags('logs/gemma4b+reasoning-902.json')
flags_B = load_correct_flags('logs/gemma4b-902.json')

# Build contingency counts
counts = Counter((a, b) for (a, b) in zip(flags_A, flags_B))
# counts[(1,0)] = b ; counts[(0,1)] = c
table = [[counts[(1,1)], counts[(1,0)]],
         [counts[(0,1)], counts[(0,0)]]]

# Perform McNemar’s test (exact)
result = mcnemar(table, exact=True)
print("Contingency table:")
print(f"    A✔ B✔ = {table[0][0]:>4}   A✔ B✘ = {table[0][1]:>4}")
print(f"    A✘ B✔ = {table[1][0]:>4}   A✘ B✘ = {table[1][1]:>4}")
print(f"McNemar’s χ² = {result.statistic:.3f}, p-value = {result.pvalue:.3f}")
if result.pvalue < 0.05:
    print("→ Significant difference in accuracy.")
else:
    print("→ No significant difference detected.")
