import json

with open("../data/clean/lcb_seen_clean.jsonl") as f:
    problems = [json.loads(l) for l in f]

counts = [len(p["test_cases"]) for p in problems]
print(f"Total problems: {len(problems)}")
print(f"Avg test cases: {sum(counts)/len(counts):.1f}")
print(f"Min: {min(counts)}, Max: {max(counts)}, Median: {sorted(counts)[len(counts)//2]}")
print(f"Problems with only 1 test: {sum(1 for c in counts if c == 1)}")
print(f"Problems with 2-3 tests:   {sum(1 for c in counts if 2 <= c <= 3)}")
print(f"Problems with 4+ tests:    {sum(1 for c in counts if c >= 4)}")