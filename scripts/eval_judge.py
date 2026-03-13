"""
eval_judge.py

Automated judge evaluation pipeline — Vertex AI:
  Sample 20 easy + 20 medium + 20 hard problems with test cases.
  Generate Qwen 7B style traces via Gemini, score them, report per-tier stats.

Difficulty mapping:
  LCB:  easy / medium / hard
  APPS: introductory→easy, interview→medium, competition→hard

Auth: Uses Application Default Credentials (ADC)
  gcloud auth application-default login

Usage:
    python scripts/eval_judge.py --project grpo-reasoning-prj
    python scripts/eval_judge.py --project grpo-reasoning-prj --per-tier 5  # quick test
"""

import json
import random
import argparse
import asyncio
import sys
from pathlib import Path

sys.set_int_max_str_digits(100000)

import httpx
import google.auth
import google.auth.transport.requests

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────

DATA_DIR    = Path(__file__).parent.parent / "data" / "clean"
APPS_PATH   = DATA_DIR / "apps_clean.jsonl"
LCB_PATH    = DATA_DIR / "lcb_seen_clean.jsonl"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "judge_eval_results.json"

PER_TIER    = 20
RANDOM_SEED = 99

GEMINI_MODEL       = "gemini-2.0-flash-001"
GEMINI_CONCURRENCY = 5
GCP_REGION         = "us-central1"

# Map raw difficulty labels → easy / medium / hard
APPS_DIFF_MAP = {
    "introductory": "easy",
    "interview":    "medium",
    "competition":  "hard",
}
LCB_DIFF_MAP = {
    "easy":   "easy",
    "medium": "medium",
    "hard":   "hard",
}

TIERS = ["easy", "medium", "hard"]


def get_token() -> str:
    creds, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    creds.refresh(google.auth.transport.requests.Request())
    return creds.token


def vertex_url(project_id: str) -> str:
    return (
        f"https://{GCP_REGION}-aiplatform.googleapis.com/v1/projects/{project_id}"
        f"/locations/{GCP_REGION}/publishers/google/models/{GEMINI_MODEL}:generateContent"
    )


# ─────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    problems = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    return problems


def normalize_difficulty(p: dict, source: str) -> str | None:
    raw = p.get("difficulty", "")
    if source == "apps":
        return APPS_DIFF_MAP.get(raw)
    elif source == "lcb":
        return LCB_DIFF_MAP.get(raw)
    return None


def has_test_cases(p: dict) -> bool:
    return len(p.get("test_cases", [])) > 0


def format_problem(p: dict, source: str, tier: str) -> dict:
    tcs = p.get("test_cases", [])[:3]
    formatted = []
    for i, tc in enumerate(tcs):
        formatted.append({
            "test_number": i + 1,
            "input": tc.get("input", "")[:300],
            "expected_output": tc.get("output", "")[:150],
        })
    return {
        "source": source,
        "platform": p.get("platform", source),
        "difficulty": tier,
        "problem": p.get("question", p.get("problem_statement", ""))[:2000],
        "test_cases": formatted,
        "starter_code": p.get("starter_code", None),
    }


def sample_problems(per_tier: int) -> list[tuple[str, dict]]:
    random.seed(RANDOM_SEED)

    # Build pool per tier
    pools: dict[str, list[tuple[str, dict]]] = {t: [] for t in TIERS}

    if APPS_PATH.exists():
        for p in load_jsonl(APPS_PATH):
            tier = normalize_difficulty(p, "apps")
            if tier and has_test_cases(p):
                pools[tier].append(("apps", p))
    else:
        print(f"WARNING: {APPS_PATH} not found")

    if LCB_PATH.exists():
        for p in load_jsonl(LCB_PATH):
            tier = normalize_difficulty(p, "lcb")
            if tier and has_test_cases(p):
                pools[tier].append(("lcb", p))
    else:
        print(f"WARNING: {LCB_PATH} not found")

    # Report pool sizes
    for tier in TIERS:
        print(f"  {tier:6s}: {len(pools[tier])} problems with test cases")

    # Sample per tier
    sampled = []
    for tier in TIERS:
        pool = pools[tier]
        n = min(per_tier, len(pool))
        if n == 0:
            print(f"  WARNING: no {tier} problems found")
            continue
        chosen = random.sample(pool, n)
        sampled.extend([(src, format_problem(p, src, tier)) for src, p in chosen])
        print(f"  Sampled {n} {tier} problems")

    print(f"\nTotal: {len(sampled)} problems\n")
    return sampled


# ─────────────────────────────────────────
# Vertex AI API call
# ─────────────────────────────────────────

async def gemini_call(
    client: httpx.AsyncClient,
    url: str,
    token: str,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
    sem: asyncio.Semaphore,
) -> str:
    async with sem:
        payload = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        for attempt in range(5):
            try:
                resp = await client.post(url, json=payload, headers=headers, timeout=60)
                if resp.status_code == 429:
                    wait = 10 * (attempt + 1)
                    print(f"  Rate limited, waiting {wait}s...")
                    await asyncio.sleep(wait)
                    continue
                if not resp.is_success:
                    print(f"  Error body: {resp.text[:300]}")
                resp.raise_for_status()
                data = resp.json()
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()
            except Exception as e:
                if attempt == 4:
                    print(f"  API error after 5 attempts: {e}")
                    return ""
                wait = 5 * (attempt + 1)
                print(f"  Attempt {attempt+1} failed ({e}), retrying in {wait}s...")
                await asyncio.sleep(wait)
        return ""


# ─────────────────────────────────────────
# Trace generation
# ─────────────────────────────────────────

TRACE_SYSTEM = """You are simulating a 7B parameter code generation model solving 
competitive programming problems. You have limited reasoning ability.

Produce exactly 6 reasoning steps labeled [STEP 1] through [STEP 6].

Produce VARIED quality traces — not always correct:
- Sometimes identify the right algorithm but implement it wrongly
- Sometimes start with a brute force approach that is too slow
- Sometimes make an edge case error
- Sometimes get steps 1-3 right then go wrong on step 4-5
- Occasionally produce a fully correct trace
- Occasionally produce a mostly wrong trace

Each step: 2-4 sentences. Be specific about algorithms and data structures.
Output ONLY the 6 labeled steps, nothing else."""


def build_trace_prompt(problem: dict) -> str:
    tc_text = ""
    for tc in problem["test_cases"][:2]:
        tc_text += f"Input: {tc['input'][:200]}\nOutput: {tc['expected_output'][:100]}\n\n"
    starter = ""
    if problem.get("starter_code"):
        starter = f"\nStarter code:\n{problem['starter_code']}\n"
    return f"""Problem ({problem['difficulty']} difficulty):
{problem['problem'][:1500]}

Example test cases:
{tc_text}{starter}Generate 6 reasoning steps as a limited 7B model would. Be realistic — make some mistakes."""


# ─────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────

SCORE_SYSTEM = """You are evaluating reasoning steps produced by a code generation model 
for a competitive programming problem.

Score each reasoning step from 0.0 to 1.0:
  1.0 = logically sound, relevant, shows genuine understanding
  0.5 = partially correct, right direction but vague or missing key insight
  0.0 = wrong approach, irrelevant, or logically incorrect

Also provide an overall score (0.0 to 1.0) for the full reasoning trace.

Be strict and consistent:
- Right algorithm, wrong application: 0.3-0.5
- Vague but not wrong: 0.4-0.6
- Clear logical error: 0.0-0.2
- Fully correct and specific: 0.8-1.0

Return ONLY valid JSON, no markdown fences, nothing else:
{"step_scores": [s1, s2, s3, s4, s5, s6], "overall": score}"""


def build_score_prompt(problem: dict, trace: str) -> str:
    tc_text = ""
    for tc in problem["test_cases"][:2]:
        tc_text += f"Input: {tc['input'][:200]}\nOutput: {tc['expected_output'][:100]}\n\n"
    return f"""Problem ({problem['difficulty']} difficulty):
{problem['problem'][:1500]}

Example test cases:
{tc_text}Model reasoning trace:
{trace}

Score the 6 steps and provide an overall score."""


def parse_scores(raw: str) -> dict:
    try:
        clean = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(clean)
        assert "step_scores" in result and "overall" in result
        assert len(result["step_scores"]) == 6
        return result
    except Exception:
        print(f"  Score parse error. Raw: {raw[:120]}")
        return {"step_scores": [0.5] * 6, "overall": 0.5, "parse_error": True}


# ─────────────────────────────────────────
# Stats helpers
# ─────────────────────────────────────────

def tier_summary(results: list[dict], tier: str) -> dict:
    subset = [r for r in results if r["difficulty"] == tier]
    if not subset:
        return {}
    scores = [r["scores"].get("overall", 0.5) for r in subset]
    mean = sum(scores) / len(scores)
    std  = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5
    return {
        "n": len(subset),
        "mean": round(mean, 3),
        "std":  round(std, 3),
        "min":  round(min(scores), 3),
        "max":  round(max(scores), 3),
        "distribution": {
            "0.0-0.2": sum(1 for s in scores if s <= 0.2),
            "0.2-0.4": sum(1 for s in scores if 0.2 < s <= 0.4),
            "0.4-0.6": sum(1 for s in scores if 0.4 < s <= 0.6),
            "0.6-0.8": sum(1 for s in scores if 0.6 < s <= 0.8),
            "0.8-1.0": sum(1 for s in scores if s > 0.8),
        },
    }


# ─────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────

async def run(project_id: str, per_tier: int):
    print("=== Step 1: Sampling problems ===")
    problems = sample_problems(per_tier)
    n = len(problems)

    print("Authenticating with Vertex AI...")
    token = get_token()
    url = vertex_url(project_id)
    print(f"Endpoint: {url}\n")

    sem = asyncio.Semaphore(GEMINI_CONCURRENCY)

    print("=== Step 2: Generating traces (Gemini simulating Qwen 7B) ===")
    async with httpx.AsyncClient() as client:
        traces = await asyncio.gather(*[
            gemini_call(
                client, url, token,
                system=TRACE_SYSTEM,
                user=build_trace_prompt(p),
                max_tokens=1000,
                temperature=0.7,
                sem=sem,
            )
            for _, p in problems
        ])
    for i, (_, p) in enumerate(problems):
        status = "ok" if traces[i] else "EMPTY"
        print(f"  [{i+1:02d}/{n}] {p['difficulty']:6s} {status}")

    print(f"\n=== Step 3: Scoring traces (Gemini as judge, temp=0) ===")
    token = get_token()  # refresh before second batch
    async with httpx.AsyncClient() as client:
        raw_scores = await asyncio.gather(*[
            gemini_call(
                client, url, token,
                system=SCORE_SYSTEM,
                user=build_score_prompt(p, t),
                max_tokens=150,
                temperature=0.0,
                sem=sem,
            )
            for (_, p), t in zip(problems, traces)
        ])

    scores = [parse_scores(r) for r in raw_scores]
    for i, s in enumerate(scores):
        _, p = problems[i]
        print(f"  [{i+1:02d}/{n}] {p['difficulty']:6s} overall={s.get('overall', 0):.2f}")

    # ── compile results ──
    results = []
    for (src, p), trace, score in zip(problems, traces, scores):
        results.append({
            "source": src,
            "platform": p["platform"],
            "difficulty": p["difficulty"],
            "problem_snippet": p["problem"][:200],
            "trace": trace,
            "scores": score,
        })

    # ── per-tier summary ──
    by_tier = {tier: tier_summary(results, tier) for tier in TIERS}

    overall_scores = [r["scores"].get("overall", 0.5) for r in results]
    mean = sum(overall_scores) / len(overall_scores)
    std  = (sum((s - mean) ** 2 for s in overall_scores) / len(overall_scores)) ** 0.5

    summary = {
        "total_problems": n,
        "overall_mean": round(mean, 3),
        "overall_std":  round(std, 3),
        "by_tier": by_tier,
        "parse_errors": sum(1 for r in results if r["scores"].get("parse_error")),
    }

    output = {"summary": summary, "results": results}
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nFull results → {OUTPUT_PATH}")


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, help="GCP project ID")
    parser.add_argument("--per-tier", type=int, default=PER_TIER,
                        help="Problems per difficulty tier (default 20, use 5 for quick test)")
    args = parser.parse_args()

    asyncio.run(run(args.project, args.per_tier))


if __name__ == "__main__":
    main()