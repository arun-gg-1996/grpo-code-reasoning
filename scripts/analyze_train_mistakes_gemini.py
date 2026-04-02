#!/usr/bin/env python3
"""
Analyze training completion mistakes with Gemini and track evolution over time.

This script:
1) Loads train_details.jsonl (latest by default)
2) Selects per-problem snapshots (first/last by default)
3) Calls Gemini to classify mistake type for each snapshot
4) Produces:
   - overall mistake distribution
   - early/mid/late mistake distribution
   - first->last per-problem evolution summary
5) Writes a report JSON under results/analysis/

Example:
  python scripts/analyze_train_mistakes_gemini.py --points first,last
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Tuple

import httpx

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import GEMINI_API_KEY, JUDGE_MODEL

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


VERTEX_URL_TMPL = "https://aiplatform.googleapis.com/v1/publishers/google/models/{model}:generateContent"

MISTAKE_TYPES = [
    "no_code_or_empty_output",
    "prompt_leak_or_garbage_output",
    "wrong_interface_or_function_name",
    "python_syntax_error",
    "runtime_or_timeout_risk",
    "logic_incorrect",
    "partial_correct",
    "likely_correct_format_minor_issue",
]

MISTAKE_RANK = {
    "no_code_or_empty_output": 0,
    "prompt_leak_or_garbage_output": 1,
    "wrong_interface_or_function_name": 2,
    "python_syntax_error": 3,
    "runtime_or_timeout_risk": 4,
    "logic_incorrect": 5,
    "partial_correct": 6,
    "likely_correct_format_minor_issue": 7,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=None, help="Path to train_details.jsonl (default: latest results/train/*/train_details.jsonl)")
    p.add_argument("--points", default="first,last", help="Comma list among: first,mid,last")
    p.add_argument("--max-problems", type=int, default=0, help="Limit number of unique problems (0 = all)")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--timeout", type=float, default=45.0)
    p.add_argument("--model", default=JUDGE_MODEL)
    p.add_argument("--max-problem-chars", type=int, default=1200)
    p.add_argument("--max-completion-chars", type=int, default=1800)
    p.add_argument("--max-code-chars", type=int, default=1200)
    p.add_argument("--max-retries", type=int, default=6, help="Retries for 429/5xx/transient network errors")
    p.add_argument("--base-backoff", type=float, default=1.0, help="Base seconds for exponential backoff")
    p.add_argument("--max-backoff", type=float, default=30.0, help="Maximum backoff seconds")
    return p.parse_args()


def load_rows(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_idx"] = idx
            rows.append(row)
    return rows


def key_of(row: Dict[str, Any]) -> str:
    src = row.get("source", "unknown")
    if src == "apps":
        return f"apps:{row.get('problem_id')}"
    return f"{src}:{row.get('question_id')}"


def select_snapshots(rows: List[Dict[str, Any]], points: List[str], max_problems: int) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[key_of(r)].append(r)

    keys = list(grouped.keys())
    if max_problems > 0:
        keys = keys[:max_problems]

    selected: List[Dict[str, Any]] = []
    for k in keys:
        g = grouped[k]
        picks: List[Dict[str, Any]] = []
        if "first" in points:
            picks.append(g[0])
        if "mid" in points:
            picks.append(g[len(g) // 2])
        if "last" in points:
            picks.append(g[-1])
        # dedupe by index
        seen = set()
        for r in picks:
            if r["_idx"] in seen:
                continue
            seen.add(r["_idx"])
            selected.append(r)
    return selected


def _strip_json(text: str) -> Dict[str, Any] | None:
    text = text.strip().strip("`").strip()
    if text.startswith("json"):
        text = text[4:].strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None


def _clip(s: str, n: int) -> str:
    return (s or "")[:n]


def build_prompt(row: Dict[str, Any], max_problem_chars: int, max_completion_chars: int, max_code_chars: int) -> str:
    return (
        "Classify the mistake in this model completion for a coding task.\n"
        "Return ONLY JSON with this exact schema:\n"
        "{"
        "\"mistake_type\":\"one_of_enum\","
        "\"confidence\":0_to_1,"
        "\"completeness_score\":0_to_1,"
        "\"short_reason\":\"one short sentence\","
        "\"suggested_fix\":\"one short sentence\""
        "}\n\n"
        "completeness_score should estimate how close the completion is to a fully correct solution.\n"
        "Use 0.0 when clearly unusable, 1.0 when very likely correct, and intermediate values otherwise.\n\n"
        "Allowed mistake_type enum:\n"
        + ", ".join(MISTAKE_TYPES)
        + "\n\n"
        f"Source: {row.get('source')}\n"
        f"Difficulty: {row.get('difficulty')}\n"
        f"Execution score: {row.get('execution_score')}\n"
        f"Problem key: {key_of(row)}\n"
        f"Problem statement:\n{_clip(row.get('problem_statement', ''), max_problem_chars)}\n\n"
        f"Completion text:\n{_clip(row.get('completion_text', ''), max_completion_chars)}\n\n"
        f"Extracted code:\n{_clip(row.get('extracted_code', ''), max_code_chars)}\n"
    )


async def classify_one(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    api_key: str,
    model: str,
    timeout: float,
    row: Dict[str, Any],
    max_problem_chars: int,
    max_completion_chars: int,
    max_code_chars: int,
    max_retries: int,
    base_backoff: float,
    max_backoff: float,
) -> Dict[str, Any]:
    body = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": build_prompt(
                            row=row,
                            max_problem_chars=max_problem_chars,
                            max_completion_chars=max_completion_chars,
                            max_code_chars=max_code_chars,
                        )
                    }
                ],
            }
        ],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 256},
    }
    url = f"{VERTEX_URL_TMPL.format(model=model)}?key={api_key}"

    async with sem:
        out = {
            "_idx": row["_idx"],
            "problem_key": key_of(row),
            "source": row.get("source"),
            "difficulty": row.get("difficulty"),
            "execution_score": float(row.get("execution_score", 0.0) or 0.0),
            "mistake_type": "logic_incorrect",
            "confidence": 0.0,
            "completeness_score": None,
            "short_reason": "fallback",
            "suggested_fix": "fallback",
            "api_ok": False,
            "attempts": 0,
        }
        for attempt in range(max(1, max_retries) + 1):
            out["attempts"] = attempt + 1
            try:
                resp = await client.post(url, json=body, timeout=timeout)
                if resp.status_code == 429 or 500 <= resp.status_code < 600:
                    if attempt < max_retries:
                        delay = min(max_backoff, base_backoff * (2**attempt))
                        await asyncio.sleep(delay + random.uniform(0.0, 0.25))
                        continue
                resp.raise_for_status()
                data = resp.json()
                txt = data["candidates"][0]["content"]["parts"][0]["text"]
                parsed = _strip_json(txt)
                if parsed is None:
                    out["short_reason"] = "unparseable json response"
                    return out

                mt = str(parsed.get("mistake_type", "logic_incorrect"))
                if mt not in MISTAKE_TYPES:
                    mt = "logic_incorrect"
                conf = parsed.get("confidence", 0.0)
                try:
                    conf = max(0.0, min(1.0, float(conf)))
                except Exception:
                    conf = 0.0

                comp = parsed.get("completeness_score", None)
                comp_val = None
                try:
                    if comp is not None:
                        comp_val = max(0.0, min(1.0, float(comp)))
                except Exception:
                    comp_val = None

                out["mistake_type"] = mt
                out["confidence"] = conf
                out["completeness_score"] = comp_val
                out["short_reason"] = str(parsed.get("short_reason", ""))[:300]
                out["suggested_fix"] = str(parsed.get("suggested_fix", ""))[:300]
                out["api_ok"] = True
                return out
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                retryable = any(x in err for x in ["429", "timed out", "Timeout", "ConnectError", "ReadError"])
                if retryable and attempt < max_retries:
                    delay = min(max_backoff, base_backoff * (2**attempt))
                    await asyncio.sleep(delay + random.uniform(0.0, 0.25))
                    continue
                out["short_reason"] = f"api_error: {e}"
                return out
        out["short_reason"] = "api_error: retries_exhausted"
        return out


def segment_name(idx: int, n: int) -> str:
    if n <= 0:
        return "unknown"
    x = idx / n
    if x < 1 / 3:
        return "early"
    if x < 2 / 3:
        return "mid"
    return "late"


def summarize(results: List[Dict[str, Any]], total_rows: int) -> Dict[str, Any]:
    overall = Counter(r["mistake_type"] for r in results)
    by_segment: Dict[str, Counter] = {"early": Counter(), "mid": Counter(), "late": Counter()}
    for r in results:
        seg = segment_name(r["_idx"], total_rows)
        by_segment[seg][r["mistake_type"]] += 1

    by_problem: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_problem[r["problem_key"]].append(r)
    for k in by_problem:
        by_problem[k].sort(key=lambda x: x["_idx"])

    transitions = Counter()
    improved = 0
    worsened = 0
    unchanged = 0
    for k, vals in by_problem.items():
        if len(vals) < 2:
            continue
        a = vals[0]
        b = vals[-1]
        transitions[(a["mistake_type"], b["mistake_type"])] += 1
        rank_a = MISTAKE_RANK.get(a["mistake_type"], 0)
        rank_b = MISTAKE_RANK.get(b["mistake_type"], 0)
        if rank_b > rank_a:
            improved += 1
        elif rank_b < rank_a:
            worsened += 1
        else:
            unchanged += 1

    top_transitions = [
        {"from": f, "to": t, "count": c}
        for (f, t), c in transitions.most_common(20)
    ]

    comp_pairs = [
        (float(r["completeness_score"]), float(r["execution_score"]))
        for r in results
        if r.get("api_ok") and r.get("completeness_score") is not None
    ]
    comp_count = len(comp_pairs)
    comp_stats: Dict[str, Any] = {
        "count": comp_count,
        "mean_completeness": None,
        "mean_execution": None,
        "mae_vs_execution": None,
        "pearson_corr": None,
        "high_comp_low_exec_frac": None,
    }
    if comp_count > 0:
        comps = [x for x, _ in comp_pairs]
        execs = [y for _, y in comp_pairs]
        mean_c = sum(comps) / comp_count
        mean_e = sum(execs) / comp_count
        mae = sum(abs(x - y) for x, y in comp_pairs) / comp_count

        # Pearson correlation without numpy
        var_c = sum((x - mean_c) ** 2 for x in comps) / comp_count
        var_e = sum((y - mean_e) ** 2 for y in execs) / comp_count
        cov = sum((x - mean_c) * (y - mean_e) for x, y in comp_pairs) / comp_count
        corr = None
        if var_c > 0 and var_e > 0:
            corr = cov / ((var_c**0.5) * (var_e**0.5))

        high_comp_low_exec = sum(1 for x, y in comp_pairs if x >= 0.7 and y <= 0.2) / comp_count
        comp_stats = {
            "count": comp_count,
            "mean_completeness": round(mean_c, 4),
            "mean_execution": round(mean_e, 4),
            "mae_vs_execution": round(mae, 4),
            "pearson_corr": None if corr is None else round(corr, 4),
            "high_comp_low_exec_frac": round(high_comp_low_exec, 4),
        }

    # First->last completeness movement by problem
    comp_improved = 0
    comp_worsened = 0
    comp_unchanged = 0
    comp_deltas: List[float] = []
    for vals in by_problem.values():
        vals2 = [v for v in vals if v.get("api_ok") and v.get("completeness_score") is not None]
        if len(vals2) < 2:
            continue
        a = float(vals2[0]["completeness_score"])
        b = float(vals2[-1]["completeness_score"])
        d = b - a
        comp_deltas.append(d)
        if d > 0.05:
            comp_improved += 1
        elif d < -0.05:
            comp_worsened += 1
        else:
            comp_unchanged += 1
    comp_move = {
        "problems_with_2plus_points": int(comp_improved + comp_worsened + comp_unchanged),
        "improved_count": int(comp_improved),
        "worsened_count": int(comp_worsened),
        "unchanged_count": int(comp_unchanged),
        "mean_delta": None if not comp_deltas else round(sum(comp_deltas) / len(comp_deltas), 4),
    }

    return {
        "total_classified": len(results),
        "api_ok_count": sum(1 for r in results if r["api_ok"]),
        "overall_mistake_counts": dict(overall),
        "segment_mistake_counts": {k: dict(v) for k, v in by_segment.items()},
        "completeness_vs_execution": comp_stats,
        "completeness_evolution": comp_move,
        "evolution_summary": {
            "problems_with_2plus_points": int(improved + worsened + unchanged),
            "improved_count": int(improved),
            "worsened_count": int(worsened),
            "unchanged_count": int(unchanged),
            "top_transitions": top_transitions,
        },
    }


async def run_async(args: argparse.Namespace) -> int:
    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY is not set (config reads gemini_api_key env).")
        return 2

    in_path = args.input
    if not in_path:
        matches = sorted(glob.glob("results/train/*/train_details.jsonl"))
        if not matches:
            print("ERROR: no results/train/*/train_details.jsonl found")
            return 2
        in_path = matches[-1]

    points = [x.strip() for x in args.points.split(",") if x.strip()]
    for p in points:
        if p not in {"first", "mid", "last"}:
            print(f"ERROR: invalid point '{p}' (use first/mid/last)")
            return 2

    rows = load_rows(in_path)
    selected = select_snapshots(rows, points=points, max_problems=args.max_problems)
    if not selected:
        print("ERROR: no rows selected")
        return 2

    print(f"Input: {in_path}")
    print(f"Total rows: {len(rows)}")
    print(f"Selected snapshots: {len(selected)} (points={points}, max_problems={args.max_problems or 'all'})")
    print(f"Model: {args.model} | concurrency={args.concurrency}")

    sem = asyncio.Semaphore(max(1, args.concurrency))
    async with httpx.AsyncClient() as client:
        tasks = [
            asyncio.create_task(
                classify_one(
                    client=client,
                    sem=sem,
                    api_key=GEMINI_API_KEY,
                    model=args.model,
                    timeout=args.timeout,
                    row=r,
                    max_problem_chars=args.max_problem_chars,
                    max_completion_chars=args.max_completion_chars,
                    max_code_chars=args.max_code_chars,
                    max_retries=args.max_retries,
                    base_backoff=args.base_backoff,
                    max_backoff=args.max_backoff,
                )
            )
            for r in selected
        ]

        results: List[Dict[str, Any]] = []
        if tqdm is not None:
            bar = tqdm(total=len(tasks), desc="Gemini classify", unit="item")
            try:
                for fut in asyncio.as_completed(tasks):
                    results.append(await fut)
                    bar.update(1)
            finally:
                bar.close()
        else:
            done = 0
            for fut in asyncio.as_completed(tasks):
                results.append(await fut)
                done += 1
                if done % 25 == 0 or done == len(tasks):
                    print(f"Progress: {done}/{len(tasks)}")

    summary = summarize(results=results, total_rows=len(rows))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("results", "analysis")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"mistake_evolution_{timestamp}.json")
    payload = {
        "input_file": in_path,
        "generated_at": timestamp,
        "points": points,
        "max_problems": args.max_problems,
        "model": args.model,
        "selected_count": len(selected),
        "summary": summary,
        "records": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print("\n=== Summary ===")
    print(f"Saved: {out_path}")
    print(f"Classified: {summary['total_classified']} | API ok: {summary['api_ok_count']}")
    print("Overall mistake counts:")
    for k, v in sorted(summary["overall_mistake_counts"].items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {k}: {v}")
    comp = summary.get("completeness_vs_execution", {})
    print("Completeness vs execution:")
    print(
        "  "
        f"count={comp.get('count')} "
        f"mean_comp={comp.get('mean_completeness')} "
        f"mean_exec={comp.get('mean_execution')} "
        f"corr={comp.get('pearson_corr')} "
        f"mae={comp.get('mae_vs_execution')} "
        f"high_comp_low_exec_frac={comp.get('high_comp_low_exec_frac')}"
    )
    ce = summary.get("completeness_evolution", {})
    print(
        "Completeness evolution:"
        f" improved={ce.get('improved_count')},"
        f" worsened={ce.get('worsened_count')},"
        f" unchanged={ce.get('unchanged_count')},"
        f" mean_delta={ce.get('mean_delta')}"
    )
    evo = summary["evolution_summary"]
    print(
        "Evolution: "
        f"improved={evo['improved_count']}, worsened={evo['worsened_count']}, unchanged={evo['unchanged_count']}"
    )
    return 0


def main() -> None:
    args = parse_args()
    raise SystemExit(asyncio.run(run_async(args)))


if __name__ == "__main__":
    main()
