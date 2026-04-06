#!/usr/bin/env python3
"""
Probe safe Gemini concurrency using API-key auth against Vertex generateContent.

This script is designed for practical tuning of GEMINI_MAX_WORKERS.
It sends short requests at increasing worker counts and reports 2xx/429/error rates
plus latency stats.

Example:
  export GEMINI_API_KEY=...
  python scripts/probe_gemini_concurrency.py \
    --workers 1,2,4,6,8,10,12,16 \
    --calls-per-level 48 \
    --model gemini-2.5-flash-lite
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import random
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


DEFAULT_WORKERS = [1, 2, 4, 6, 8, 10, 12, 16]
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def _load_dotenv_vars(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip("'").strip('"')
        if k:
            out[k] = v
    return out


def _resolve_api_key(env_name: str, explicit_key: str) -> tuple[str, str]:
    if explicit_key:
        return explicit_key.strip(), "--api-key"

    val = os.getenv(env_name, "").strip()
    if val:
        return val, f"env:{env_name}"

    val = os.getenv("GOOGLE_API_KEY", "").strip()
    if val:
        return val, "env:GOOGLE_API_KEY"
    val = os.getenv("gemini_api_key", "").strip()
    if val:
        return val, "env:gemini_api_key"

    cwd_env = Path.cwd() / ".env"
    repo_env = Path(__file__).resolve().parents[1] / ".env"
    for p in (cwd_env, repo_env):
        vars_map = _load_dotenv_vars(p)
        for key in (env_name, "GOOGLE_API_KEY", "gemini_api_key"):
            cand = vars_map.get(key, "").strip()
            if cand:
                return cand, f"dotenv:{p}:{key}"

    return "", ""


def _build_body(max_output_tokens: int, temperature: float) -> dict[str, Any]:
    # Keep payload intentionally small and deterministic to reduce cost/noise.
    return {
        "system_instruction": {"parts": [{"text": "Return compact JSON only."}]},
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            'Return ONLY: {"overall": 0.5}. '
                            "No markdown, no extra keys."
                        )
                    }
                ],
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        },
    }


def _build_judge_like_body(
    problem: str,
    think_block: str,
    difficulty: str,
    max_output_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    user_prompt = (
        "Evaluate this reasoning trace for a competitive programming problem.\n"
        "Evaluate all [STEP] blocks present (variable number of steps).\n"
        "Do not reward verbosity. Penalize repetition, filler, and redundant restatement.\n"
        "Reward correctness and useful technical progress only.\n"
        "Return ONLY valid JSON with this exact schema:\n"
        '{"overall": o}\n'
        "Rules:\n"
        "- overall must be a float in [0.0, 1.0]\n"
        "- no markdown, no prose, no extra keys\n\n"
        f"Difficulty: {difficulty}\n\n"
        "Problem:\n"
        f"{problem}\n\n"
        "Reasoning trace:\n"
        f"{think_block}"
    )
    return {
        "system_instruction": {
            "parts": [
                {
                    "text": (
                        "You are a strict competitive-programming reasoning judge. "
                        "Score reasoning quality only."
                    )
                }
            ]
        },
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        },
    }


def _default_train_details_path() -> str:
    paths = sorted(glob.glob("results/train/*/train_details.jsonl"))
    return paths[-1] if paths else ""


def _load_train_payload_bodies(
    path: str,
    sample_rows: int,
    seed: int,
    min_think_chars: int,
    max_output_tokens: int,
    temperature: float,
) -> list[dict[str, Any]]:
    if not path:
        return []
    rows = []
    with open(path) as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            completion = obj.get("completion_text", "")
            problem = obj.get("problem_statement", "")
            difficulty = obj.get("difficulty", "medium")
            if not completion or not problem:
                continue
            m = THINK_RE.search(completion)
            if not m:
                continue
            think = m.group(1).strip()
            if len(think) < min_think_chars:
                continue
            rows.append((problem, think, str(difficulty)))
    if not rows:
        return []
    random.seed(seed)
    random.shuffle(rows)
    rows = rows[: max(1, sample_rows)]
    return [
        _build_judge_like_body(
            problem=p,
            think_block=t,
            difficulty=d,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
        for p, t, d in rows
    ]


@dataclass
class CallResult:
    status_code: int
    latency_s: float
    error: str = ""


async def _single_call(
    client: httpx.AsyncClient,
    url: str,
    body: dict[str, Any],
    timeout_s: float,
) -> CallResult:
    t0 = time.perf_counter()
    try:
        resp = await client.post(url, json=body, timeout=timeout_s)
        latency = time.perf_counter() - t0
        return CallResult(status_code=resp.status_code, latency_s=latency)
    except Exception as e:
        latency = time.perf_counter() - t0
        return CallResult(status_code=0, latency_s=latency, error=str(e))


async def _run_level(
    workers: int,
    n_calls: int,
    url: str,
    payload_bodies: list[dict[str, Any]],
    timeout_s: float,
) -> tuple[list[CallResult], float]:
    sem = asyncio.Semaphore(max(1, workers))
    results: list[CallResult] = []

    async with httpx.AsyncClient() as client:
        async def _guarded(i: int) -> None:
            async with sem:
                idx = i % len(payload_bodies)
                body = payload_bodies[idx]
                results.append(await _single_call(client, url, body, timeout_s))

        t0 = time.perf_counter()
        await asyncio.gather(*[_guarded(i) for i in range(n_calls)])
        elapsed = time.perf_counter() - t0
    return results, elapsed


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = min(len(vals) - 1, max(0, int((len(vals) - 1) * q)))
    return vals[idx]


def _parse_workers(raw: str) -> list[int]:
    out = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out or DEFAULT_WORKERS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key-env", default="GEMINI_API_KEY")
    ap.add_argument("--api-key", default="", help="Optional direct API key override")
    ap.add_argument("--model", default="gemini-2.5-flash-lite")
    ap.add_argument(
        "--payload-mode",
        default="synthetic",
        choices=["synthetic", "train-details"],
        help="synthetic: tiny fixed payload, train-details: replay realistic judge payloads",
    )
    ap.add_argument(
        "--train-details",
        default="",
        help="Path to train_details.jsonl (defaults to latest under results/train/*/train_details.jsonl)",
    )
    ap.add_argument("--sample-rows", type=int, default=128, help="Rows to sample in train-details mode")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-think-chars", type=int, default=80)
    ap.add_argument(
        "--workers",
        default=",".join(str(x) for x in DEFAULT_WORKERS),
        help="Comma-separated worker sweep, e.g. 1,2,4,6,8,10",
    )
    ap.add_argument("--calls-per-level", type=int, default=48)
    ap.add_argument("--timeout-s", type=float, default=20.0)
    ap.add_argument("--max-output-tokens", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--sleep-between-level-s", type=float, default=2.0)
    ap.add_argument("--json-out", default="")
    ap.add_argument(
        "--max-429-rate",
        type=float,
        default=0.01,
        help="Recommendation rule: acceptable 429 fraction per level",
    )
    ap.add_argument(
        "--max-error-rate",
        type=float,
        default=0.02,
        help="Recommendation rule: acceptable non-HTTP transport error fraction",
    )
    args = ap.parse_args()

    api_key, key_source = _resolve_api_key(args.api_key_env, args.api_key)
    if not api_key:
        raise SystemExit(
            "Missing API key. Checked --api-key, "
            f"env:{args.api_key_env}, env:GOOGLE_API_KEY, and local .env files."
        )

    url = (
        "https://aiplatform.googleapis.com/v1/publishers/google/models/"
        f"{args.model}:generateContent?key={api_key}"
    )
    payload_bodies: list[dict[str, Any]]
    if args.payload_mode == "synthetic":
        payload_bodies = [_build_body(args.max_output_tokens, args.temperature)]
    else:
        details_path = args.train_details or _default_train_details_path()
        payload_bodies = _load_train_payload_bodies(
            path=details_path,
            sample_rows=args.sample_rows,
            seed=args.seed,
            min_think_chars=args.min_think_chars,
            max_output_tokens=args.max_output_tokens,
            temperature=args.temperature,
        )
        if not payload_bodies:
            raise SystemExit(
                f"No usable payloads found from train details: {details_path or '<none>'}"
            )
        print(f"train_details_path={details_path}")
        print(f"sampled_payloads={len(payload_bodies)}")
    workers_list = _parse_workers(args.workers)

    print("=" * 80)
    print("Gemini Concurrency Probe")
    print("=" * 80)
    print(f"model={args.model}")
    print(f"workers={workers_list}")
    print(f"calls_per_level={args.calls_per_level}")
    print(f"timeout_s={args.timeout_s}")
    print(f"api_key_source={key_source}")

    all_rows: list[dict[str, Any]] = []
    recommended = 1

    for i, w in enumerate(workers_list, start=1):
        print(f"\n[{i}/{len(workers_list)}] workers={w} ...", flush=True)
        results, elapsed = asyncio.run(
            _run_level(
                workers=w,
                n_calls=args.calls_per_level,
                url=url,
                payload_bodies=payload_bodies,
                timeout_s=args.timeout_s,
            )
        )

        codes = [r.status_code for r in results]
        lat = [r.latency_s for r in results]
        ok = sum(1 for c in codes if 200 <= c < 300)
        n429 = sum(1 for c in codes if c == 429)
        http_err = sum(1 for c in codes if c not in {0, 429} and not (200 <= c < 300))
        transport_err = sum(1 for c in codes if c == 0)
        total = len(results)
        rpm = (total / elapsed) * 60.0 if elapsed > 0 else 0.0

        row = {
            "workers": w,
            "total": total,
            "ok": ok,
            "ok_rate": ok / total if total else 0.0,
            "http_429": n429,
            "rate_429": n429 / total if total else 0.0,
            "http_other_err": http_err,
            "rate_http_other_err": http_err / total if total else 0.0,
            "transport_err": transport_err,
            "rate_transport_err": transport_err / total if total else 0.0,
            "elapsed_s": elapsed,
            "rpm_observed": rpm,
            "latency_p50_s": _pct(lat, 0.5),
            "latency_p95_s": _pct(lat, 0.95),
            "latency_mean_s": statistics.fmean(lat) if lat else 0.0,
        }
        all_rows.append(row)

        healthy = (
            row["rate_429"] <= args.max_429_rate
            and row["rate_transport_err"] <= args.max_error_rate
        )
        if healthy:
            recommended = w

        print(
            "  ok={ok}/{total} ({ok_rate:.1%}) 429={http_429} ({rate_429:.1%}) "
            "transport_err={transport_err} ({rate_transport_err:.1%}) "
            "p50={latency_p50_s:.2f}s p95={latency_p95_s:.2f}s rpm={rpm_observed:.1f}".format(**row)
        )

        if i < len(workers_list):
            time.sleep(max(0.0, args.sleep_between_level_s))

    print("\n" + "-" * 80)
    print(f"Recommended GEMINI_MAX_WORKERS: {recommended}")
    print(
        f"(rule: 429_rate <= {args.max_429_rate:.1%}, "
        f"transport_error_rate <= {args.max_error_rate:.1%})"
    )

    if args.json_out:
        payload = {
            "model": args.model,
            "calls_per_level": args.calls_per_level,
            "workers": workers_list,
            "results": all_rows,
            "recommended_workers": recommended,
        }
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved JSON report: {args.json_out}")


if __name__ == "__main__":
    main()
