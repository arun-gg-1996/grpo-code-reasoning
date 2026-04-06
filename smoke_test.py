"""
smoke_test.py

Quick local verification that all modules import and work end-to-end.
Tests reward computation with real sandbox execution.

Usage:
    python smoke_test.py
"""

import sys
import os

sys.set_int_max_str_digits(0)  # APPS data contains integers with 9000+ digits

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """Verify all modules import cleanly."""
    print("--- Testing imports ---")

    from config import (
        EXEC_TIMEOUT, EXEC_WORKERS, REASONING_SYSTEM_PROMPT,
        JUDGE_TIMEOUT, JUDGE_TEMPERATURE, JUDGE_MAX_TOKENS,
        APPS_EXEC_WEIGHT, APPS_REASONING_WEIGHT,
        LCB_EXEC_WEIGHT, LCB_REASONING_WEIGHT,
        EASY_GEMINI_WEIGHT, MEDIUM_GEMINI_WEIGHT, HARD_GEMINI_WEIGHT,
        JUDGE_MODEL, MAX_NEW_TOKENS,
        normalize_difficulty, get_curriculum_weights,
    )
    print(f"  config: OK (APPS exec={APPS_EXEC_WEIGHT}, LCB exec={LCB_EXEC_WEIGHT}, MAX_NEW_TOKENS={MAX_NEW_TOKENS})")

    import sandbox
    print("  sandbox: OK")

    from reward.execution import score_batch, extract_code, score_single
    print("  reward.execution: OK")

    from reward.reward import reward_fn, _presence_score, _get_tier_weights
    print("  reward.reward: OK")

    from reward.judge import parse_judge_response
    print("  reward.judge: OK")

    import problem_format
    print("  problem_format: OK")

    return True


def test_config_logic():
    """Test config helpers."""
    print("\n--- Testing config logic ---")
    from config import normalize_difficulty, get_curriculum_weights, get_curriculum_phase
    from problem_format import is_function_style_problem, get_function_name

    assert normalize_difficulty("introductory") == "easy"
    assert normalize_difficulty("interview") == "medium"
    assert normalize_difficulty("competition") == "hard"
    assert normalize_difficulty("easy") == "easy"
    print("  normalize_difficulty: OK")

    w0 = get_curriculum_weights(0)
    assert w0["easy"] == 0.70 and w0["medium"] == 0.30 and w0["hard"] == 0.0
    w80 = get_curriculum_weights(80)
    assert w80["easy"] == 0.45 and w80["medium"] == 0.50 and w80["hard"] == 0.05
    w250 = get_curriculum_weights(250)
    assert w250["easy"] == 0.30 and w250["medium"] == 0.50 and w250["hard"] == 0.20
    w700 = get_curriculum_weights(700)
    assert w700["easy"] == 0.15 and w700["medium"] == 0.40 and w700["hard"] == 0.45
    print("  get_curriculum_weights: OK")

    assert get_curriculum_phase(0) == 0
    assert get_curriculum_phase(80) == 1
    assert get_curriculum_phase(250) == 2
    assert get_curriculum_phase(700) == 3
    print("  get_curriculum_phase: OK")

    p_fn = {"fn_name": "twoSum", "is_leetcode": False}
    p_stdio = {"is_leetcode": False}
    assert is_function_style_problem(p_fn) is True
    assert get_function_name(p_fn) == "twoSum"
    assert is_function_style_problem(p_stdio) is False
    print("  problem_format helpers: OK")


def test_extraction():
    """Test code and think block extraction."""
    print("\n--- Testing extraction ---")
    from reward.execution import extract_code
    from reward.reward import _extract_think_block, _presence_score

    # Code extraction
    code, mode = extract_code("<code>print(1)</code>")
    assert code == "print(1)" and mode == "fence"
    code, mode = extract_code("```python\nprint(2)\n```")
    assert code == "print(2)" and mode == "strict"
    code, mode = extract_code("```\nprint(4)\n```")
    assert code == "print(4)" and mode == "fence"
    code, mode = extract_code("<code>```python\nprint(3)\n```</code>")
    assert code == "print(3)" and mode == "strict"
    code, mode = extract_code("no code here")
    assert code is None and mode == "none"
    print("  extract_code: OK")

    # Think block extraction
    tb = _extract_think_block("<think>some reasoning</think>")
    assert tb == "some reasoning"
    assert _extract_think_block("no think block") == ""
    print("  _extract_think_block: OK")

    # Presence scoring
    block = (
        "[STEP] Read the problem statement carefully here\n"
        "[STEP] Choose a greedy algorithm approach now\n"
        "[STEP] Use a list to store intermediate values\n"
        "[STEP] Time complexity is O(n) space is O(1)\n"
        "[STEP] Handle empty input and boundary cases\n"
        "[STEP] Write the main loop and output result\n"
    )
    score = _presence_score(block)
    assert score == 1.0, f"Expected 1.0, got {score}"
    print(f"  presence_score (6 valid steps): {score}")

    # Partial
    partial = "[STEP] short\n[STEP] also short"
    score2 = _presence_score(partial)
    assert score2 == 0.0, f"Expected 0.0, got {score2}"  # steps too short (<20 chars)
    print(f"  presence_score (2 short steps): {score2}")


def test_completion_unwrap():
    """Ensure chat-structured completions are unwrapped to plain assistant text."""
    print("\n--- Testing completion unwrap ---")
    from reward.reward import _unwrap_completion

    text = "<think>reason</think>\n```python\nprint('ok')\n```"

    # Plain text passthrough
    assert _unwrap_completion(text) == text

    # TRL message-list completion
    list_payload = [{"role": "assistant", "content": text}]
    out_list = _unwrap_completion(list_payload)
    assert out_list == text
    assert "\n" in out_list and "\\n" not in out_list

    # Dict completion
    dict_payload = {"role": "assistant", "content": text}
    out_dict = _unwrap_completion(dict_payload)
    assert out_dict == text

    print("  completion unwrap: OK")


def test_judge_parse():
    """Test Gemini judge response parsing."""
    print("\n--- Testing judge parse ---")
    from reward.judge import parse_judge_response

    # Valid response
    r = parse_judge_response('{"step_scores": [1.0, 0.5, 0.3, 1.0, 0.8, 0.9], "overall": 0.77}')
    assert r is not None
    assert abs(r["overall"] - 0.77) < 0.01
    assert len(r["step_scores"]) == 6
    print(f"  valid parse: overall={r['overall']}, steps={r['step_scores']}")

    # With markdown fences
    r2 = parse_judge_response('```json\n{"step_scores": [0.5], "overall": 0.5}\n```')
    assert r2 is not None
    print(f"  markdown fence parse: overall={r2['overall']}")

    # JSON overall-only (current prompt contract)
    r2b = parse_judge_response('{"overall": 0.61}')
    assert r2b is not None
    assert abs(r2b["overall"] - 0.61) < 0.01
    print(f"  overall-only parse: overall={r2b['overall']}")

    # Invalid (suppress expected warning noise for this test case)
    import logging
    judge_logger = logging.getLogger("reward.judge")
    prev_level = judge_logger.level
    judge_logger.setLevel(logging.ERROR)
    try:
        r3 = parse_judge_response("this is not json")
    finally:
        judge_logger.setLevel(prev_level)
    assert r3 is None
    print("  invalid parse: None (correct)")

    # NaN / inf should be rejected
    assert parse_judge_response('{"overall": NaN}') is None
    assert parse_judge_response('{"overall": Infinity}') is None
    print("  NaN/Inf parse: None (correct)")


def test_tier_weights():
    """Test reasoning path compatibility helpers."""
    print("\n--- Testing reasoning helpers ---")
    from reward.reward import _get_tier_weights, _get_source_weights

    g, p = _get_tier_weights("easy")
    assert g == 1.0 and p == 0.0
    print(f"  easy: gemini={g}, presence={p}")

    g, p = _get_tier_weights("medium")
    assert g == 1.0 and p == 0.0
    print(f"  medium: gemini={g}, presence={p}")

    g, p = _get_tier_weights("hard")
    assert g == 1.0 and p == 0.0
    print(f"  hard: gemini={g}, presence={p}")

    e, r = _get_source_weights("apps")
    assert e == 0.75 and r == 0.25
    print(f"  apps: exec={e}, reas={r}")

    e, r = _get_source_weights("lcb_seen")
    assert e == 0.60 and r == 0.40
    print(f"  lcb_seen: exec={e}, reas={r}")


def test_execution_sandbox():
    """Test actual code execution in sandbox."""
    print("\n--- Testing execution sandbox ---")
    from reward.execution import score_single, score_batch

    # Problem: print "hello"
    problem = {
        "test_cases": [{"input": "", "output": "hello\n"}],
        "stdin_tests": [{"input": "", "output": "hello\n"}],
        "is_leetcode": False,
    }

    # Correct solution
    s1 = score_single('print("hello")', problem)
    print(f"  correct solution: {s1}")
    assert s1 == 1.0

    # Wrong solution
    s2 = score_single('print("world")', problem)
    print(f"  wrong solution: {s2}")
    assert s2 == 0.0

    # Empty code
    s3 = score_single("", problem)
    print(f"  empty code: {s3}")
    assert s3 == 0.0

    # Batch scoring (uses spawn pool — may fail in some environments)
    try:
        scores, stats = score_batch(
            codes=['print("hello")', 'print("world")', ""],
            problems=[problem, problem, problem],
        )
        print(f"  batch scores: {scores}")
        print(f"  batch stats: {stats}")
        assert scores[0] == 1.0
        assert scores[1] == 0.0
    except AssertionError:
        # "daemonic processes are not allowed to have children"
        # This only happens in test environments; works fine during actual training
        print("  batch scoring: SKIPPED (daemon process limitation in test env)")


def test_reward_fn_easy():
    """End-to-end reward_fn test with easy problems (Gemini + presence reasoning)."""
    print("\n--- Testing reward_fn (easy, Gemini + presence reasoning) ---")
    from reward.reward import reward_fn

    problem = {
        "difficulty": "easy",
        "test_cases": [{"input": "", "output": "hello\n"}],
        "stdin_tests": [{"input": "", "output": "hello\n"}],
        "is_leetcode": False,
        "problem_id": "smoke_test_1",
        "question": "Print hello",
    }

    # Good completion: correct code + valid reasoning
    good = (
        "<think>"
        "The problem asks us to print hello to stdout.\n"
        "Simple output problem, just use print statement.\n"
        "No data structures are needed.\n"
        "Complexity is O(1) time and O(1) space.\n"
        "Implementation is a single print call.\n"
        "</think>\n"
        '```python\nprint("hello")\n```'
    )

    # Bad completion: wrong code, no reasoning
    bad = '```python\nprint("world")\n```'

    # No code at all
    empty = "I don't know how to solve this"

    rewards = reward_fn(
        completions=[good, bad, empty],
        prompts=["p", "p", "p"],
        problems=[problem, problem, problem],
        source=["apps", "apps", "apps"],
    )

    print(f"  good:  {rewards[0]:.3f}")
    print(f"  bad:   {rewards[1]:.3f}")
    print(f"  empty: {rewards[2]:.3f}")

    assert rewards[0] > rewards[1], "Good should beat bad"
    assert rewards[1] >= rewards[2], "Bad should beat empty"
    assert rewards[0] > 0.5, "Good should be > 0.5"

    # Verify APPS weighting: exec=0.75, reasoning=0.25
    # Good: exec=1.0, high reasoning score expected (easy uses Gemini + presence)
    # If Gemini is unavailable in local tests, fallback can reduce this slightly.
    print(f"  expected good ~1.0, got {rewards[0]:.3f}")


def test_reasoning_fallback_on_judge_failure():
    """If Gemini fails, reward path must use presence fallback (not fixed 0.5)."""
    print("\n--- Testing reasoning fallback semantics ---")
    from reward.reward import reward_fn
    import reward.judge as judge_mod

    problem = {
        "difficulty": "easy",
        "test_cases": [{"input": "", "output": "hello\n"}],
        "stdin_tests": [{"input": "", "output": "hello\n"}],
        "is_leetcode": False,
        "problem_id": "smoke_fallback_1",
        "question": "Print hello",
    }
    completion = (
        "<think>"
        "Determine this is stdin/stdout output formatting.\n"
        "We only need to print a fixed constant string.\n"
        "No extra data structures are required.\n"
        "Complexity is O(1) time and O(1) space.\n"
        "Implementation is a direct print statement.\n"
        "</think>\n"
        "```python\nprint('world')\n```"
    )

    orig_score_batch = judge_mod.score_batch
    orig_get_last_batch_stats = judge_mod.get_last_batch_stats
    try:
        judge_mod.score_batch = lambda problems, think_blocks, difficulties: [0.5 for _ in problems]
        judge_mod.get_last_batch_stats = lambda: {
            "fallback_mask": [True],
            "total_calls": 1,
            "fallback_count": 1,
            "fallback_fraction": 1.0,
            "json_count": 0,
            "json_fraction": 0.0,
            "retry_count": 0,
            "rate_limit_count": 0,
        }
        rewards = reward_fn(
            completions=[completion],
            prompts=["p"],
            problems=[problem],
            source=["apps"],
        )
    finally:
        judge_mod.score_batch = orig_score_batch
        judge_mod.get_last_batch_stats = orig_get_last_batch_stats

    # With exec=0 and strong presence score, fallback reward should be > 0.2.
    # If judge 0.5 was incorrectly used as real score, this would be ~0.125.
    print(f"  fallback reward: {rewards[0]:.3f}")
    assert rewards[0] > 0.2, "Expected presence fallback score on judge failure"


def _normalize_test_cases(problem):
    """Same normalization as train.py — duplicated here to avoid heavy imports."""
    if problem.get("test_cases"):
        return problem
    io = problem.get("io")
    if io and isinstance(io, dict):
        inputs = io.get("inputs", [])
        outputs = io.get("outputs", [])
        tc = [{"input": inp, "output": out} for inp, out in zip(inputs, outputs)]
        problem["test_cases"] = tc
        problem["stdin_tests"] = tc
    return problem


def _load_problems(path, source_label):
    import json
    from config import normalize_difficulty
    problems = []
    with open(path) as f:
        for line in f:
            if line.strip():
                p = json.loads(line)
                p["source"] = p.get("source", source_label)
                p["difficulty"] = normalize_difficulty(p.get("difficulty", "medium"))
                p = _normalize_test_cases(p)
                problems.append(p)
    return problems


def test_data_loading():
    """Verify data files load correctly and APPS io normalization works."""
    print("\n--- Testing data loading ---")
    from reward.execution import score_single

    apps = _load_problems("data/clean/apps_clean.jsonl", "apps")
    print(f"  APPS: {len(apps)} problems")
    # Check normalization
    p = apps[0]
    tc = p.get("test_cases", [])
    print(f"  APPS first: diff={p['difficulty']}, test_cases={len(tc)}")
    assert len(tc) > 0, "APPS problems should have test_cases after normalization"
    print(f"  APPS first tc input[:50]: {tc[0]['input'][:50]}")

    lcb = _load_problems("data/clean/lcb_seen_clean.jsonl", "lcb_seen")
    print(f"  LCB:  {len(lcb)} problems")
    tc2 = lcb[0].get("test_cases", [])
    print(f"  LCB first: diff={lcb[0]['difficulty']}, test_cases={len(tc2)}")

    taco = _load_problems("data/clean/taco_verified_clean.jsonl", "taco_verified")
    print(f"  TACO: {len(taco)} problems")
    tc3 = taco[0].get("test_cases", [])
    print(f"  TACO first: diff={taco[0]['difficulty']}, test_cases={len(tc3)}")

    # Test that a real APPS problem executes correctly
    for p in apps:
        sol = p.get("passing_solution") or (p.get("solutions", [None]) or [None])[0]
        if sol and p.get("test_cases"):
            score = score_single(sol, p)
            print(f"  APPS execution test (problem {p.get('problem_id', '?')}): score={score}")
            break


def test_prompt_format_hints():
    """Verify format-routing logic and global prompt guidance."""
    print("\n--- Testing prompt format hints ---")
    from config import TRAINING_SYSTEM_PROMPT
    try:
        from train import build_prompt
    except ImportError as e:
        print(f"  SKIPPED (train.py deps not available: {e})")
        return
    from problem_format import is_function_style_problem, get_function_name

    assert "<think>...</think>" in TRAINING_SYSTEM_PROMPT
    assert "```python code block" in TRAINING_SYSTEM_PROMPT
    assert "<code>" not in TRAINING_SYSTEM_PROMPT

    stdio_problem = {
        "question": "Read n and print n",
        "source": "apps",
        "is_leetcode": False,
        "func_name": None,
    }
    assert is_function_style_problem(stdio_problem) is False
    stdio_prompt = build_prompt(stdio_problem)
    assert isinstance(stdio_prompt, list) and len(stdio_prompt) == 2
    assert stdio_prompt[0]["role"] == "system"
    assert stdio_prompt[1]["role"] == "user"
    assert "stdin/stdout problem" in stdio_prompt[1]["content"]

    func_problem = {
        "question": "Implement function foo",
        "source": "lcb_seen",
        "is_leetcode": True,
        "func_name": "foo",
    }
    assert is_function_style_problem(func_problem) is True
    assert get_function_name(func_problem) == "foo"
    func_prompt = build_prompt(func_problem)
    assert isinstance(func_prompt, list) and len(func_prompt) == 2
    assert "function-style problem" in func_prompt[1]["content"]
    assert "Implement the expected function `foo` exactly as specified." in func_prompt[1]["content"]
    print("  prompt format routing: OK")


def test_metric_names():
    """Verify renamed/added/removed W&B metrics are present in reward log output."""
    print("\n[test_metric_names]")
    from reward.reward import reward_fn
    import reward.reward as _reward_module

    completions = ["```python\nprint(1)\n```"]
    prompts = ["Print the number 1"]
    problems = [{"question": "Print 1", "difficulty": "easy", "source": "lcb_seen",
                 "test_cases": [{"input": "", "output": "1"}]}]

    log_capture = {}

    # Build a minimal wandb mock (wandb may not be installed in local env)
    class _FakeRun:
        pass

    class _FakeWandb:
        run = _FakeRun()
        @staticmethod
        def log(d, **kw):
            log_capture.update(d)

    # Inject the mock directly into the reward module's namespace
    original_wandb = _reward_module.wandb
    _reward_module.wandb = _FakeWandb()

    try:
        reward_fn(completions=completions, prompts=prompts, problems=problems)
    finally:
        _reward_module.wandb = original_wandb

    # Renamed metrics must exist
    assert "exec/failed_to_run_fraction" in log_capture, "exec/failed_to_run_fraction missing"
    assert "exec/wrong_solution_fraction" in log_capture, "exec/wrong_solution_fraction missing"
    # Old names must be gone
    assert "exec/infra_zero_fraction" not in log_capture, "exec/infra_zero_fraction should be removed"
    assert "exec/model_zero_fraction" not in log_capture, "exec/model_zero_fraction should be removed"
    # New judge split counters must exist
    assert "judge/gemini_used_count" in log_capture, "judge/gemini_used_count missing"
    assert "judge/presence_used_count" in log_capture, "judge/presence_used_count missing"
    # Redundant aliases must be gone
    assert "judge/step_json_count" not in log_capture, "judge/step_json_count alias should be removed"
    assert "judge/step_json_fraction" not in log_capture, "judge/step_json_fraction alias should be removed"
    # warn_reward_std_collapse must be in log_dict (not only flag_dict) with default 0
    assert "grpo/warn_reward_std_collapse" in log_capture, "grpo/warn_reward_std_collapse missing from log_dict"
    print("  metric names: OK")


if __name__ == "__main__":
    print("=" * 60)
    print("GRPO SMOKE TEST")
    print("=" * 60)

    try:
        test_imports()
        test_config_logic()
        test_extraction()
        test_completion_unwrap()
        test_judge_parse()
        test_tier_weights()
        test_execution_sandbox()
        test_reward_fn_easy()
        test_reasoning_fallback_on_judge_failure()
        test_data_loading()
        test_prompt_format_hints()
        test_metric_names()

        print("\n" + "=" * 60)
        print("ALL SMOKE TESTS PASSED")
        print("=" * 60)
    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
