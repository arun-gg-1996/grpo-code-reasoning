# Metrics Glossary (Simple English)

This file explains the custom metrics logged by this project to Weights & Biases.

## Operator Watchlist (Pin These First)

For the first 30-60 minutes of a training run, prioritize:

- `reward/execution_mean`
- `exec/zero_fraction`
- `exec/infra_zero_fraction`
- `exec/model_zero_fraction`
- `gen/valid_code_fraction`
- `grpo/reward_std_mean`
- `judge/fallback_fraction`
- `judge/retry_count`
- `judge/rate_limit_count`

Quick interpretation:
- High `exec/infra_zero_fraction` = format/runtime/infrastructure failures.
- High `exec/model_zero_fraction` = code runs but fails tests (capability issue).
- Near-zero `grpo/reward_std_mean` = weak GRPO contrast signal.
- High `judge/fallback_fraction` with rising retries/rate limits = unreliable judge signal.

## Learning (`reward/*`, `grpo/*`, `exec/*`, `judge/*`)

### `reward/*`
- `reward/mean`: average final reward for the current batch.
- `reward/std`: spread of final rewards in the batch.
- `reward/non_zero_fraction`: fraction of completions with reward > 0.
- `reward/execution_mean`: average execution score before reward weighting.
- `reward/reasoning_mean`: average reasoning score before reward weighting.
- `reward/mean_easy|medium|hard`: average final reward for each difficulty.
- `reward/execution_easy|medium|hard`: average execution score by difficulty.
- `reward/reasoning_easy|medium|hard`: average reasoning score by difficulty.
- `reward/apps_mean`: average final reward on APPS source rows.
- `reward/lcb_mean`: average final reward on LCB source rows.

### `grpo/*`
- `grpo/reward_std_mean`: average reward spread inside each GRPO group.
- `grpo/all_zero_fraction`: fraction of groups where all rollouts got 0 reward.
- `grpo/all_perfect_fraction`: fraction of groups where all rollouts are near perfect.

### `exec/*`
- `exec/mean_score`: average execution score.
- `exec/zero_scores`: count of completions with execution score 0.
- `exec/perfect_scores`: count of completions with execution score 1.
- `exec/ok_count`: count of executions that ran and returned test results.
- `exec/error_count`: count of executions that failed with runtime/checker errors.
- `exec/empty_count`: count with empty/missing code or missing test cases.
- `exec/timeout_count`: count of sandbox timeouts.
- `exec/zero_ok_count`: zero-score executions among successful runs (`ok_count`).
- `exec/ok_fraction`: ok_count / batch_size.
- `exec/error_fraction`: error_count / batch_size.
- `exec/empty_fraction`: empty_count / batch_size.
- `exec/timeout_fraction`: timeout ratio in batch.
- `exec/zero_fraction`: fraction of completions with execution score 0.
- `exec/infra_zero_fraction`: zeros caused by error/timeout/empty statuses.
- `exec/model_zero_fraction`: zeros from `ok` runs that passed 0 tests.
- `exec/nonzero_mean`: average execution score among non-zero completions only.
- `exec/apps_mean`: average execution score on APPS samples.
- `exec/lcb_mean`: average execution score on LCB samples.

### `judge/*`
- `judge/gemini_mean`: average Gemini judge score for called items.
- `judge/gemini_calls`: how many completions were sent to Gemini in this batch.
- `judge/total_calls`: total Gemini requests attempted for this batch.
- `judge/fallback_count`: how many requests used fallback score (0.5).
- `judge/fallback_fraction`: fallback_count / total_calls.
- `judge/step_json_count`: how many judge responses had valid per-step JSON.
- `judge/step_json_fraction`: step_json_count / total_calls.
- `judge/retry_count`: total retry attempts made in this batch.
- `judge/rate_limit_count`: number of Gemini 429 responses encountered.

## Generation Quality (`gen/*`, `presence/*`)

- `gen/valid_code_fraction`: fraction with extractable code block.
- `gen/has_reasoning_fraction`: fraction with `<think>` block.
- `gen/empty_completion_fraction`: fraction of empty outputs.
- `gen/mean_completion_chars`: average output length in characters.
- `gen/mean_think_chars`: average reasoning length in characters.
- `gen/mean_code_chars`: average code length in characters.
- `gen/likely_truncated_fraction`: rough fraction likely cut by max token cap.
- `presence/mean`: average heuristic reasoning-structure score.

## Data Coverage (`data/*`)

- `data/unique_problems_seen`: cumulative unique problems seen so far.
- `data/apps_seen`: cumulative unique APPS problems seen.
- `data/lcb_seen`: cumulative unique LCB problems seen.
- `data/easy_seen|medium_seen|hard_seen`: cumulative unique problems seen per difficulty.

## Timing (`timing/*`)

### Trainer-level timing
- `timing/step_total_s`: total wall time per optimizer step.
- `timing/generation_s`: rollout generation time.
- `timing/reward_calc_s`: total reward calculation time in trainer path.
- `timing/loss_compute_s`: loss compute time.
- `timing/training_step_s`: training step wrapper time.

### Reward-level timing
- `timing/reward_total_s`: full reward_fn time.
- `timing/reward_execution_s`: sandbox scoring time inside reward_fn.
- `timing/reward_judge_s`: Gemini judge time inside reward_fn.

## System + GPU (`system/*`, `gpu/*`)

### `system/*`
- `system/cpu_percent`: process CPU usage percent.
- `system/process_rss_gb`: process RAM usage (resident set size) in GB.
- `system/ram_used_gb`: machine total RAM currently used.
- `system/ram_percent`: machine RAM utilization percent.

### `gpu/*`
- `gpu/utilization_percent`: GPU compute utilization percent (NVML).
- `gpu/nvml_mem_used_gb`: total GPU memory in use (driver view).
- `gpu/nvml_mem_total_gb`: total GPU memory capacity.
- `gpu/torch_allocated_gb`: memory currently allocated by PyTorch.
- `gpu/torch_reserved_gb`: memory reserved by PyTorch allocator.
- `gpu/torch_max_allocated_gb`: peak PyTorch allocated memory.
- `gpu/torch_max_reserved_gb`: peak PyTorch reserved memory.
- `gpu/non_torch_mem_gb_est`: estimated non-PyTorch memory.
  Formula: `nvml_mem_used_gb - torch_reserved_gb` (rough proxy, often includes vLLM/runtime memory).
- `gpu/model_param_gb_est`: estimated model parameter memory size.
- `gpu/optimizer_state_gb_est`: estimated optimizer state memory size.

## Events (`event/*`)

- `event/checkpointing`: pulse metric (1) when checkpoint save happens.
- `event/checkpoint_save_s`: checkpoint save duration in seconds.
- `event/hf_push_s`: Hugging Face push duration in seconds.

## Warning Flags (`warn/*`, `grpo/warn_*`)

These are binary indicators (`1` fired, `0` clear):
- `grpo/warn_reward_std_collapse`
- `grpo/warn_all_zero_collapse`
- `warn/format_failure`
- `warn/exec_formatting_breakdown`
- `warn/exec_zero_sandbox`
- `warn/exec_low_nonzero`
- `warn/reasoning_collapse`
- `warn/gemini_silent_failure`
- `warn/problems_too_easy`
- `warn/empty_completions`
- `warn/high_timeout_rate`

If one flips to `1`, check console warning text for diagnosis and next action.
