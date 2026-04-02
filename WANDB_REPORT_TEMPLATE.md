# W&B Report Template (GRPO)

Use this when creating a run report for `grpo-code-gen`.

## Section 1: Learning Signal

Pin:
- `reward/mean`
- `reward/execution_mean`
- `reward/reasoning_mean`
- `grpo/reward_std_mean`
- `grpo/all_zero_fraction`
- `grpo/all_perfect_fraction`
- `kl` (or `train/kl` depending TRL/W&B naming)

Read:
- `grpo/reward_std_mean` near 0 means weak GRPO contrast.
- high `grpo/all_zero_fraction` means collapse on current distribution.
- if `kl` is missing, try pinning `train/kl` or `objective/kl`.

## Section 2: Execution Health

Pin:
- `exec/zero_fraction`
- `exec/infra_zero_fraction`
- `exec/model_zero_fraction`
- `exec/timeout_fraction`
- `exec/ok_fraction`
- `exec/error_fraction`
- `exec/empty_fraction`
- `exec/nonzero_mean`

Read:
- high `infra_zero_fraction` points to format/runtime/infrastructure issues.
- high `model_zero_fraction` means code executes but fails logic tests.
- low `nonzero_mean` means capability gap on solved-format outputs.

## Section 3: Generation Quality

Pin:
- `gen/valid_code_fraction`
- `gen/has_reasoning_fraction`
- `gen/empty_completion_fraction`
- `gen/mean_think_chars`
- `gen/mean_code_chars`
- `gen/likely_truncated_fraction`

Read:
- low `valid_code_fraction` usually indicates formatting drift.
- high `likely_truncated_fraction` means increase `max_new_tokens`.

## Section 4: Gemini Reliability

Pin:
- `judge/fallback_fraction`
- `judge/step_json_fraction`
- `judge/retry_count`
- `judge/rate_limit_count`
- `timing/reward_judge_s`

Read:
- rising `rate_limit_count` + `retry_count` indicates quota/concurrency pressure.
- high `fallback_fraction` indicates degraded judge signal quality.

## Section 5: Throughput and System

Pin:
- `timing/step_total_s`
- `timing/generation_s`
- `timing/reward_execution_s`
- `timing/loss_compute_s`
- `gpu/utilization_percent`
- `gpu/nvml_mem_used_gb`
- `gpu/non_torch_mem_gb_est`
- `system/ram_percent`

Read:
- use timing breakdown to locate bottleneck before retuning.

## Section 6: Checkpoints and Pushes

Pin:
- `event/checkpointing`
- `event/checkpoint_save_s`
- `event/hf_push_s`

Read:
- spikes here are expected at save/push boundaries.
- `event/hf_push_s` appears only when push-to-hub is enabled and a push happens.

## Section 7: Warning Flags

Pin all warning flags:
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

Read:
- investigate immediately when any warning flag stays at `1` for multiple steps.
