# W&B Report Template (Copy/Paste)

Use this as a starter template in a W&B Report.

---

## Learning

Track whether policy learning is improving and whether GRPO has useful contrast.

**Pin these charts**

- `reward/mean`
- `reward/execution_mean`
- `reward/reasoning_mean`
- `grpo/reward_std_mean`
- `grpo/all_zero_fraction`
- `grpo/all_perfect_fraction`
- `kl`

**How to read**

- `reward/mean` should trend up over time.
- `grpo/reward_std_mean` should stay clearly above zero.
- `grpo/all_zero_fraction` should stay low (too high means collapse).

---

## Quality

Track generation quality and judge reliability.

**Pin these charts**

- `gen/valid_code_fraction`
- `gen/has_reasoning_fraction`
- `gen/empty_completion_fraction`
- `exec/timeout_fraction`
- `exec/zero_fraction`
- `judge/fallback_fraction`
- `judge/step_json_fraction`

**How to read**

- `gen/valid_code_fraction` should remain high.
- `judge/fallback_fraction` should be near zero.
- `judge/step_json_fraction` should be near one.

---

## Timing

Track where step time is spent.

**Pin these charts**

- `timing/step_total_s`
- `timing/generation_s`
- `timing/reward_calc_s`
- `timing/reward_execution_s`
- `timing/reward_judge_s`
- `timing/loss_compute_s`

**How to read**

- `timing/step_total_s` is end-to-end optimizer step time.
- Use component charts to find bottlenecks (generation vs reward vs loss).

---

## System

Track machine pressure and GPU usage.

**Pin these charts**

- `gpu/utilization_percent`
- `gpu/nvml_mem_used_gb`
- `gpu/torch_reserved_gb`
- `gpu/non_torch_mem_gb_est`
- `system/cpu_percent`
- `system/ram_percent`

**How to read**

- `gpu/nvml_mem_used_gb` is total GPU memory in use.
- `gpu/non_torch_mem_gb_est` is a rough proxy for non-PyTorch usage (often vLLM/runtime).

---

## Events

Track save/push overhead and cadence.

**Pin these charts**

- `event/checkpointing`
- `event/checkpoint_save_s`
- `event/hf_push_s`

**How to read**

- `event/checkpointing` acts as a pulse marker when a save occurs.
- Duration charts show overhead added by saving/uploading.

---

## Optional Debug Section

If needed, add a separate section for low-level debug metrics:

- warning flags (`warn/`*, `grpo/warn_*`)
- per-source reward splits (`reward/apps_mean`, `reward/lcb_mean`)
- data coverage (`data/*`)

