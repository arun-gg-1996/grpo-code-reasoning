# Extra Dataset Option: CodeContests

## What it is
DeepMind's CodeContests dataset (~10k competitive programming problems from Codeforces, AtCoder, CodeChef).
HuggingFace: `deepmind/code_contests`

## Why we excluded it initially
Average 2.3 test cases across all difficulties — too noisy for reliable execution reward.

## Why it's worth revisiting
With our per-source reward weighting design, we can include it with upweighted reasoning:
```
CodeContests: 0.55 exec + 0.35 reasoning + 0.10 similarity
```
Same treatment as LCB with reference solution. Compensates for noisy execution with stronger reasoning signal.

## What it adds
~5k additional problems, skewing toward medium/hard difficulty.
Directly addresses the main weakness: only ~500-600 hard problems in current dataset.

## What it needs before inclusion
- Same sandbox compat check as LCB (lcb_sandbox_compat_check.py pattern)
- Claude API solution enrichment for similarity score reference
- Claude API reference reasoning generation

## Estimated cost to enrich
~$10-15 for 5k problems via Claude Haiku.

## Decision
Not blocking current training run. Add if initial results show hard problem performance is weak.