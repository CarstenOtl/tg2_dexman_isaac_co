---
name: reward-breakdown
description: Parse all reward terms from a dextrah task env file and produce a summary table — weight, formula type, contact-gated or not, logged to extras or not, and current configured value. Use when the user asks about rewards, reward shaping, or wants to understand what the policy is being incentivized to do.
tools: Read, Grep
---

# Reward Breakdown

Parse and summarize all reward terms in a dextrah environment.

## Workflow

### Step 1: Find the task

Locate `dextrah_*_env.py` and `dextrah_*_env_cfg.py` for the relevant task.

### Step 2: Extract reward weights from cfg

In the env_cfg, find all reward weight fields (pattern: `*_weight`, `*_penalty_weight`, `*_scale`). Note their values.

### Step 3: Parse compute_rewards()

Find the `compute_rewards()` function (usually a `@torch.jit.script` decorated standalone function). For each reward term extract:
- **Name**
- **Formula type**: exponential (`torch.exp`), squared penalty, linear, binary/masked
- **Contact-gated**: does it multiply by `contact_mask`?
- **Sign**: positive reward or negative penalty
- **Clamped**: is there a `clamp()` on it?

### Step 4: Check extras logging

Search for `self.extras[...]` assignments in `_get_rewards()` or `compute_rewards()`. Note which terms are logged and which are silent.

### Step 5: Print summary table

| Term | Weight | Formula | Gated on contact? | Clamped? | Logged? |
|---|---|---|---|---|---|
| hand_to_object | 0.5 | exp(-sharpness * dist) | No | No | Yes |
| lift_reward | 2.0 | exp(sharp * height) - 1 | Yes | max=50 | Yes |
| action_rate_penalty | -0.008 | -w*(arm_delta² + 2.5*hand_delta²) | No | No | Yes |
| ... | | | | | |

Then add a short plain-English summary of what the reward structure is trying to achieve (approach → contact → lift → hold).

Flag any reward terms with weight=0 (disabled), or terms that appear in the function but aren't added to the total reward.
