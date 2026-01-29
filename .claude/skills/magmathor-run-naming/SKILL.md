---
name: magmathor-run-naming
description: Interpret and decode Magmathor experiment run names. Use this when asked to understand what a run name means, decode experiment configurations, or troubleshoot configuration-related failures.
---

# Magmathor Run Name Interpretation

Decode experiment run names to understand their configuration and troubleshoot issues.

## When to Use This Skill

- User asks what a run name means
- User wants to decode experiment configuration from a run name
- User encounters configuration validation errors
- User wants to understand the naming convention for experiments

## Run Name Format

Run names follow this pattern:

```
{model}--{config_suffix}--rep{N}--{test_suite}
```

### Components

| Component | Description | Example |
|-----------|-------------|---------|
| `{model}` | Model identifier | `gpt-4o`, `Magmar_Grounded_SFT_C` |
| `{config_suffix}` | Encoded experiment settings | `T1_Fs_H60_C1_P2_I0_R0_S1_E60_M4096` |
| `rep{N}` | Repetition number | `rep1`, `rep2` |
| `{test_suite}` | Test suite name | `magt_benchmark_p1` |

## Config Suffix Encoding

The config suffix uses this fixed order:

```
T{text}_F{feedback}_H{hand}_C{common}_P{prompt}_I{images}_R{remember}_S{full_steps}_E{temp}_M{max}
```

### Parameter Reference

| Code | Parameter | Values | Description |
|------|-----------|--------|-------------|
| `T` | `text_only` | `0`=False, `1`=True | Text-only mode (no images) |
| `F` | `feedback_type` | `n`=none, `s`=simple, `d`=detailed | Feedback type |
| `H` | `hand_transparency` | `00`-`99` | Hand overlay transparency (2-digit) |
| `C` | `include_common_sense` | `0`=False, `1`=True | Include common sense hints |
| `P` | `prompt_version` | `1`, `2`, ... | Prompt version number |
| `I` | `previous_image` | `0`=none, `1`=color, `2`=grayscale | Previous image type |
| `R` | `use_memory` | `0`=False, `1`=True | Use memory/remember |
| `S` | `full_steps` | `0`=False, `1`=True | Full steps mode |
| `E` | `temperature` | `00`-`99` | Temperature × 100 (e.g., `60` = 0.6) |
| `M` | `max_completion_tokens` | integer | Max tokens |

## Decoding Examples

### Example 1: `gpt-4o--T1_Fs_H60_C1_P2_I0_R0_S1_E60_M4096--rep1`

| Code | Value | Meaning |
|------|-------|---------|
| `T1` | text_only=True | Text-only mode |
| `Fs` | feedback_type=simple | Simple feedback |
| `H60` | hand_transparency=60 | **⚠️ Invalid with T1** |
| `C1` | include_common_sense=True | Common sense enabled |
| `P2` | prompt_version=v2 | Prompt version 2 |
| `I0` | previous_image=none | No previous image |
| `R0` | use_memory=False | Memory disabled |
| `S1` | full_steps=True | Full steps mode |
| `E60` | temperature=0.6 | Temperature 0.6 |
| `M4096` | max_completion_tokens=4096 | Max 4096 tokens |

### Example 2: `T0_Fs_H60_C1_P2_I1_R0_S1_E60_M4096`

| Code | Value | Meaning |
|------|-------|---------|
| `T0` | text_only=False | Images enabled |
| `Fs` | feedback_type=simple | Simple feedback |
| `H60` | hand_transparency=60 | Hand transparency 60% |
| `C1` | include_common_sense=True | Common sense enabled |
| `P2` | prompt_version=v2 | Prompt version 2 |
| `I1` | previous_image=color | Color previous image |
| `R0` | use_memory=False | Memory disabled |
| `S1` | full_steps=True | Full steps mode |
| `E60` | temperature=0.6 | Temperature 0.6 |
| `M4096` | max_completion_tokens=4096 | Max 4096 tokens |

## Configuration Validation Rules

The following constraints are enforced at runtime:

### text_only Mode Constraints

When `T1` (text_only=True):
- `I` must be `0` (previous_image=none)
- `H` must be `00` (hand_transparency=0)

**Error message:** `text_only mode must have hand_transparency of 0`

### Value Ranges

- `H` (hand_transparency): 0-100
- `E` (temperature): 0-100 (representing 0.0-1.0)

## Directory Structure

Experiment outputs are stored at:

```
Test/{test_suite}/{model}--{config_suffix}--rep{N}/
├── logs.txt          # Full execution log
├── Plans/            # Task execution plans
│   ├── task_name/    # Successful task
│   └── _task_name/   # Failed task (prefixed with _)
└── ...
```

## Quick Troubleshooting

### "text_only mode must have hand_transparency of 0"

The run name has `T1` (text_only) but `H` is not `00`.

**Fix:** Either:
- Change to `T0` if images are needed
- Change `H` to `00` for text-only mode

### Finding logs for a run

```bash
# Find the log file
find Test -path "*/{run_name}/logs.txt"

# Example
find Test -path "*gpt-4o--T1_Fs_H60_C1_P2_I0_R0_S1_E60_M4096--rep1*/logs.txt"
```

## Source Reference

The naming convention is implemented in:
- `Magmathor/Utils/config_utils.py` - `EvaluationConfig.get_output_suffix()`
- `experiment_runner/runner.py` - Run orchestration
