---
name: magmathor-results-analysis
description: Analyze experiment results from Test/results.csv. Use this when asked to report on experiment performance, compare configurations, find trends, or identify interesting patterns in results.
---

# Magmathor Results Analysis

Analyze aggregated experiment results from `Test/results.csv` to report on performance, compare configurations, and find patterns.

## When to Use This Skill

- User asks about experiment performance or success rates
- User wants to compare different configurations (e.g., feedback types, hand_transparency)
- User asks to find trends or interesting connections in results
- User asks about which settings perform best
- User wants to understand how ablations affect performance

## Results File Location

```
Test/results.csv
```

This is a large CSV file. **Do not read the entire file**. Use csvkit for analysis.

## Tool: csvkit

Use `csvkit` for all CSV analysis. Run with `uv run`.

### Key Commands

| Command | Purpose |
|---------|---------|
| `csvstat` | Column statistics and summary |
| `csvsql` | Run SQL queries on CSV files |
| `csvgrep` | Filter rows by column value |
| `csvcut` | Select specific columns |
| `csvlook` | Pretty-print as table |

## CSV Column Reference

### Key Columns for Filtering

| Column | Description |
|--------|-------------|
| `model` | Model identifier (e.g., `gpt-4o`) |
| `config` | Config string (e.g., `T0_Fs_H60_C1_P2_I1_R0_S1_E60_M4096`) |
| `rep` | Replicate: `rep1`, `rep2`, or `AVG` (average across reps) |
| `test_set_name` | Test partition: `magt_benchmark_p1`–`p6` or `ALL` (aggregated) |

### Config Columns

| Column | Description |
|--------|-------------|
| `feedback_type` | `none`, `simple`, or `detailed` |
| `full_steps` | True/False - full action sequence mode |
| `hand_transparency` | 0–100 - hand overlay visibility |
| `include_common_sense` | True/False - common sense hints |
| `previous_image` | `none`, `color`, or `grayscale` |
| `text_only` | True/False - no images sent |
| `use_memory` | True/False - context persistence |

### Metric Columns

| Column | Description |
|--------|-------------|
| `success %` | Success rate (0.0–1.0) |
| `fail %` | Failure rate (0.0–1.0) |
| `success #` | Tasks completed successfully |
| `fail #` | Tasks that failed |
| `step %` | Average steps / optimal steps |
| `valid %` | Ratio of valid actions |
| `undoable %` | Actions that couldn't be undone |
| `goals reached %` | Percentage of goals reached |
| `expected_num_plans` | Number of tasks (18 per partition, 108 for ALL) |
| `test_completed` | True if run finished |

## Key Metrics

| Metric | Description | Good Value |
|--------|-------------|------------|
| `success_percentage` | Primary success metric | Higher is better |
| `goals_reached_percent` | Partial credit metric | Higher is better |
| `valid_ratio` | Action validity | Higher is better |
| `step %` | Efficiency (steps/optimal) | Lower is better (1.0 = optimal) |
| `undoable %` | Ratio of bad actions | Lower is better |

## Analysis Examples

### View Column Names
```bash
uv run csvstat --names Test/results.csv
```

### Get Results for a Specific Config
```bash
uv run csvsql --query "
  SELECT config, rep, test_set_name, \"success %\", \"goals reached %\"
  FROM results
  WHERE config = 'T0_Fs_H60_C1_P2_I1_R0_S1_E60_M4096'
  AND rep = 'AVG'
  AND test_set_name = 'ALL'
  LIMIT 1
" Test/results.csv | csvlook
```

### Filter Aggregated Results

**Always use `rep='AVG'` and `test_set_name='ALL'`** for overall config comparisons.

```bash
uv run csvsql --query "
  SELECT config, \"success %\", \"goals reached %\"
  FROM results
  WHERE rep = 'AVG' AND test_set_name = 'ALL'
  GROUP BY config
  ORDER BY \"success %\" DESC
" Test/results.csv | csvlook
```

### Compare Feedback Types
```bash
uv run csvsql --query "
  SELECT feedback_type,
         AVG(\"success %\") as avg_success,
         AVG(\"goals reached %\") as avg_goals
  FROM results
  WHERE rep = 'AVG' AND test_set_name = 'ALL'
  GROUP BY feedback_type
  ORDER BY avg_success DESC
" Test/results.csv | csvlook
```

### Compare Previous Image Settings
```bash
uv run csvsql --query "
  SELECT previous_image,
         AVG(\"success %\") as avg_success,
         COUNT(*) as configs
  FROM results
  WHERE rep = 'AVG' AND test_set_name = 'ALL'
  GROUP BY previous_image
  ORDER BY avg_success DESC
" Test/results.csv | csvlook
```

### Best Configurations
```bash
uv run csvsql --query "
  SELECT config, \"success %\", \"goals reached %\"
  FROM results
  WHERE rep = 'AVG' AND test_set_name = 'ALL'
  GROUP BY config
  ORDER BY \"success %\" DESC
  LIMIT 10
" Test/results.csv | csvlook
```

### Effect of Memory
```bash
uv run csvsql --query "
  SELECT use_memory, AVG(\"success %\") as avg_success
  FROM results
  WHERE rep = 'AVG' AND test_set_name = 'ALL'
  GROUP BY use_memory
" Test/results.csv | csvlook
```

### Per-Partition Performance
```bash
uv run csvsql --query "
  SELECT test_set_name, AVG(\"success %\") as avg_success
  FROM results
  WHERE rep = 'AVG' AND test_set_name LIKE 'magt_benchmark%'
  GROUP BY test_set_name
  ORDER BY test_set_name
" Test/results.csv | csvlook
```

### Filter by Specific Config
```bash
uv run csvsql --query "
  SELECT config, \"success %\", \"goals reached %\"
  FROM results
  WHERE config LIKE '%Fd%'
  AND rep = 'AVG' AND test_set_name = 'ALL'
  GROUP BY config
" Test/results.csv | csvlook
```

### Multi-Factor Comparison
```bash
uv run csvsql --query "
  SELECT feedback_type, previous_image, use_memory,
         AVG(\"success %\") as avg_success,
         AVG(\"valid %\") as avg_valid,
         COUNT(DISTINCT config) as configs
  FROM results
  WHERE rep = 'AVG' AND test_set_name = 'ALL'
  GROUP BY feedback_type, previous_image, use_memory
  ORDER BY avg_success DESC
" Test/results.csv | csvlook
```

## Understanding Experiment Design

Each row is one **experiment run** with a specific configuration.

### Key Columns

| Column | Description |
|--------|-------------|
| `config` | Unique config identifier (e.g., `T0_Fs_H60_C1_P2_I1_R0_S1_E60_M4096`) |
| `rep` | `rep1`, `rep2`, or `AVG` (average across replicates) |
| `test_set_name` | `magt_benchmark_p1`–`p6` or `ALL` (aggregate) |

### Ablation Dimensions

| Parameter | Ablation Values | Baseline |
|-----------|-----------------|----------|
| `feedback_type` | none, simple, detailed | simple |
| `hand_transparency` | 0, 60 | 60 |
| `previous_image` | none, color, grayscale | color |
| `use_memory` | False, True | False |
| `full_steps` | False, True | True |
| `text_only` | False, True | False |
| `include_common_sense` | False, True | True |

### Filtering Best Practices

**For overall config comparisons, always use:**
```sql
WHERE rep = 'AVG' AND test_set_name = 'ALL'
```

This gives you one row per config with averaged results across all replicates and partitions.

### Rows with `test_set_name=ALL`

These are **aggregated results** across all 6 partitions (108 tasks total). Use these for overall comparisons.

### Rows with `test_set_name=magt_benchmark_pN`

These are **per-partition results** (18 tasks each). Use these to identify partition-specific patterns.

## Linking to Run Names

The `config` column directly matches the config part of run directory names:

```
{model}--{config}--rep{N}
```

For example, directory `gpt-4o--T0_Fs_H60_C1_P2_I1_R0_S1_E60_M4096--rep1` has:
- `model = gpt-4o`
- `config = T0_Fs_H60_C1_P2_I1_R0_S1_E60_M4096`
- `rep = rep1`

See `magmathor-run-naming` skill for decoding the config string.

## Quick Tips

1. **Always use `rep='AVG'` and `test_set_name='ALL'`** for overall config comparisons
2. **Use `config` column** to filter/group by experiment configuration
3. **Use `success %` as primary metric**, `goals reached %` for partial credit
4. **Lower `step %`** = more efficient (fewer steps to complete tasks)
5. **Look at `valid %`** to understand action quality
6. **Pipe to `csvlook`** for readable table output
7. **Use `GROUP BY config`** to deduplicate when there are multiple rows per config
