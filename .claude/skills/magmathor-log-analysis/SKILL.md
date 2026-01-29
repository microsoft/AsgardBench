---
name: magmathor-log-analysis
description: Analyze Magmathor experiment logs to identify, categorize, and report failure patterns. Use this when asked to debug failing Magmathor experiments, analyze test results, generate a performance report, or investigate why tasks failed.
---

# Magmathor Experiment Log Analysis

Analyze experiment logs from the Magmathor embodied AI testing framework and produce comprehensive error reports.

## 🎯 You Are The Orchestrator

This skill uses **sub-agents** to handle each step. Your job is to:
1. Determine what needs to be analyzed
2. Delegate tasks to specialized agents
3. Coordinate the outputs
4. Ensure the final report has IMAGES

## Sub-Agents Available

| Agent | Purpose | Invoke With |
|-------|---------|-------------|
| `log-parser` | Parse logs.txt → JSON | "Use the log-parser agent to..." |
| `results-scorer` | Get metrics from CSV | "Use the results-scorer agent to..." |
| `failure-sampler` | Sample failures with images | "Use the failure-sampler agent to..." |
| `report-writer` | Write final REPORT.md | "Use the report-writer agent to..." |

## Working Directory

All files go in `copilot-reports/<config-id>/`:

```
copilot-reports/T0_Fs_H60_C1_P2_I1_R0_S1_E60_M4096/
├── logs.json           # From log-parser
├── failure_samples.json # From failure-sampler
└── REPORT.md           # From report-writer (WITH IMAGES!)
```

---

## Orchestration Workflow

### Step 1: Identify the experiment

Determine the config string from the user's request or folder name:
- Folder: `Test/magt_benchmark_p1/gpt-4o--T0_Fs_H60_C1_P2_I1_R0_S1_E60_M4096--rep1/`
- Config: `gpt-4o--T0_Fs_H60_C1_P2_I1_R0_S1_E60_M4096`

Create the working directory:
```bash
mkdir -p copilot-reports/<model-name>--<config-id>
```

### Step 2: Get metrics (delegate to results-scorer)

```
Use the results-scorer agent to get metrics for config gpt-4o--T0_Fs_H60_C1_P2_I1_R0_S1_E60_M4096.
Save the results summary.
```

The agent will:
1. Run `python -m Magmathor.Model.generate_reports` if needed
2. Query `Test/results.csv` for overall and per-partition metrics
3. Return a metrics summary

### Step 3: Parse logs (delegate to log-parser)

```
Use the log-parser agent to parse the logs for config gpt-4o--T0_Fs_H60_C1_P2_I1_R0_S1_E60_M4096.
Input: Test/magt_benchmark_p1/gpt-4o--T0_Fs_H60_C1_P2_I1_R0_S1_E60_M4096--rep1/logs.txt
Output: copilot-reports/gpt-4o--T0_Fs_H60_C1_P2_I1_R0_S1_E60_M4096/logs.json
```

The agent will:
1. Run `parse_log.py` on the logs
2. Auto-load config.json for config-aware parsing
3. Output structured JSON with absolute image paths

### Step 4: Sample failures (delegate to failure-sampler)

```
Use the failure-sampler agent to sample failures from:
Input: copilot-reports/gpt-4o--T0_Fs_H60_C1_P2_I1_R0_S1_E60_M4096/logs.json
Output: copilot-reports/gpt-4o--T0_Fs_H60_C1_P2_I1_R0_S1_E60_M4096/failure_samples.json

Sample 5 failures per error category. Include image paths.
```

The agent will:
1. Load the parsed JSON
2. Sample from each `errors_by_category`
3. Include absolute image paths for each sample
4. Output failure_samples.json

### Step 5: Write report (delegate to report-writer)

```
Use the report-writer agent to create the final report:
- Logs JSON: copilot-reports/gpt-4o--T0_Fs_H60_C1_P2_I1_R0_S1_E60_M4096/logs.json
- Metrics: [paste the metrics summary from step 2]
- Failure samples: copilot-reports/gpt-4o--T0_Fs_H60_C1_P2_I1_R0_S1_E60_M4096/failure_samples.json
- Output: copilot-reports/gpt-4o--T0_Fs_H60_C1_P2_I1_R0_S1_E60_M4096/REPORT.md

⚠️ CRITICAL: The report MUST include <img> tags with absolute paths for all failure examples.
```

The agent will:
1. Follow `report_template.md`
2. Include metrics from results-scorer
3. Include failure samples WITH IMAGES
4. Write REPORT.md

### Step 6: Verify images

After the report is written, verify images are included:
```bash
grep -c "<img src=" copilot-reports/<model-name>--<config-id>/REPORT.md
```

If count is 0, the report-writer agent needs to redo the failure samples section.

---

## Supporting Files

| File | Purpose |
|------|---------|
| `parse_log.py` | Log parser script (outputs absolute image paths) |
| `config_failure_modes.md` | Config-to-failure mapping |
| `report_template.md` | Full markdown template |
| `error_categories.md` | The 8 error categories |

## Related Skills

| Skill | Use For |
|-------|---------|
| `magmathor-results-analysis` | Aggregate metrics, comparing configs |
| `magmathor-run-naming` | Decoding experiment run names |

---

## Quick Single-Agent Fallback

If sub-agents aren't available, you can do everything yourself:

1. **Parse**: `uv run python .claude/skills/magmathor-log-analysis/parse_log.py <logs.txt> -o <output.json>`
2. **Score**: `uv run csvsql --query "SELECT ..." Test/results.csv`
3. **Sample**: Read parsed JSON, extract 5 failures per category with images
4. **Write**: Follow `report_template.md`, include `<img src="...">` tags

⚠️ Every failure example MUST include images from the parsed JSON's `step.images` field.
