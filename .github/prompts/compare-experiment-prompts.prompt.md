---
agent: agent
description: 'Compare two Magmathor experiment runs to understand how prompt differences affect model behavior.'
name: compare-experiment-prompts
---

# Magmathor Prompt Comparison Analysis (Multi-Agent)

You are the **orchestrator agent** responsible for comparing two experiment runs to understand the impact of prompt changes. You will identify differences in the prompts, find tasks where model behavior diverged, and delegate deep analysis to sub-agents.

## Inputs

1. **Baseline Experiment ID**: `${input:baseline_id:The ID of the baseline experiment (e.g., gpt-4o--T0...)}`
2. **Comparison Experiment ID**: `${input:comparison_id:The ID of the experiment with the modified prompt}`

## Context

You are analyzing **Magmathor** logs. The core goal is to answer: **"Why are the prompt differences causing the model to behave differently?"**

To do this, you must:
1. Understand the difference between the prompts used in the two experiments.
2. Identify tasks where the outcome (Pass/Fail) or performance (Steps) changed.
3. Analyze the execution traces to trace the behavioral change back to the prompt change.

---

## Phase 1: Discovery & Prompt Diffing (You, the Orchestrator)

### Step 1: Locate Logs and Extract Prompts
1. **Find log files** for both experiments across all partitions (p1-p6).
   - Use `find Test -name "<experiment-id>" -type d` for both IDs.
2. **Extract the System Prompt** from one log file of each experiment.
   - Read the first ~500 lines of a `logs.txt` file from each experiment.
   - Extract text between `===============PROMPT=================` and `=============END PROMPT===============`.
3. **Analyze the Prompt Difference**:
   - Compare the two extracted prompts.
   - **Create a "Prompt Delta Summary"**: A concise list of what changed (e.g., "Added rule about checking inventory", "Removed chain-of-thought requirement", "Changed tone to be more cautious").
   - *You will pass this summary to all sub-agents.*

### Step 2: Compare Task Outcomes
1. **Parse Summary Tables**:
   - For every matching partition (e.g., p1 in Baseline vs. p1 in Comparison), read the summary table at the end of `logs.txt`.
2. **Identify "Interesting" Tasks**:
   - Compare the outcome of each task between Baseline and Comparison.
   - Select tasks for analysis based on these priorities:
     - **Priority A (Flip-flops)**: Passed in one, Failed in the other. (Most important).
     - **Priority B (Performance Shifts)**: Passed in both, but step count changed significantly (>20% difference).
     - **Priority C (New Failures)**: Failed in both, but potentially for different reasons (optional, if few A/B cases exist).
3. **Selection Limit**: Select up to **10 tasks** total, prioritizing Priority A.

---

## Phase 2: Parallel Comparative Analysis (Sub-Agents)

For each selected task, spawn a sub-agent to perform a comparative deep dive.

### Sub-Agent Prompt Template

```
You are analyzing the behavioral difference between two Magmathor runs for the same task.

**Task Name**: [task name]
**Partition**: [partition]
**Prompt Delta Summary**:
[Insert the summary of prompt changes identified in Phase 1]

**Run A (Baseline)**:
- ID: [baseline_id]
- Outcome: [Pass/Fail]
- Log File: [path to baseline log]
- Line Range: [start] to [end]

**Run B (Comparison)**:
- ID: [comparison_id]
- Outcome: [Pass/Fail]
- Log File: [path to comparison log]
- Line Range: [start] to [end]

**Your Goal**: Explain WHY the behavior changed, linking it specifically to the **Prompt Delta**.

### Analysis Steps
1. **Trace the Divergence**:
   - Compare the action sequences of Run A and Run B.
   - Identify the **Divergence Point**: The exact step where the model made a different decision (different action, different object, or significantly different reasoning).
2. **Analyze the Reasoning (<think> block)** at the Divergence Point:
   - Did Run B explicitly mention the new prompt instructions?
   - Did Run B notice something Run A missed (or vice versa)?
   - Did the prompt change cause a hallucination or a correction?
3. **Verify the Hypothesis**:
   - Does the behavioral change align with the intended prompt change? (e.g., if the prompt said "be careful with knives", did the model check for knives more often?)

### Output Format
**Save findings to**: `copilot-reports/comparisons/[baseline_vs_comparison]/analysis_[task_slug].md`

**Format**:
# Comparative Analysis: [Task Name]

## Executive Summary
- **Baseline Outcome**: [Pass/Fail] ([Steps] steps)
- **Comparison Outcome**: [Pass/Fail] ([Steps] steps)
- **Primary Cause of Change**: [One sentence summary, e.g., "Stricter inventory rules caused the model to double-check hands, preventing a premature PICKUP."]

## The Divergence Point
- **Step #**: [X]
- **Baseline Action**: `[Action]`
- **Comparison Action**: `[Action]`
- **Context**: [Briefly describe what was happening]

## Reasoning Analysis
- **Baseline Logic**: "[Quote relevant parts of <think> block]"
- **Comparison Logic**: "[Quote relevant parts of <think> block]"
- **Impact of Prompt Change**: [Explain how the specific prompt difference influenced this decision. Did the model cite the new rule?]

## Conclusion
- **Is the new prompt better?**: [Yes/No/Mixed]
- **Why?**: [Explanation]

After saving, return only the file path.
```

---

## Phase 3: Synthesis (Report Writer)

Once sub-agents return, dispatch a **Report Writer** to synthesize the findings.

### Report Writer Prompt

```
You are synthesizing a comparative analysis between two experiment runs.

**Baseline ID**: [baseline_id]
**Comparison ID**: [comparison_id]
**Prompt Delta Summary**: [Insert summary]

**Analyzed Tasks**:
- [List of task analysis files returned by sub-agents]

**Goal**: Answer "How did the prompt changes affect model behavior?"

### Report Template
**Save to**: `copilot-reports/comparisons/[baseline_vs_comparison]/FINAL_COMPARISON_REPORT.md`

# Prompt Comparison Report

## Experiment Details
- **Baseline**: [baseline_id]
- **Comparison**: [comparison_id]

## Prompt Changes (The "Delta")
[Summarize the key differences in the prompt]

## Impact Summary
| Metric | Baseline | Comparison | Delta |
|--------|----------|------------|-------|
| Total Passed (Common Tasks) | X | Y | +/- Z |
| Average Steps (Passed Tasks) | X | Y | +/- Z |

## Behavioral Analysis
[Synthesize findings from the task reports]

### 1. Positive Impacts
- **Observation**: [e.g., "Reduced precondition violations"]
- **Evidence**: [Cite specific tasks/examples]
- **Link to Prompt**: [e.g., "Likely due to the new 'Check Inventory' section"]

### 2. Negative Impacts / Regressions
- **Observation**: [e.g., "Increased hesitancy / loopiness"]
- **Evidence**: [Cite specific tasks]
- **Link to Prompt**: [e.g., "The 'Safety First' rule may be too aggressive"]

### 3. Neutral/Unintended Changes
- [Any other observations]

## Conclusion
[Final verdict: Is the new prompt an improvement? What should be tweaked?]
```
