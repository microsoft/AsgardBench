---
agent: agent
description: 'Orchestrate multi-agent analysis of Magmathor experiment logs to identify, categorize, and report failure patterns.'
name: analyze-model-responses-in-log
---

# Magmathor Experiment Error Analysis (Multi-Agent)

You are the **orchestrator agent** responsible for analyzing an experiment log file and producing a comprehensive error report. You will delegate detailed analysis to **sub-agents** and then synthesize their findings.

## Input

**Experiment identifier:** `${input:experiment_id:Experiment ID without partition (e.g., gpt-4o--T0_Fs_H60_C1_P2_I1_R0_S1_E60_M4096--rep2)}`

The analysis will automatically discover and analyze all partition splits for this experiment configuration.

## Context

You are analyzing logs from **Magmathor**, an embodied AI testing framework. The agent receives visual observations (images) and must perform household tasks in a simulated 3D environment by issuing actions like `FIND`, `PICKUP`, `PUT`, `SLICE`, `TOGGLE_ON`, etc.

### Log Structure

- **Model responses** are delimited by:
  - `===============RESPONSE=================` (start)
  - `=============END RESPONSE===============` (end)
- **Prompts** are delimited by:
  - `===============PROMPT=================` (start)
  - `=============END PROMPT===============` (end)
- **Action outcomes** appear after responses as:
  - `Executing action: <action> <object>`
- **Images** are referenced as:
  - `=== IMAGES ===` followed by `Current: <image_filename>`
  - In the action history, feedback depends on the experiment's feedback mode:
    - **None (`Fn`)**: No success/failure indication
    - **Simple (`Fs`)**: Each action marked with `Success` or `Failure`
    - **Detailed (`Fd`)**: Failures include an explanation of why the action couldn't be executed
- **Goal evaluation** appears as `Evaluating goals...` followed by `Pass` or `Failed` state goals
- **Task boundaries** are marked by:
  - Task start: Lines starting with `Testing: [X/Y]` followed by the task name (e.g., `Testing: [0/18] ... - task_name - ...`)
  - Task end: Lines containing `test passed` or `test failed` followed by summary stats
- **Task context** is shown as `Your task is: <description>`
- **Summary Table**: At the end of the log, a table lists all tasks with columns for Pass/Fail counts and a final "Total:" row.

---

## Phase 1: Structure Extraction (You, the Orchestrator)

First, discover all partitions and build a comprehensive picture of the experiment:

1. **Discover all experiment partitions**:
   - Use: `find Test -name "<experiment-id>" -type d`
   - This finds all directories matching the experiment ID across all benchmark partitions
   - Each found directory contains a `logs.txt` file
   - You'll typically find 6 partitions (p1-p6), but the find command is flexible

2. **For each discovered partition's logs.txt**:
   - **Locate the summary table** at the very end of the file. It lists each task with "Pass/Fail" counts and ends with a "Total:" row.
   - **Extract statistics** directly from this table (Pass/Fail counts per task and partition totals).
   - **Scan for task boundaries** (start/end markers) specifically to find the **line ranges** for the *failed* tasks.
   - Build a unified list of failed tasks with their line ranges.

3. **Aggregate statistics** using the extracted summary tables:
   - Use the "Total" row from each partition's summary table to get accurate counts.
   - Calculate overall pass/fail rates across all partitions.
   - Note any significant performance differences between partitions.

4. **Identify failed tasks** that need deep analysis:
   - Include failed tasks from ALL discovered partitions
   - You'll dispatch sub-agents for each failed task
   - Note which partition each task comes from (p1, p2, etc.)

---

## Phase 2: Parallel Task Analysis (Sub-Agents)

For **each failed task**, spawn a sub-agent to perform deep analysis.

### Sub-Agent Prompt Template

Use this template when dispatching each sub-agent:

```
You are analyzing a single failed task from a Magmathor embodied AI experiment.

**Task Name**: [task name]
**Task Goal**: [the "Your task is:" description]
**Partition**: [partition name, e.g., magt_benchmark_p1]
**Log File**: [full path to logs.txt]
**Line Range**: [start line] to [end line]

**Image Access**:
If you need to verify visual observations, you can find the images at:
`[Log Directory]/Plans/_[Task Name]/[Image Name]`
- **Log Directory**: The folder containing `logs.txt`.
- **Task Folder**: Inside `Plans/`, the task name prefixed with `_` (since it failed).
- **Image Name**: Referenced in the log under `=== IMAGES ===`.
  - **Important**: The log lists images *after* the action execution.
  - Use `Previous: <file>` to see what the model saw *before* the action (the observation it acted upon).
    - **Note**: `Previous` images are only available if the experiment ID contains `_I1_` (color) or `_I2_` (grayscale).
  - Use `Current: <file>` to see the result *after* the action.

Read the specified section of the log file and analyze the model's behavior.

## What to Look For

Identify errors in these categories (only report categories where you find issues):

1. **Precondition Violations**: Actions attempted without meeting requirements (e.g., SLICE without Knife, PUT into dirty container)
2. **Inventory Mismanagement**: PICKUP while holding something, PUT while holding nothing
3. **Visual Misinterpretation**: Claiming to see objects that aren't visible, or missing visible objects
4. **Memory Inconsistency**: Model's memory contradicts actual state
5. **World Rule Violations**: Ignoring stated rules (e.g., not OPENing container before PICKUP)
6. **Repetitive Failures**: Same failed action repeated without adaptation
7. **Reasoning Errors**: Flawed logic in <think> block
8. **Other**: Any pattern that doesn't fit above

## Output

**Save your findings to**: `copilot-reports/<experiment-id>/analysis_p{partition}_{task-slug}.md`

Use this exact format:

### Task: [task name]
**Outcome**: Failed
**Steps Taken**: [number]
**Actions Failed**: [number]

#### Errors Found

**[Error Category 1]** (count: X)
- Description: [what happened]
- Line numbers with examples: [line X, line Y, ...]
- Model reasoning excerpt: "[quote from model's response]"
- Image Evidence: [Full absolute path to the relevant image, if applicable]
- Why it's wrong: [brief explanation]

**[Error Category 2]** (count: X)
...

#### Key Failure Point
[Describe the critical error(s) that most likely caused task failure]

After saving the file, return only the file path you wrote to.
```

### Dispatching Sub-Agents

For each failed task:
1. Extract the relevant log segment (or note the line range)
2. Spawn a sub-agent with the template above, filled in with task details
3. Sub-agent saves findings to: `copilot-reports/<experiment-id>/analysis_p{partition}_{task-slug}.md`
4. Sub-agent returns only the file path it wrote to
5. Collect all returned file paths

You can run sub-agents **in parallel** for efficiency.

---

## Phase 3: Synthesis and Report Generation (Report Writer Sub-Agent)

Once all task analysis sub-agents complete, you (the orchestrator) will receive a list of file paths.

Dispatch a **Report Writer sub-agent** with these file paths and the following instructions.

> **Important**: When dispatching the Report Writer, replace the `[ORCHESTRATOR: Insert the final report template here...]` placeholder with the actual template from the "Final Output Format" section below.

### Report Writer Sub-Agent Prompt

```
You are synthesizing error analysis results from multiple task analyses across all test partitions.

**Experiment ID**: [experiment-id]

**Partition statistics** (from orchestrator's Phase 1 extraction):
[Include the pass/fail counts per partition that you gathered in Phase 1]

**Task analysis files** (read all of these):
- [path 1]
- [path 2]
- ... (all paths returned from task analysis sub-agents)

## Your Task

1. Read all the provided task analysis files
2. Aggregate error patterns across all failed tasks from all partitions:
   - Count total occurrences of each error category
   - Identify which errors are systemic (appear in multiple tasks/partitions) vs. isolated
   - Note if error frequencies differ between partitions (may indicate partition-specific issues)

3. Use the partition statistics provided to build the Partition Summary table

4. Identify cross-task and cross-partition patterns:
   - Are certain error types correlated with certain task types?
   - Do error patterns vary across partitions?
   - Are there cascading failures?

5. Generate recommendations based on error frequency and impact

6. **Write the final report** to: `copilot-reports/[experiment-id]/REPORT.md`
   - Use the report template provided below
   - **Images**: When including examples, embed the relevant image using HTML syntax: `<img src="/absolute/path/to/image.png" width="400" alt="description">`. Ensure the path is absolute so it renders correctly in VS Code.
   - After saving, return only the file path of the completed report

## Report Template

[ORCHESTRATOR: Insert the final report template here when dispatching this sub-agent]
```

---

## File Structure

All outputs for an experiment go under one folder:

```
copilot-reports/
└── <experiment-id>/
    ├── analysis_p1_task_name_1.md
    ├── analysis_p1_task_name_2.md
    ├── analysis_p2_task_name_3.md
    ├── ...
    └── REPORT.md  (final synthesized report)
```

---

## Final Output Format

**Report Writer saves to**: `copilot-reports/<experiment-id>/REPORT.md`

Use this template for the final report:

```markdown
# Error Report: [Experiment ID]

## Summary
- **Test partitions analyzed**: [list discovered partitions]
- **Total tasks attempted**: X (across all partitions)
- **Tasks passed**: X (Y%)
- **Tasks failed**: X (Y%)
- **Total actions analyzed**: ~X
- **Total failed actions**: ~X

### Partition Summary
| Partition | Tasks | Pass | Fail | Pass Rate |
|-----------|-------|------|------|-----------|
| p1 | X | X | X | Y% |
| p2 | X | X | X | Y% |
| ... | ... | ... | ... | ... |
| **Total** | **X** | **X** | **X** | **Y%** |

### Task Breakdown (Failed Tasks Only)
| Task Name | Partition | Outcome | Steps | Key Issue |
|-----------|-----------|---------|-------|-----------|
| [name] | p1 | Fail | X | [brief description] |
...

## Error Categories (by frequency across all partitions)

### 1. [Most Common Error Category]
**Total occurrences**: X across Y tasks (Z partitions)
**Description**: [What this error pattern looks like]
**Partition distribution**: [e.g., appears in all partitions, concentrated in p1-p3, etc.]

**Example 1** (Task: [task name], Partition: p#, Step X):
> [Relevant excerpt from model's response]

<img src="[absolute path to image]" width="400" alt="Example Image">

**Why it failed**: [Explanation]
**Correct action would have been**: [What should have happened]

---

### 2. [Second Most Common]
...

## Cross-Partition Analysis
[Observations about whether error patterns are consistent across partitions or vary significantly]

## Cross-Task Patterns
[Observations about patterns that span multiple tasks]

## Recommendations

### High Priority
1. [Most impactful suggestion]
2. ...

### Medium Priority
3. ...

### Lower Priority
...

## Appendix: Per-Task Details by Partition
[Optional: Include condensed sub-agent reports organized by partition for reference]
```

---

## Important Notes

- **Sub-agent context**: Each sub-agent has a fresh context window — give them all the information they need in the prompt
- **Only report observed errors**: Skip categories with zero occurrences
- **Prioritize by impact**: Focus on errors that caused task failures, not minor issues
- **Be specific**: Include line numbers, timestamps, and exact quotes where helpful
- **Look for root causes**: Often one error (e.g., visual misinterpretation) cascades into others (repetitive failures)
- **Check the `<think>` block**: The model's reasoning often reveals why it made wrong decisions
- **Sub-agents write files**: Task analysis sub-agents write their own output files and return only the path. This keeps the orchestrator's context clean.
