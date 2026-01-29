# Tips & Checklist

Use this checklist when generating or reviewing a Magmathor log-analysis report.

## Working Directory

All files should be saved to `copilot-reports/<experiment-id>/`:
- `logs.json` - Parsed log data
- `filtered_logs.txt` - Prompt-stripped log
- `failure_samples.json` - Sampled failures
- `REPORT.md` - Final report

## Sampling

- Include **5 samples per error category** (not just a handful overall).
- Prefer samples from **different tasks/partitions** to show breadth.
- Use the `errors_by_category` from parsed JSON to find samples.

## Provenance

Every sample must include:
- The **`logs.txt` path** it came from.
- The **parsed JSON path** (`copilot-reports/<experiment-id>/logs.json`).

## Action execution block

For each sample, include the **full same-timestamp block**:
```
Action: ...
-  Action content: ...
-  Extracted action: [...]
-  Extracted item: [...]
Executing action: ...
InvalidOperationException: ... (if any)
(0) <LowLevelCall> ...
     [<ObjectID>] <error message>
=== IMAGES ===
Previous: <file>
Current: <file>
[STEP ERROR] <message>
```

## Images

- Always use **both** `Previous` and `Current` when available.
- **Semantics**:
  - `Previous` = frame before the *previous* action was applied.
  - `Current` = frame representing the *current* state before the model's current action.
- Use **absolute paths** (`<img src="/home/...">`) so VS Code / Streamlit can render them.

## Assessment

For each sample, write an explicit assessment:
1. What does the model claim in `<think>`?
2. What does the `Current` image actually show?
3. What does the simulator feedback say?
4. Is this a **model/prompt failure** or a **sim/controller issue**?

## Prompt linkage

If the failure pattern seems encouraged or not covered by the example prompts (`Magmathor/Data/prompts/example{1,2,3}.prompt`) or the main prompt, note it explicitly.

Examples:
- "Prompt linkage: Example1/2 demonstrate OPEN→PUT but don't teach recovery when sim says CLOSED after a PUT."
- "Prompt linkage: This failure mode (Ran out of candidate poses) is not covered by examples 1–3."

## Sharing

If the user asks to share the report with others:
- Create a minimal **Streamlit viewer** (`streamlit_report_app.py`) that:
  - Loads `copilot-reports/*/REPORT.md`
  - Renders markdown
  - Inlines local images as base64 data URIs so the report is portable.
