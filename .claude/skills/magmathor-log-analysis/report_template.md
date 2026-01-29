# Report Template

Use this template when generating `copilot-reports/<experiment-id>/REPORT.md`.

## ⚠️ CRITICAL REQUIREMENTS

1. **Get pass/fail metrics from `Test/results.csv`** - Run `python -m Magmathor.Model.generate_reports` first if needed. Use the `magmathor-results-analysis` skill for CSV queries.

2. **IMAGES ARE REQUIRED** - Every failure example in the report MUST include `<img>` tags with absolute paths from the parsed JSON. Without images, visual analysis is impossible.

3. **Use parsed JSON for images** - The `parse_log.py` script outputs absolute image paths in `step.images.current` and `step.images.previous`.

---

```markdown
# Error Report: [Experiment ID]

## Summary

*Metrics from `Test/results.csv` (query with `csvsql`)*

- **Test partitions analyzed**: [list, e.g. p1, p2, …, p6]
- **Total tasks attempted**: X
- **Tasks passed**: X (Y%)
- **Tasks failed**: X (Y%)
- **Goals reached**: Y% (partial credit metric)

## Configuration

### Experiment Settings

| Setting | Value | Impact |
|---------|-------|--------|
| `text_only` | False/True | [No images / Images enabled] |
| `feedback_type` | none/simple/detailed | [Error feedback level] |
| `previous_image` | none/color/grayscale | [Temporal context level] |
| `use_memory` | False/True | [Stateless / Context persists] |
| `include_common_sense` | False/True | [Domain hints in prompt] |
| `full_steps` | False/True | [Single action / Full plan] |
| `hand_transparency` | N | [Hand visibility level] |
| `temperature` | X.X | [Model temperature] |

### Expected Failure Types Given Config

Based on `config_failure_modes.md`, these failure types are **expected** given the configuration:

| Config Setting | Expected Failures |
|----------------|-------------------|
| `text_only=True` | Spatial confusion, visibility errors, navigation failures |
| `feedback_type=none` | Repetitive identical errors, no error recovery |
| `previous_image=none` | State tracking failures, action verification errors |
| `use_memory=False` | Goal forgetting, progress loss, context errors |
| `include_common_sense=False` | Physics violations, prerequisite skips |

*Note: Failures marked as "config-natural" should be evaluated differently than unexpected failures.*

## Failures Natural to This Config

These failures are **expected** given the experiment configuration and should not be attributed to model capability issues:

### Summary

| Failure Type | Count | Config Cause |
|--------------|-------|--------------|
| [category] | N | [which config setting explains this] |

### Examples

For each config-natural failure type, include 2-3 examples showing:
1. The error that occurred
2. Why this is expected given the config
3. What would be needed to avoid it (e.g., "requires visual grounding")

## Unexpected Failures

These failures are **NOT explained** by the configuration and indicate potential model or prompt issues:

### Summary

| Failure Type | Count | Severity |
|--------------|-------|----------|
| [category] | N | High/Medium/Low |

## What the model thinks about recurring failures
- Summarize recurring *beliefs* from `<think>` that drive loops (e.g., "receptacle is open" while sim says CLOSED; "target is visible" while sim says not visible).
- Include 2–3 **direct quotes** from logs.

## Failure samples (5 per category) — ⚠️ MUST INCLUDE IMAGES

For **each error category** (see `error_categories.md`), include **5 concrete examples**.

### Complete Example Structure

**not_visible_target #1** — `cook__Bread_Plate(d)_FloorPlan3_V1` (p1) @ **14:18:36**

**Config-natural?**: No

**Provenance**:
- Source log: `Test/magt_benchmark_p1/gpt-4o--T0_Fs.../logs.txt`
- Parsed JSON: `copilot-reports/T0_Fs.../logs.json`

**Action history:**
```
1. FIND Bread (here include feedback if any)
```

**Memory from previous steps**:
(only if `use_memory=True` in config)
```
- I need to cook the Bread on the Plate.
- I'm currently attempting to find the Bread.
```

**Model reasoning**:
```
I can see the Bread on the CounterTop. I should pick it up to proceed with cooking.
```

**Extracted Suggested Next Steps:**
(only if present; only if `full_steps=True` in config)
```
1. PICKUP Bread
2. PUT CounterTop
```

**Extracted action:**

```
PICKUP Bread
```


**Action execution**:
```
Action: PICKUP Bread
Extracted action: pickup, Extracted item: Bread
Executing action: pickup Bread
[STEP ERROR] Target Bread is not visible
```

**Images** (from parsed JSON `step.images`):

| Previous | Current |
|----------|---------|
| <img src="/home/.../Plans/_cook__Bread.../4_slice Bread.png" width="360" /> | <img src="/home/.../Plans/_cook__Bread.../5_pickup Bread.png" width="360" /> |

**Assessment**: Model claims "Bread is visible on CounterTop" but the Current image shows Bread is behind the Fridge door. This is a perception error — the model hallucinated object visibility.

---

### Required Elements Checklist

Each example must contain:

1. ✅ **Header**: `**<category> #N** — \`<task>\` (<partition>) @ **<timestamp>**`
2. ✅ **Config-natural?**: Yes/No — [explanation if yes]
3. ✅ **Provenance**: Source log path + parsed JSON path
4. ✅ **Model reasoning**: Quote from `<think>` block
5. ✅ **Action execution**: The action and error from the log
6. ✅ **Images**: `<img>` tags with absolute paths from `step.images.previous` and `step.images.current`
7. ✅ **Assessment**: Compare model claim vs Current image vs error

### Getting Image Paths

From the parsed JSON (`logs.json`), each step has:
```json
"images": {
  "previous": "/absolute/path/to/previous.png",  // may be null
  "current": "/absolute/path/to/current.png"     // always present
}
```

Copy these paths directly into `<img src="...">` tags.

---

## Recommendations

### Config-Related

If many failures are config-natural, consider:
1. [Suggest config changes that would reduce expected failures]
2. [Trade-offs of those changes]

### High Priority
1. [Most impactful fix based on **unexpected** failure frequency / severity]

### Medium Priority
2. [Secondary fixes for unexpected failures]
```
