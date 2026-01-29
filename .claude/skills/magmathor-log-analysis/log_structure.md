# Magmathor Log Structure Reference

## Delimiters

| Element | Start marker | End marker |
|---------|--------------|------------|
| Model response | `===============RESPONSE=================` | `=============END RESPONSE===============` |
| Prompt (noise) | `===============PROMPT=================` | `=============END PROMPT===============` |

## Key log lines

- **Action execution**: `Executing action: <action> <object>`
- **Low-level call**: `(0) PutObject ...`, `(0) PickupObject ...`, etc.
- **Error feedback**: `[STEP ERROR] <message>`
- **Image filenames**: block starting with `=== IMAGES ===`, then `Previous: <file>` and `Current: <file>`
- **Task start**: `Testing: [X/Y] ...`
- **Task end**: `test passed` or `test failed`
- **Summary table**: at end of `logs.txt`, lists every task with Pass/Fail counts and a `Total:` row

## Image Naming Convention

Images are saved as `{step_number}_{action}.png` in the `Plans/{task_name}/` directory.

Example sequence:
```
0_find Mug.png       <- Result after step 0 (FIND Mug)
1_pickup Mug.png     <- Result after step 1 (PICKUP Mug)
2_put CoffeeMachine.png <- Result after step 2 (PUT CoffeeMachine)
```

## Image Semantics (IMPORTANT)

The `=== IMAGES ===` block appears AFTER each action is executed:

```
[Response N - model outputs action]
=============END RESPONSE===============
-  Action content: PUT Cabinet
-  Extracted action: [PUT]
Executing action: put Cabinet
(0) PutObject ...
=== IMAGES ===
Previous: 4_pickup Mug.png    <- Image from step N-1 (was model's input at step N)
Current: 5_put Cabinet.png    <- Result AFTER step N's action executed
```

### When Analyzing Step N:

| Semantic Name | Log Field | Description | File Pattern |
|---------------|-----------|-------------|--------------|
| `model_input` | `Previous:` | What the model SAW when making decision at step N | `(N-1)_*.png` |
| `action_result` | `Current:` | Result AFTER step N's action was executed | `N_*.png` |

### Example - Analyzing a failure at step 5:

```
Step 5:
  Error: "PUT: No object held"
  model_input: "4_pickup Mug.png"    <- Check this to see what model saw
  action_result: "5_put Cabinet.png" <- Check this to see result after failure
```

To understand WHY the model made a bad decision:
1. Look at `model_input` (what the model saw)
2. Read the model's `<think>` reasoning
3. Compare what model claimed to see vs what's actually in `model_input`

### Special Cases

- **Step 0**: No `Previous:` field (first action, no prior image)
- **Failed tasks**: Directory name is prefixed with `_` (e.g., `_coffee__empty_FloorPlan13_V1`)

## Filtering prompt noise

Use `parse_log.py` with the `--filter` flag to automatically strip prompts:

```bash
python .claude/skills/magmathor-log-analysis/parse_log.py logs.txt --filter
# Creates filtered_logs.txt in the source directory
```

Or manually with sed:

```bash
sed '/===============PROMPT=================/,/=============END PROMPT===============/d' logs.txt > filtered.txt
```
