# Magmathor Error Categories

When analyzing failed tasks, classify errors into the categories below. Each includes a typical `[STEP ERROR]` pattern you can grep for.

**Note**: The `parse_log.py` script automatically categorizes errors into these categories. Check the `error_category` and `is_config_natural` fields in the parsed JSON.

| # | Category | Typical error text (grep pattern) |
|---|----------|-----------------------------------|
| 1 | **Openable-precondition** | `Target openable Receptacle is CLOSED` |
| 2 | **Not-visible target** | `is not visible` |
| 3 | **Inventory / hand-state** | `PUT: No object held` or `while holding` |
| 4 | **Dirty-state blocks** | `is dirty` |
| 5 | **Sink-basin constraints** | `SinkBasin contains` |
| 6 | **Navigation / controller** | `Ran out of candidate poses` |
| 7 | **Repetitive failures** | (same action repeated 3+ times with errors—detected automatically) |
| 8 | **Reasoning / state mismatch** | (model `<think>` contradicts `Current` image or sim feedback) |

## Quick grep to find errors

```bash
grep -n "\[STEP ERROR\]" logs.txt
```

## Using parsed JSON

After running `parse_log.py`, use the `errors_by_category` field:

```python
import json
with open("copilot-reports/<experiment-id>/logs.json") as f:
    data = json.load(f)

for category, errors in data["errors_by_category"].items():
    print(f"{category}: {len(errors)} errors")
    for e in errors[:5]:  # Sample 5
        print(f"  - {e['task']} step {e['step']}: {e['error']}")
        print(f"    Config-natural: {e.get('is_config_natural', False)}")
```

## Categorization tips

- Categories 1–6 are detected automatically via patterns in `parse_log.py`.
- Category 7 (repetitive failures) is detected when the same action fails 3+ times.
- Category 8 requires manual inspection: compare `<think>` claims to `Current` image + sim feedback.
