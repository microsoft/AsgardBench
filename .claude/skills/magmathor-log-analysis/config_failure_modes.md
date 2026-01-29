# Config-Based Failure Mode Analysis

This document describes the "natural" or expected failure modes for different `EvaluationConfig` settings. When analyzing experiment logs, use this to determine whether failures are **expected given the configuration** versus **unexpected/anomalous**.

---

## Quick Reference: Config Impact on Failures

| Config Setting | Primary Impact | Expected Failure Rate |
|----------------|----------------|----------------------|
| `text_only=True` | No visual grounding | High - spatial/visual tasks |
| `feedback_type=none` | No error recovery | Medium - repeated errors |
| `previous_image=none` | No temporal context | Medium - state change tracking |
| `use_memory=False` | Stateless reasoning | High - multi-step tasks |
| `full_steps=True` | Plan all at once | High - long sequences |
| `include_common_sense=False` | Missing domain rules | Medium - domain-specific errors |
| `hand_transparency=0` | Hand occlusion | Low-Medium - object identification |

---

## 1. `text_only` (bool)

When `True`, the model receives only text descriptions—no images are sent.

### Natural Failure Modes

| Failure Pattern | Description | Log Indicators |
|-----------------|-------------|----------------|
| **Spatial confusion** | Model cannot determine object positions, distances, or relative locations | Actions target wrong objects; "not visible" errors despite object being present |
| **Object misidentification** | Cannot distinguish between multiple similar objects | Picks wrong instance (e.g., wrong Apple when multiple exist) |
| **State blindness** | Cannot verify object states (open/closed, clean/dirty) | Toggles already-toggled objects; cleans already-clean items |
| **Navigation failures** | Cannot judge reachability or facing direction | Repeated "too far" or "not facing" errors |
| **Receptacle estimation errors** | Cannot see if receptacle is full or has space | Failed placements due to space constraints |

### Example Error Patterns in Logs

```
# Spatial confusion
ERROR: Object Fridge_abc is not visible from current position
# (Model assumed position based on text, but was wrong)

# State blindness
Action: toggle_on(Microwave_123)
Result: Microwave is already on
# (Model couldn't see the ON indicator)

# Object misidentification
Action: pickup(Apple_456)
Result: Success
# (But this was the wrong apple - the task needed Apple_789)
```

### Identification Criteria

Check `config.json` for:
```json
"text_only": true
```

If `text_only=true`, consider spatial/visual failures as **expected** rather than model capability issues.

---

## 2. `feedback_type` (FeedbackType: none/simple/detailed)

Controls whether the model receives error feedback from the simulator after failed actions.

### `feedback_type: none`

| Failure Pattern | Description | Log Indicators |
|-----------------|-------------|----------------|
| **Repeated identical errors** | Model retries same failed action | Same action appears multiple times consecutively |
| **No error adaptation** | Model doesn't adjust strategy after failure | Pattern of try → fail → same approach |
| **Cascading failures** | Initial error causes downstream failures | Error chain with no recovery attempts |

### `feedback_type: simple`

| Failure Pattern | Description | Log Indicators |
|-----------------|-------------|----------------|
| **Partial recovery** | Model knows something failed but not why | Retries with minor variations that still fail |
| **Misinterpreted feedback** | Model changes wrong aspect of approach | Switches target when should change action type |

### `feedback_type: detailed`

With detailed feedback, most error-recovery failures are **unexpected** and indicate model reasoning issues rather than config limitations.

### Example Error Patterns

```
# feedback_type=none - repeated errors
Step 5: pickup(Knife_123) → FAIL (too far)
Step 6: pickup(Knife_123) → FAIL (too far)  # No adaptation
Step 7: pickup(Knife_123) → FAIL (too far)  # Still no adaptation

# feedback_type=simple - partial recovery
Step 5: pickup(Knife_123) → FAIL
Step 6: pickup(Fork_456) → FAIL  # Changed object instead of approaching
```

### Identification Criteria

```json
"feedback_type": "none"  // Repeated errors are expected
"feedback_type": "simple"  // Some recovery expected
"feedback_type": "detailed"  // Full recovery expected
```

---

## 3. `previous_image` (PreviousImageType: none/color/grayscale)

Controls whether the model sees the previous frame for temporal context.

### `previous_image: none`

| Failure Pattern | Description | Log Indicators |
|-----------------|-------------|----------------|
| **Action verification failures** | Cannot confirm if last action succeeded | Redundant actions or missed failures |
| **State change blindness** | Cannot track what changed between steps | Repeats completed actions; misses new problems |
| **Lost object tracking** | Cannot follow objects that moved | References objects at old locations |
| **Animation/transition misses** | Cannot see in-progress state changes | Interrupts ongoing actions |

### `previous_image: grayscale`

| Failure Pattern | Description | Log Indicators |
|-----------------|-------------|----------------|
| **Color-based state confusion** | Cannot use color cues for state | Misses "cooked" (brown) vs "raw" states |
| **Subtle change detection** | Harder to spot minor differences | Misses small state changes |

### `previous_image: color`

With full color previous images, temporal tracking failures are **unexpected**.

### Example Error Patterns

```
# previous_image=none - can't verify success
Step 3: open(Fridge_123) → Success
Step 4: open(Fridge_123) → FAIL (already open)
# Model couldn't see it was already open

# previous_image=none - lost tracking
Step 5: pickup(Apple_123) from CounterTop
Step 6: find(Apple_123)  # Forgot it's already in hand
```

### Identification Criteria

```json
"previous_image": "none"  // Temporal tracking issues expected
"previous_image": "grayscale"  // Color-based issues expected
"previous_image": "color"  // Full temporal context available
```

---

## 4. `use_memory` (bool)

Controls whether conversation context persists between steps.

### `use_memory: False`

| Failure Pattern | Description | Log Indicators |
|-----------------|-------------|----------------|
| **Progress loss** | Forgets what steps are already done | Repeats completed subtasks |
| **Strategy inconsistency** | Different approach each step | Contradictory action sequences |
| **Context reference errors** | References unknown prior actions | "As I mentioned" with no prior mention |
| **Multi-step task failures** | Cannot maintain plan across steps | Complex tasks fail more than simple ones |

**Note**: Goal forgetting is NOT a config-natural failure since the current goal is always provided in the system prompt.

### Example Error Patterns

```
# use_memory=false - progress loss
Step 3: put(Egg, Pan) → Success
Step 4: pickup(Egg)  # Undoing previous step
Step 5: put(Egg, Pan)  # Re-doing what was done

# use_memory=false - strategy inconsistency
Step 1: Looking for Knife to slice Bread
Step 2: Looking for Toaster (skipped slicing!)
```

### Identification Criteria

```json
"use_memory": false  // Stateless - every step is independent
```

When `use_memory=false`, expect **significantly higher failure rates on multi-step tasks** (3+ steps). Single-step tasks should be less affected.

---

## 5. `full_steps` (bool)

Controls whether model generates a complete action plan vs single next action.

### `full_steps: True`

| Failure Pattern | Description | Log Indicators |
|-----------------|-------------|----------------|
| **Early plan errors compound** | Mistake in step 1 invalidates steps 2-N | First few steps succeed, then cascade failures |
| **State prediction failures** | Plan assumes states that don't occur | Actions fail because preconditions not met |
| **Over-planning** | Plans too many steps for simple tasks | Unnecessarily complex action sequences |
| **Rigidity** | Cannot adapt when plan fails | Continues invalid plan instead of replanning |
| **Long sequence degradation** | Quality drops in later planned steps | Early steps correct, late steps nonsensical |

### `full_steps: False`

Single-step mode is more adaptive but may lack coherence. Failures here are **unexpected** to be plan-related.

### Example Error Patterns

```
# full_steps=true - cascade from early error
Planned: [open(Fridge), pickup(Egg), close(Fridge), put(Egg,Pan), toggle_on(Burner)]
Step 1: open(Fridge) → FAIL (already open)
Step 2: pickup(Egg) → Success
Step 3: close(Fridge) → Success
Step 4: put(Egg, Pan) → FAIL (Egg fell and broke when fridge state was wrong)

# full_steps=true - state prediction failure
Planned: [pickup(Knife), slice(Bread), put(Knife,Drawer)]
Reality: Bread was already sliced → slice action fails
```

### Identification Criteria

```json
"full_steps": true  // Planning failures expected
"full_steps": false  // Step-by-step, more adaptive
```

---

## 6. `include_common_sense` (bool)

Controls whether common-sense rules are included in prompts.

### `include_common_sense: False`

| Failure Pattern | Description | Log Indicators |
|-----------------|-------------|----------------|
| **Physics violations** | Actions that violate physical constraints | Putting large object in small container |
| **Prerequisite skipping** | Missing necessary precursor actions | Trying to slice without knife in hand |
| **Affordance errors** | Wrong action for object type | Trying to open non-openable objects |
| **Sequence violations** | Wrong order of operations | Cook before slice when should slice first |
| **Domain-specific errors** | Missing kitchen/household knowledge | Using wrong appliance for task |

### Example Error Patterns

```
# include_common_sense=false - prerequisite skip
Action: slice(Tomato)
Result: FAIL - No knife in hand

# include_common_sense=false - affordance error
Action: open(CounterTop)
Result: FAIL - CounterTop is not openable

# include_common_sense=false - physics violation
Action: put(Pan, Cup)
Result: FAIL - Pan doesn't fit in Cup
```

### Identification Criteria

```json
"include_common_sense": false  // Domain errors expected
"include_common_sense": true  // Domain errors are model failures
```

---

## 7. `hand_transparency` (int 0-100)

Controls visibility of hand overlay in images. 0 = fully visible hand (occludes objects), 100 = fully transparent (invisible hand).

### `hand_transparency: 0` (Opaque Hand)

| Failure Pattern | Description | Log Indicators |
|-----------------|-------------|----------------|
| **Occluded object errors** | Cannot see objects behind hand | Fails to identify held object; wrong object references |
| **Hand state confusion** | Cannot tell if hand is empty/full | pickup when holding; put when empty |
| **Target occlusion** | Cannot see interaction target | Misses objects directly in front |

### `hand_transparency: 100` (Invisible Hand)

| Failure Pattern | Description | Log Indicators |
|-----------------|-------------|----------------|
| **Hand state uncertainty** | No visual cue for held objects | Model unsure if holding anything |
| **Floating object confusion** | Held objects appear to float | May not recognize object is in hand |

### Moderate Values (30-70)

Generally provide good balance. Failures at these levels are less expected.

### Example Error Patterns

```
# hand_transparency=0 - occlusion
Action: pickup(Apple_123)  # Apple behind hand, model sees Egg
Result: pickup(Egg_456) → unintended object

# hand_transparency=0 - hand state confusion
Current state: Holding Knife
Action: pickup(Fork)  # Model thinks hand is empty
Result: FAIL - Hand is not empty
```

### Identification Criteria

```json
"hand_transparency": 0  // High occlusion - visual errors expected
"hand_transparency": 100  // No hand cue - state confusion possible
"hand_transparency": 50  // Balanced - fewer expected failures
```

---

## Combined Configuration Analysis

When multiple limiting configs are combined, failure rates compound. Use this matrix:

| Config Combination | Expected Difficulty | Notes |
|--------------------|---------------------|-------|
| `text_only + use_memory=false` | **Very High** | No visual grounding AND no context |
| `feedback_type=none + full_steps` | **High** | Cannot recover from plan errors |
| `previous_image=none + feedback_type=none` | **High** | Cannot track state or errors |
| `text_only + full_steps` | **Very High** | Blind planning |
| All defaults | **Baseline** | Expected performance level |

---

## Log Analysis Decision Tree

```
1. Load config.json for the experiment
2. For each failure in logs:
   a. Identify failure type (spatial, state, temporal, planning, domain)
   b. Check relevant config setting
   c. If config explains failure → mark as "expected/natural"
   d. If config should support success → mark as "unexpected/investigate"
3. Calculate:
   - Expected failure rate given config
   - Actual failure rate
   - Anomaly ratio (actual/expected)
```

---

## Config Suffix Decoding

Experiment folders use encoded config suffixes. Reference:

| Code | Field | Values |
|------|-------|--------|
| T | text_only | 0=images, 1=text only |
| F | feedback_type | n=none, s=simple, d=detailed |
| H | hand_transparency | 00-99 (2-digit) |
| C | include_common_sense | 0=no, 1=yes |
| P | prompt_version | 1=v1, 2=v2 |
| I | previous_image | 0=none, 1=color, 2=grayscale |
| R | use_memory | 0=no, 1=yes |
| S | full_steps | 0=single, 1=full |
| E | temperature | 00-99 (value×100) |
| M | max_completion_tokens | token count |

Example: `T0_Fn_H00_C1_P2_I1_R1_S0_E60_M4096`
- Images enabled, no feedback, no hand transparency, common sense on, prompt v2, color previous image, memory on, single steps, temp 0.6
