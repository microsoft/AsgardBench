# AsgardBench Public Release - Handoff Document

## Current Status: ~90% Complete

The internal "Magmathor" benchmark has been cleaned up and renamed to "AsgardBench" for public release. Most cleanup work is done; remaining items are validation and user testing.

---

## What's Been Done ✅

### Phase 1: Security & Sensitive Data
- Removed exposed API keys from `.env`
- Audited and removed hardcoded secrets, subscription IDs, resource names
- Cleaned user-specific paths
- Created `.env.example` with generic OpenAI-compatible config

### Phase 2: Remove Internal Tooling
- Deleted `experiment_runner/`, `run_runner.sh`
- Deleted all Azure ML files (`.amltconfig`, `.amltignore`, `aml_*.yaml`, `gpt_actor_aml.py`)
- Deleted `.github/agents/`, `.github/prompts/`, `.claude/skills/`
- Kept `CLAUDE.md` (updated for public contributors)

### Phase 3: Rename & Restructure
- Renamed `Magmathor/` → `AsgardBench/`
- Updated all imports throughout codebase
- Updated `pyproject.toml` (name, description, package name)

### Phase 4: Simplify Model API
- Created unified `AsgardBench/Model/openai_actor.py`:
  - Works with any OpenAI-compatible endpoint (OpenAI, Azure, OpenRouter, vLLM, etc.)
  - Environment variables: `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_API_VERSION`
  - Auto-detects provider for prompt caching (Anthropic/Gemini need explicit cache_control)
- Removed specialized actors: `gpt_actor.py`, `glm_actor.py`, `vllm_actor.py`, `qwenvl25_actor.py`
- Kept `openrouter_actor.py` with deprecation note (for reproducibility of paper results)
- Simplified `model_tester.py`: consolidated to single `--model` parameter

### Phase 5: Dead Code Cleanup
- Deleted internal utilities: `DataTransformations/`, `upgrade/`, calibration scripts, `keyvault.py`
- Deleted internal scripts: `amulet/`, `vllm/`, various migration scripts
- Kept useful utilities: `config_utils.py`, `json_utils.py`, `compare_images.py`, etc.
- Kept useful scripts: `extract_model_errors.py`, `failure_summary.py`, `remove_api_failures.py`

### Phase 6: Include Test Data
- Updated `.gitignore` to allow `Generated/magt_benchmark/` and `Generated/magt_benchmark_sanity/`
- **Stripped plan.json files**: removed `steps` array (not needed for evaluation)
  - Added `step_count` and `initial_pose` as top-level fields
  - Size reduction: 5.8MB → 309KB (94.7% smaller!)
- **Consolidated partitions**: merged `magt_benchmark_p1` through `p6` into single `magt_benchmark/`
- Created `magt_benchmark_sanity/` with 2 easy "turn on TV" tasks for quick setup verification

### Phase 7: Documentation
- Created comprehensive `README.md`:
  - Quick start guide, configuration options, provider setup (OpenAI, Azure, OpenRouter, vLLM)
  - Benchmark structure: 108 tasks across 29 scenes
  - Task breakdown: Cooking (36), Distribution (27), Put away (21), Coffee (9), Table setting (9), Cleaning (3), TV (3)
  - Docker usage instructions
  - TODO(andrea) placeholder for paper link
- Created `LICENSE` (MIT)
- Created `Dockerfile`
- Updated `CLAUDE.md` for public contributors

---

## What's Left To Do 📋

### Phase 8: Final Validation
- [ ] Run linters: `uv run pre-commit run --all-files`
- [ ] Test imports work: `uv run python -c "from AsgardBench.Model.model_tester import run_tests"`
- [ ] **Test end-to-end benchmark run** on a fresh machine
- [ ] Verify Docker build works: `docker build -t asgardbench .`
- [ ] Review final file list for anything missed

### User Tasks (Andrea)
- [ ] Stage new benchmark data: `git add Generated/magt_benchmark/`
- [ ] Fill in paper link in README.md (search for `TODO(andrea)`)
- [ ] Fill in citation info in README.md
- [ ] Final review and commit
- [ ] Test on fresh machine without existing environment

---

## Key Files

| File | Purpose |
|------|---------|
| `AsgardBench/Model/model_tester.py` | Main entry point for running benchmark |
| `AsgardBench/Model/openai_actor.py` | Unified OpenAI-compatible client |
| `AsgardBench/Model/openrouter_actor.py` | Original actor (kept for reproducibility) |
| `AsgardBench/plan.py` | Plan data structure (supports stripped format) |
| `Generated/magt_benchmark/` | 108 benchmark tasks (plan.json files) |
| `Generated/magt_benchmark_sanity/` | 2 quick sanity check tasks |
| `README.md` | Main documentation |
| `.env.example` | Template for environment config |

---

## Quick Commands

```bash
# Install dependencies
uv sync

# Run linters
uv run pre-commit run --all-files

# Run sanity check (2 tasks, quick)
uv run python -m AsgardBench.Model.model_tester --test magt_benchmark_sanity --model gpt-4o

# Run full benchmark (108 tasks)
uv run python -m AsgardBench.Model.model_tester --test magt_benchmark --model gpt-4o

# View results
uv run python -m AsgardBench.Model.model_tester --test magt_benchmark --model gpt-4o --print-results
```

---

## Environment Variables

```bash
# Required
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api.openai.com/v1  # Or Azure/OpenRouter/vLLM endpoint

# Optional (Azure OpenAI only)
OPENAI_API_VERSION=2024-02-15-preview
```

---

## Architecture Notes

### Stripped Plan Format
The benchmark uses a "stripped" plan.json format that excludes the `steps` array (the reference solution). This is intentional:
- The `steps` array is only needed for replay/debugging, not evaluation
- Evaluation checks goal conditions, not step-by-step matching
- Reduces data size by ~95%

The `Plan` class in `AsgardBench/plan.py` handles both full and stripped formats via properties:
- `plan.step_count` - returns count from either format
- `plan.initial_pose` - returns pose from either format

### Provider Detection for Caching
The `OpenAIActor` auto-detects provider from model name or base URL:
- Anthropic (claude-*): Adds `cache_control: {type: "ephemeral"}` to system messages
- Google (gemini-*): Adds cache_control to system messages
- Others (OpenAI, DeepSeek, etc.): Uses automatic prefix caching (no explicit markup needed)

---

## Potential Issues to Watch For

1. **AI2-THOR rendering**: Requires display (X server or virtual framebuffer). Docker uses `xvfb-run`.
2. **First run downloads**: AI2-THOR downloads assets on first run (~2GB). May take time.
3. **GPU not required**: Benchmark runs on CPU (slower but works).
4. **Results directory**: Created at `Test/<test_set>/<model>/` - ensure write permissions.

---

## Files NOT in Git Yet

The `Generated/magt_benchmark/` folder with 108 tasks needs to be staged:
```bash
git add Generated/magt_benchmark/
git add Generated/magt_benchmark_sanity/
```

These are currently untracked but configured in `.gitignore` to be allowed.
