# AsgardBench Public Release - Cleanup Plan

## Problem Statement
Clean up the internal Magmathor benchmark codebase for public release as "AsgardBench". This involves removing internal tooling, simplifying the API, fixing security issues, and preparing proper documentation.

## Key Decisions Made
- **Entry Point**: Remove `experiment_runner/`, use simplified `model_tester.py` or module-level API
- **AML Integration**: Remove all Azure ML-specific code
- **Model Providers**: Single unified OpenAI-compatible client (works with OpenAI, Azure OpenAI, OpenRouter, VLLM, etc.)
- **Agent Assets**: Remove `.github/agents/`, `.github/prompts/`, `.claude/skills/`; keep only `CLAUDE.md`
- **Scripts**: Keep only scripts useful for benchmark users
- **Test Data**: Include benchmark JSON files directly in repo (no images, no LFS needed)
- **Repository**: Clean up and release this repo directly

---

## Workplan

### Phase 1: Security & Sensitive Data ✅ COMPLETE
- [x] **Remove exposed API key** in `.env` file (deleted)
- [x] Audit all files for hardcoded secrets, subscription IDs, resource names
  - Found in: README, experiment_runner/*, Magmathor/Model/aml_*.yaml, scripts/*
  - These files will be deleted in Phase 2
- [x] Review `.gitignore` to ensure secrets aren't committed (improved patterns)
- [x] **General security review** - scan for potential vulnerabilities:
  - [x] No pickle.load or yaml.unsafe_load found
  - [x] subprocess usage is safe (internal commands, no user input)
  - [x] os.system in scenario.py uses internal paths only (low risk)
  - [x] No eval()/exec() on user input found
- [x] User-specific paths exist in files scheduled for deletion in Phase 2
  - .vscode/launch.json has one hardcoded path (will clean in Phase 3)

### Phase 2: Remove Internal Tooling & Dependencies ✅ COMPLETE
*Bulk deletion of internal-only code.*

#### 2a: Remove experiment_runner/ ✅
- [x] Delete entire `experiment_runner/` directory
- [x] Remove `run_runner.sh`

#### 2b: Remove Azure ML integration ✅
- [x] Delete `Magmathor/Model/aml_*.yaml` files (4 files)
- [x] Delete `Magmathor/Model/gpt_actor_aml.py`
- [x] Delete `.amltconfig` and `.amltignore`
- [x] Remove `Magmathor/Utils/keyvault.py` (Azure Key Vault)
- [x] Remove `Magmathor/Utils/remount_blob.sh` (Azure blob mounting)

#### 2c: Remove Copilot/Agent assets ✅
- [x] Delete `.github/agents/` directory (6 agent files)
- [x] Delete `.github/prompts/` directory (2 prompt files)
- [x] Delete `.claude/` directory (skills)
- [x] Delete empty `.github/` directory

#### 2d: Clean up scripts/ ✅
- [x] Delete `scripts/sync_tests_to_blob.py` (Azure-specific)
- [x] Delete `scripts/move_configs.py` (Azure-specific)
- [x] Delete `scripts/check_retry_logs.py` (hardcoded internal paths)
- [x] Delete `scripts/rename_exps_from_config.py` (experiment_runner related)

#### 2e: Remove misc internal files ✅
- [x] Delete `Magmathor/TODO.txt` and `Magmathor/TASK_IDEAS.txt`
- [x] Delete `Magmathor/Model/Magmathor.code-workspace`
- [x] Delete `test_push_slices_apart.py` (test file)

#### 2f: Dependencies & code cleanup ✅
- [x] Remove from `pyproject.toml`: `azure-identity`, `azure-keyvault-secrets`, `azureml-core`, `debugpy`, `rich`, `prompt-toolkit`, `httpx`, `dacite`, `csvkit`
- [x] Remove `GPTActorAML` references from `model_tester.py`
- [ ] Regenerate `uv.lock` after changes (do at end of all phases)

### Phase 3: Rename & Restructure ✅ COMPLETE
*After bulk deletions, fewer files to update*

#### 3a: Rename package ✅
- [x] Rename `Magmathor/` directory to `AsgardBench/`
- [x] Update all imports throughout codebase (sed replacement)
- [x] Update `pyproject.toml` (name, description, package name)
- [x] Update `CLAUDE.md` references from Magmathor to AsgardBench

#### 3b: Constants cleanup ✅
- [x] Update `constants.py` to remove internal paths:
  - [x] Removed `MOUNTED_STORAGE_PATH`, `USING_MOUNTED_STORAGE`, `IN_AML`, `TEST_FOLDER_NAME`
  - [x] Added configurable env vars: `ASGARDBENCH_DATA_DIR`, `ASGARDBENCH_TEST_DIR`
- [x] Updated `storage_utils.py` - removed AML-specific code
- [x] Updated `utils.py` - replaced `IN_AML` with `ASGARDBENCH_NO_COLOR` env var
- [x] Updated `model_tester.py` - replaced `IN_AML` with `ASGARDBENCH_QUIET` env var

#### 3c: VS Code settings cleanup ✅
- [x] Updated `.vscode/launch.json`:
  - [x] Changed all Magmathor -> AsgardBench
  - [x] Removed hardcoded personal path (C:/Code/...)
  - [x] Removed "Remount Blob Storage" config
  - [x] Removed "OpenRouter Cost Report" config (script doesn't exist)
  - [x] Added inputDir input variable
- [x] Cleaned `.vscode/settings.json` - removed personal color settings
- [x] Updated `.gitignore` - removed obsolete entries, updated cache path

### Phase 4: Simplify Model API ✅ COMPLETE
*Depends on Phase 2 (old actors still exist for reference while building new one)*

#### 4a: Create unified OpenAI-compatible actor ✅
- [x] Create new `openai_actor.py` - single generic client supporting all OpenAI-compatible APIs
- [x] Support environment variables:
  - `OPENAI_API_KEY` - API key for authentication
  - `OPENAI_BASE_URL` - Base URL (OpenAI, Azure, OpenRouter, VLLM, etc.)
  - `OPENAI_API_VERSION` - Optional, for Azure OpenAI
- [x] Works with: standard OpenAI, Azure OpenAI (API key), OpenRouter, VLLM, any compatible endpoint
- [x] Port prompt caching logic from OpenRouter actor:
  - Keep `split_prompt_for_caching()` utility (already in prompt_templates.py)
  - Auto-detect provider (Anthropic/Gemini) from model name or base URL
  - Add `cache_control: {type: "ephemeral"}` only for providers that need it
  - Others (OpenAI, DeepSeek, etc.) use automatic prefix caching

#### 4b: Remove specialized actors ✅
*After new actor is working*
- [x] Delete `gpt_actor.py` (has complex Azure AD auth, credential rotation)
- [x] Keep `openrouter_actor.py` for reproducibility (added deprecation note)
- [x] Delete `glm_actor.py` (specialized, can use unified client)
- [x] Delete `vllm_actor.py` (VLLM exposes OpenAI-compatible API)
- [x] Delete `qwenvl25_actor.py` (in-memory loading not needed for public benchmark)

#### 4c: Update model_tester.py ✅
- [x] Simplify to use only the new unified OpenAI actor
- [x] Removed `on_aml` parameter from ModelTester, create_model_actor, run_tests
- [x] Removed `run_metadata` parameter (was for OpenRouter analytics)
- [x] Removed `--aml` CLI argument
- [x] Removed `MODEL_IMPLEMENTATIONS` dict entirely (only one actor now)
- [x] Removed `--implementation` and `--expected-model-path` CLI arguments
- [x] Made `--model` required (no default)
- [x] Removed `create_model_actor()` - inlined OpenAIActor instantiation
- [x] Consolidated `model_path`/`model_name` into single `model` parameter
- [x] Ensure CLI is user-friendly

#### 4d: Create new .env.example ✅
- [x] Updated `.env.example` with:
  - `OPENAI_API_KEY`
  - `OPENAI_BASE_URL`
  - `OPENAI_API_VERSION` (optional)
  - AsgardBench-specific config vars

### Phase 5: Dead Code & General Cleanup ✅ COMPLETE
*After all deletions and refactoring - now we can see what's truly unused*

#### 5a: Dead code identification ✅
- [x] Check for unused imports - fixed `CloudRendering` in scenario.py
- [x] BoxPrompts reference - already removed (not found)
- [x] TODO comments reviewed - keeping as future improvement notes

#### 5b: Utils cleanup ✅
Deleted internal-only utilities:
- [x] `Utils/DataTranformations/` - internal data migration scripts
- [x] `Utils/upgrade/` - one-time upgrade scripts
- [x] `Utils/calibrate_hand_shift.py`, `Utils/review_hand_shifts.py` - internal calibration
- [x] `Utils/convert_plan_to_prompt.py`, `Utils/convert_raw.py` - training data conversion
- [x] `Utils/generate_plan_tree.py`, `Utils/make_item_cache.py` - internal tools
- [x] `Utils/remove_test_result.py`, `Utils/run_parallel_tests.sh` - internal scripts

Kept useful utilities:
- [x] `config_utils.py` - core config (used everywhere)
- [x] `json_utils.py` - JSON reading utilities
- [x] `count_plans.py` - plan statistics
- [x] `compare_images.py` - streamlit image comparison tool
- [x] `manual_control.py` - manual testing tool
- [x] `plan_monitor.py` - live monitoring tool
- [x] `display_plan_tree.py` - visualization tool

#### 5c: Scripts cleanup ✅
Deleted internal scripts:
- [x] `scripts/amulet/` - empty AML folder
- [x] `scripts/vllm/` - internal VLLM deployment
- [x] `scripts/update_test_results_candidate_poses.py` - one-time migration
- [x] `scripts/verify_plan_directories.py` - internal verification
- [x] `scripts/analyze_candidate_poses_errors.py` - internal analysis

Kept useful scripts:
- [x] `scripts/extract_model_errors.py` - debugging tool
- [x] `scripts/failure_summary.py` - results analysis
- [x] `scripts/remove_api_failures.py` - useful for users debugging model connections
- [ ] `scripts/convert_plan_to_reasoning_prompt.py` - TODO: decide if needed (incomplete script)

### Phase 6: Include Test Data ✅ COMPLETE
*Can happen anytime, but logical to do after code is stable*

- [x] Update `.gitignore` to allow benchmark data folders (plan.json only)
  - Excludes: PNG images, txt files, raw_plan.json
  - Includes: plan.json for each task
- [x] Verify data doesn't include images or large files
  - Images excluded via .gitignore patterns
- [x] **Optimize plan.json files** - strip `steps` array (not needed for evaluation)
  - Original size: ~5.8MB → Stripped size: ~309KB (94.7% reduction!)
  - Added `step_count` and `initial_pose` fields to replace steps array data
  - Updated `Plan.from_dict()` to support both full and stripped formats
  - Updated `model_tester.py` to use new `step_count` and `initial_pose` properties
- [x] **Create sanity check partition** (`Generated/magt_benchmark_sanity/`):
  - [x] 2 "turn on TV" tasks (4 steps each) for quick verification
  - [x] Copied from magt_benchmark_p6
- [ ] Add benchmark data to git (ready - run `git add Generated/magt_benchmark_*/`)
- [ ] Document data structure in README (Phase 7)

**Benchmark data structure (stripped format):**
- `Generated/magt_benchmark_p{1-6}/` - 18 tasks each (108 total)
- `Generated/magt_benchmark_sanity/` - 2 easy tasks for setup verification
- Each task folder contains `plan.json` with:
  - `name`, `task_description`: Task identification
  - `scene`: AI2-THOR scene ID (e.g., FloorPlan202)
  - `step_count`: Number of ground-truth steps
  - `initial_pose`: Agent starting position/rotation
  - `goal`: Success criteria
  - `setup_actions`, `object_setup`, `randomization`: Scene configuration

### Phase 7: Documentation
*After all code changes are complete*

#### 7a: README.md (main entry point) ✅
- [x] Rewrite `README` → `README.md` with proper structure:
  - [x] Project description & motivation (what is AsgardBench?)
  - [x] Installation instructions (uv/pip)
  - [x] Quick start guide (sanity check example)
  - [x] How to run benchmark on your model
  - [x] Docker usage section (placeholder for coming soon)
  - [x] Configuration options (ablation settings, temperature, etc.)
  - [x] Reproducibility: how to reproduce paper results
  - [x] Citation information (placeholder BibTeX)

#### 7b: Docker support ✅
- [x] Create `Dockerfile` for easy setup
- [x] Document Docker usage in README

#### 7c: Data documentation ✅
- [x] Document benchmark data structure in README (partitions, plan.json format)
- [ ] Document output data format (test_results.json schema) - optional, can add later
- [ ] Document data generation (plan_generator.py) - internal tool, not needed for users

#### 7d: Prompt DSL documentation
- [ ] Document the custom action language format - optional for v1
- [ ] Note: editing prompts not recommended for reproducibility

#### 7e: Other docs ✅
- [x] Update `CLAUDE.md` for public contributors
- [x] Add `LICENSE` file (MIT)
- [x] Add citation info in README (placeholder BibTeX)
- [ ] Add `CITATION.cff` - optional, can add when paper is published

### Phase 8: Final Validation
- [ ] Run linters (`black`, `isort`)
- [ ] Test that benchmark runs end-to-end with mock model
- [ ] Verify all imports work after rename
- [ ] Review final file list
- [ ] Regenerate `uv.lock` after all changes

---

## User Tasks (Decisions & Coworker Confirmation)

These tasks require your decision or confirmation with coworkers:

### Decisions Needed
- [ ] **Review `vllm_actor.py`** - likely remove if unified client works, but confirm
- [ ] **Review `qwenvl25_actor.py`** - keep if needed for local HuggingFace models?
- [ ] **Review remaining scripts/** - check which are useful for benchmark users
- [ ] **Review `AsgardBench/Utils/`** - identify any other internal-only utilities
- [ ] **Review `compare_images.py`** - keep if useful for users?
- [ ] **Review `streamlit_report_app.py`** - keep or remove?
- [ ] **GPU requirements** - can we remove GPU requirements for evaluation? (AI2-THOR rendering)
- [ ] **Add `CONTRIBUTING.md`?** - decide if desired

### Confirm with Coworkers
- [ ] **VS Code launch configs** - check which launch.json configs to keep
- [ ] **Git history cleanup** - may need git-filter-branch or BFG if secrets in history

---

## Files to DELETE (Summary)
```
experiment_runner/          # Entire directory
run_runner.sh
.amltconfig
.amltignore
.github/agents/             # Entire directory
.github/prompts/            # Entire directory
.claude/                    # Entire directory
.env                        # Contains secrets!

Magmathor/Model/aml_*.yaml  # 4 files
Magmathor/Model/gpt_actor_aml.py
Magmathor/Model/gpt_actor.py        # Replace with generic
Magmathor/Model/openrouter_actor.py # Replace with generic
Magmathor/Model/glm_actor.py
Magmathor/Model/Magmathor.code-workspace
Magmathor/TODO.txt
Magmathor/TASK_IDEAS.txt
Magmathor/Utils/keyvault.py
Magmathor/Utils/remount_blob.sh

scripts/sync_tests_to_blob.py
scripts/move_configs.py
scripts/check_retry_logs.py
scripts/rename_exps_from_config.py

test_push_slices_apart.py
streamlit_report_app.py     # If internal only (pending decision)
```

## Files to CREATE
```
README.md                   # Proper documentation
LICENSE                     # MIT license
Dockerfile                  # Docker setup for easy evaluation
AsgardBench/Model/openai_actor.py  # Unified OpenAI-compatible client
.env.example                # OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_API_VERSION
docs/data_generation.md     # How data generation works + how to use it
docs/prompt_dsl.md          # Prompt DSL reference (optional/advanced)
Generated/magt_benchmark_sanity/  # Sanity check partition (2 easy tasks)
```

## Notes
- The `.env` file contains an exposed OpenRouter API key - rotate if it was ever committed
- Several files contain hardcoded Azure subscription IDs and user-specific paths
- The `uv.lock` file should be regenerated after dependency changes
- Consider whether Streamlit viewer tools are useful for public users
