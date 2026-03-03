# AsgardBench Public Release - Cleanup Plan

## Status: Core Cleanup Complete ✅

All major cleanup phases (1-7) are done and committed. The codebase has been renamed from Magmathor → AsgardBench, internal tooling removed, model API simplified, test data included, and documentation created.

---

## Completed Work

- **Phase 1**: Removed secrets, API keys, hardcoded Azure IDs, user-specific paths
- **Phase 2**: Removed experiment_runner/, AML files, agent assets, internal scripts
- **Phase 3**: Renamed Magmathor → AsgardBench, updated all imports, cleaned constants
- **Phase 4**: Created unified `openai_actor.py`, removed specialized actors, simplified model_tester.py
- **Phase 5**: Deleted dead code, internal utilities, one-time migration scripts
- **Phase 6**: Included stripped plan.json files (94.7% size reduction), consolidated into single `magt_benchmark/`
- **Phase 7**: Created README.md, LICENSE, Dockerfile, updated CLAUDE.md, .env.example

---

## Remaining Items

### Validation
- [ ] Run linters: `uv run pre-commit run --all-files`
- [ ] Verify Docker build: `docker build -t asgardbench .`
- [ ] Regenerate `uv.lock` if dependencies change

### Andrea TODOs
- [ ] Fill in paper link in README.md (search for `TODO(andrea)`)
- [ ] Fill in citation BibTeX in README.md
- [ ] Decide on `scripts/convert_plan_to_reasoning_prompt.py` (keep/expand/remove?)
- [ ] Git history cleanup - consider BFG or filter-branch if secrets were ever in history
- [ ] Final review before publishing to new public repo

### Parallelization Documentation
- [ ] Add docs (README or separate guide) explaining how to parallelize evaluation:
  - Split `magt_benchmark/` into multiple subsets (e.g., `magt_benchmark_a/`, `magt_benchmark_b/`)
  - Run one job per subset: `--test magt_benchmark_a`, `--test magt_benchmark_b`
  - `generate_reports.py` auto-discovers all test set dirs and aggregates results across them
- [ ] Verify `generate_reports.py` correctly aggregates across split test sets (code review says yes, but should test)
- [ ] ~~Fix `--print-results` flag in model_tester.py~~ — removed (was dead code)

### Nice-to-Have (Future)
- [ ] Document output data format (test_results.json schema)
- [ ] Document custom action language / prompt DSL
- [ ] Report generator overhaul (simplify for public users)
- [ ] Improve runtime logging (progress bar, --verbose/--quiet flags)
- [ ] Add CITATION.cff when paper is published
- [ ] Add CONTRIBUTING.md
