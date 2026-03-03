# Claude.md - AsgardBench Repository Guide

## Running Python

Always use `uv run` to execute Python scripts:

```bash
uv run python <script.py>
```

## Key Paths & Constants

Important paths and constants are defined in `AsgardBench/constants.py`:

- `TEST_DIR` - Test output folder (default: `./Test`, configurable via `ASGARDBENCH_TEST_DIR`)
- `DATASET_DIR` - Benchmark data folder (default: `./Generated`, configurable via `ASGARDBENCH_DATA_DIR`)

## Project Structure

```
AsgardBench/
├── AsgardBench/           # Main Python package
│   ├── Model/             # Model testing & evaluation
│   │   ├── model_tester.py    # Main entry point for evaluation
│   │   ├── openai_actor.py    # Unified OpenAI-compatible client
│   │   └── prompt_templates.py # Prompt generation
│   ├── Utils/             # Utility tools
│   └── ...                # Core simulation code
├── Generated/             # Benchmark data (plan.json files)
│   ├── magt_benchmark/    # Full benchmark (108 tasks)
│   └── magt_benchmark_sanity/ # Quick sanity check (2 tasks)
├── Test/                  # Output directory for results
└── scripts/               # Analysis & debugging scripts
```

## Model Providers

Models are accessed through OpenAI-compatible APIs. Configure via environment variables:

- `OPENAI_API_KEY` - Your API key
- `OPENAI_BASE_URL` - API endpoint (OpenAI, Azure, OpenRouter, vLLM, etc.)
- `OPENAI_API_VERSION` - API version (required for Azure OpenAI)

## Common Commands

```bash
# Install dependencies
uv sync

# Run pre-commit hooks (formatting)
uv run pre-commit run --all-files

# Run sanity check
uv run python -m AsgardBench.Model.model_tester --test magt_benchmark_sanity --model gpt-4o

# Run full benchmark
uv run python -m AsgardBench.Model.model_tester --test magt_benchmark --model gpt-4o

# View results
uv run python -m AsgardBench.Model.generate_reports
```

## Key Files for Development

- `AsgardBench/Model/model_tester.py` - Main evaluation loop
- `AsgardBench/Model/openai_actor.py` - API client (supports all OpenAI-compatible endpoints)
- `AsgardBench/plan.py` - Plan data structure
- `AsgardBench/scenario.py` - AI2-THOR environment wrapper
- `AsgardBench/player.py` - Agent execution logic
- `AsgardBench/goal.py` - Success condition checking
