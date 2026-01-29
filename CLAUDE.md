# Claude.md - AsgardBench Repository Guide

## Running Python

Always use `uv run` to execute Python scripts:

```bash
uv run python <script.py>
```

## Key Paths & Constants

Important paths and constants are defined in `AsgardBench/constants.py`:

- `TEST_DIR` - The test folder path (default: `./Test`)
- `DATASET_DIR` - Where generated data is stored (`Generated/`)

## Project Structure

- `AsgardBench/` - Main Python package
- `Generated/` - Generated dataset files (benchmark data)
- `Test/` - Test output data
- `scripts/` - Utility scripts

## Model Providers

Models are accessed through OpenAI-compatible APIs. Configure via environment variables:

- `OPENAI_API_KEY` - Your API key
- `OPENAI_BASE_URL` - API endpoint (OpenAI, Azure, OpenRouter, VLLM, etc.)

## Common Commands

```bash
# Install dependencies
uv sync

# Run pre-commit hooks
uv run pre-commit run --all-files

# Run model evaluation
uv run python AsgardBench/Model/model_tester.py --test <test_name> --model <model_name>
```
