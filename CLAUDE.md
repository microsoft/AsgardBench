# Claude.md - Magmathor Repository Guide

## Running Python

Always use `uv run` to execute Python scripts:

```bash
uv run python <script.py>
```

## Key Paths & Constants

Important paths and constants are defined in `Magmathor/constants.py`:

- `TEST_DIR` - The test folder path (currently `20260115_Test` or mounted storage path)
- `DATASET_DIR` - Where generated data is stored (`Generated/`)
- `MOUNTED_STORAGE_PATH` - Path to mounted blob storage

## Performance Warning

The `./Test` folder (or the path defined in constants) is mounted via **blobfuse**. Large operations on this folder are **extremely slow**. Avoid:

- Listing large directories
- Bulk file operations
- Recursive searches

Prefer targeted operations on specific files when working with the Test folder.

## Project Structure

- `Magmathor/` - Main Python package
- `Generated/` - Generated dataset files
- `Test/` - Test data (blobfuse mounted)
- `experiment_runner/` - Experiment automation
- `scripts/` - Utility scripts

## Model Providers

Models are accessed through two different providers:

### Azure OpenAI (`Magmathor/Model/gpt_actor.py`)
- **Models**: `gpt-4o`, `gpt-4.1`, `gpt-5`, `gpt-5.2`, `o1`, `o3`, `o3-mini`, `o4-mini`, `Llama-4-*`, `Mistral-Large-3`
- Uses Azure AD authentication via managed identity (on AML) or Azure CLI (locally)
- Resources configured in `_init_azure_rotation()` method

### OpenRouter (`Magmathor/Model/openrouter_actor.py`)
- **Models**: `anthropic/claude-*`, `google/gemini-*`, `qwen/*`, `z-ai/glm-*`, `deepseek/*`
- Uses OpenRouter API key from Key Vault
- Model names use `__` separator in configs (e.g., `google__gemini-3-pro-preview`)

## Error Analysis Scripts

Scripts for analyzing API failures and test results:

- `scripts/extract_model_errors.py` - Extract `[MODEL ERROR]` messages from plan.json files
- `scripts/failure_summary.py` - Summarize failure patterns across benchmarks
- `scripts/check_retry_logs.py` - Search for retry warnings in logs
- `scripts/remove_api_failures.py` - Remove API_Failure tasks from results

## Common Commands

```bash
# Install dependencies
uv sync

# Run pre-commit hooks
uv run pre-commit run --all-files

# Run model evaluation
uv run python Magmathor/Model/model_tester.py --test <test_name> --model <model_path>
```
