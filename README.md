# AsgardBench

A benchmark for evaluating Vision-Language Models (VLMs) on embodied household tasks.

📄 **Paper:** [AsgardBench - Evaluating Visually Grounded Interactive Planning Under Minimal Feedback](https://arxiv.org/abs/2603.15888)

## Overview

AsgardBench evaluates how well VLMs can act as embodied agents completing multi-step household tasks. Given a task description (e.g., "Make coffee") and egocentric visual observations, the model must output actions to accomplish the goal.

**Key features:**
- 🏠 108 task instances across 12 task types and 3 scene types in AI2-THOR
- 👁️ Egocentric visual observations
- 🎯 Automatic success evaluation via goal checking
- 🔧 Works with any OpenAI-compatible API endpoint

For detailed methodology, ablation studies, and results, please see our paper.

## Quick Start

### Prerequisites

- [uv](https://docs.astral.sh/uv/)
- An OpenAI-compatible API endpoint (OpenAI, Azure OpenAI, OpenRouter, vLLM, etc.)
- **Linux:** X11 display or Xvfb (for AI2-THOR rendering)

### Installation

```bash
# Clone the repository
git clone https://github.com/microsoft/AsgardBench.git
cd AsgardBench

# Install dependencies
uv sync
```

### Configuration

Create a `.env` file with your API credentials:

```bash
cp .env.example .env
# Edit .env with your API key and endpoint
```

Required environment variables:
```bash
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api.openai.com/v1  # Or your endpoint
```

### Run the Sanity Check

Verify your setup with a quick 2-task sanity check:

```bash
# On Linux without a display, use xvfb-run
xvfb-run -a uv run python -m AsgardBench.Model.model_tester \
    --test magt_benchmark_sanity \
    --model gpt-4o

# On systems with a display (or macOS)
uv run python -m AsgardBench.Model.model_tester \
    --test magt_benchmark_sanity \
    --model gpt-4o
```

### Run the Full Benchmark

```bash
xvfb-run -a uv run python -m AsgardBench.Model.model_tester \
    --test magt_benchmark \
    --model gpt-4o
```

See [Docker](#docker) for containerized execution. After running, see [Results](#results) to generate reports and view scores.


## Benchmark Structure

The benchmark consists of 108 task instances across 12 task types:

| Dataset | Tasks |
|---------|-------|
| `magt_benchmark` | 108 |
| `magt_benchmark_sanity` | 2 (quick setup verification) |

For a full list of task types and their variations, see our paper.


### Data Format

Each task in `Generated/magt_benchmark*/` contains a `plan.json` with:

```jsonc
{
  "name": "task_name",
  "task_description": "Make coffee in the mug",  // task description given to the model
  "scene": "FloorPlan1",                         // AI2-THOR scene identifier
  "step_count": 25,                              // Exepected number of steps to complete the task
  "initial_pose": {                              // Agent's starting position and orientation
    "position": {"x": 0.5, "y": 0.9, "z": -1.2},
    "rotation": 90,
    "horizon": 30,
    "standing": true
  },
  "goal": {                                      // Success conditions for the task
    "goal_type": "ObjectStateGoal",
    "conditions": [...]
  },
  "setup_actions": [...],                        // Actions to initialize the scene
  "object_setup": {...},                         // Object placements and states
  "randomization": {...}                         // Randomization parameters used
}
```

## Configuration

The default configuration runs the **baseline evaluation** used in our paper. Simply specify the test set and model:

```bash
uv run python -m AsgardBench.Model.model_tester --test <test_set> --model <model>
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--test` | Test set name (`magt_benchmark` or `magt_benchmark_sanity`) | Required |
| `--model` | Model identifier | Required |
| `--temperature` | Sampling temperature | 0.0 |
| `--max_completion_tokens` | Maximum tokens for model response | 8192 |
| `--rep` | Repetition number (for multiple runs) | 1 |

Run `--help` for the full list of parameters, including ablation flags. See our paper for details on the baseline configuration and ablation methodology.


## Provider-Specific Configuration

For Anthropic and Google APIs, set `OPENAI_CACHE_CONTROL=explicit` to enable prompt caching. Other providers (OpenAI, Azure, OpenRouter, vLLM, etc.) use automatic caching by default.


## Results

Results are saved to `Test/<test_set>/<model>--<config>--<rep>/`:

- `test_results.json` - Per-task success/failure data
- `config.json` - Run configuration
- `Plans/` - Detailed execution logs per task

### Generating Reports

To generate aggregated results across all test runs:

```bash
uv run python -m AsgardBench.Model.generate_reports
```

This produces `Test/results.xlsx` with success rates, failure breakdowns, and per-plan performance statistics.


### Viewing Individual Task Executions

To inspect step-by-step execution traces with images:

```bash
uv run streamlit run AsgardBench/plan_viewer.py
```

This launches a web UI for browsing task executions, viewing agent observations, and analyzing failures.


## Docker

A Dockerfile is provided for containerized execution.


### Building the Image

```bash
docker build -t asgardbench .
```

### Running the Benchmark

```bash
# Run sanity check
docker run --rm \
    -e OPENAI_API_KEY=sk-... \
    -e OPENAI_BASE_URL=https://api.openai.com/v1 \
    asgardbench \
    --test magt_benchmark_sanity --model gpt-4o

# Run full benchmark with results saved to host
docker run --rm \
    -v $(pwd)/results:/app/Test \
    -e OPENAI_API_KEY=sk-... \
    -e OPENAI_BASE_URL=https://api.openai.com/v1 \
    asgardbench \
    --test magt_benchmark --model gpt-4o
```

### Networking Notes

When connecting to a local API server (e.g., vLLM running on the host on port `7000`), use `--network host`:

```bash
docker run --rm --network host \
    -e OPENAI_API_KEY=dummy \
    -e OPENAI_BASE_URL=http://host.docker.internal:7000/v1 \
    asgardbench \
    --test magt_benchmark_sanity --model your-model
```

## Responsible AI

For information about intended uses, limitations, and best practices, see [RESPONSIBLE_AI_FAQ.md](RESPONSIBLE_AI_FAQ.md).


## Contributing

This project welcomes contributions and suggestions. Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact opencode@microsoft.com with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.

## Citation

If you use AsgardBench in your research, please cite:

```bibtex
@article{asgardbench2026,
  title={AsgardBench: A Benchmark for Embodied Household Tasks},
  author={...},
  journal={...},
  year={2025}
}
```
