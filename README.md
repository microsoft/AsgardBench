# AsgardBench

A benchmark for evaluating Vision-Language Models (VLMs) on embodied household tasks.

<!-- TODO(andrea): Add paper link when available -->
📄 **Paper:** [Coming Soon]()

## Overview

AsgardBench evaluates how well VLMs can act as embodied agents completing multi-step household tasks. Given a task description (e.g., "Make coffee") and egocentric visual observations, the model must output actions to accomplish the goal.

**Key features:**
- 🏠 108 household tasks across 29 scenes in AI2-THOR
- 👁️ Egocentric visual observations
- 🎯 Automatic success evaluation via goal checking
- 🔧 Works with any OpenAI-compatible API endpoint

For detailed methodology, ablation studies, and results, please see our paper.

## Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- An OpenAI-compatible API endpoint (OpenAI, Azure OpenAI, OpenRouter, vLLM, etc.)

### Installation

```bash
# Clone the repository
git clone https://github.com/microsoft/AsgardBench.git
cd AsgardBench

# Install dependencies
uv sync

# Or with pip
pip install -e .
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
uv run python -m AsgardBench.Model.model_tester \
    --test magt_benchmark_sanity \
    --model gpt-4o
```

### Run the Full Benchmark

```bash
uv run python -m AsgardBench.Model.model_tester \
    --test magt_benchmark \
    --model gpt-4o
```

## Benchmark Structure

The benchmark consists of 108 tasks plus a sanity check:

| Dataset | Tasks |
|---------|-------|
| `magt_benchmark` | 108 |
| `magt_benchmark_sanity` | 2 (quick setup verification) |

| Task Type | Count |
|-----------|-------|
| Cooking | 36 |
| Object distribution | 27 |
| Put away items | 21 |
| Coffee making | 9 |
| Table setting | 9 |
| Cleaning | 3 |
| Turn on TV | 3 |

### Data Format

Each task in `Generated/magt_benchmark_*/` contains a `plan.json` with:

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
| `--rep` | Repetition number (for multiple runs) | 1 |

> **Note:** Additional flags for ablation studies (`--text_only`, `--feedback_type`, etc.) are available for research purposes. See `--help` for details, or refer to our paper for ablation methodology.

## Using Different Model Providers

### OpenAI

```bash
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
```

### Azure OpenAI

```bash
OPENAI_API_KEY=your-azure-key
OPENAI_BASE_URL=https://your-resource.openai.azure.com
OPENAI_API_VERSION=2024-02-15-preview
```

### OpenRouter

```bash
OPENAI_API_KEY=sk-or-...
OPENAI_BASE_URL=https://openrouter.ai/api/v1
```

### Local vLLM Server

```bash
OPENAI_API_KEY=dummy
OPENAI_BASE_URL=http://localhost:8000/v1
```

## Results

Results are saved to `Test/<test_set>/<model>/`:

- `test_results.json` - Per-task success/failure data
- `config.json` - Run configuration
- `Plans/` - Detailed execution logs per task

### Viewing Results

```bash
# Print results summary
uv run python -m AsgardBench.Model.model_tester \
    --test magt_benchmark \
    --model gpt-4o \
    --print-results
```

## Reproducibility

For detailed reproducibility instructions and the exact configurations used in our experiments, please refer to our paper.

The original OpenRouter actor used for paper experiments is preserved in `AsgardBench/Model/openrouter_actor.py` for reference.

## Docker

A Dockerfile is provided for containerized execution:

```bash
# Build the image
docker build -t asgardbench .

# Run sanity check
docker run -e OPENAI_API_KEY=sk-... asgardbench \
    --test magt_benchmark_sanity --model gpt-4o

# Run full benchmark with results volume
docker run -v $(pwd)/results:/app/Test -e OPENAI_API_KEY=sk-... asgardbench \
    --test magt_benchmark --model gpt-4o
```

## Citation

If you use AsgardBench in your research, please cite:

```bibtex
@article{asgardbench2025,
  title={AsgardBench: A Benchmark for Embodied Household Tasks},
  author={...},
  journal={...},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions! Please see our contributing guidelines before submitting PRs.

## Acknowledgments

- [AI2-THOR](https://ai2thor.allenai.org/) for the simulation environment
- [OpenAI](https://openai.com/) for the API client library
