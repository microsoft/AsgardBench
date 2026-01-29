"""Data models for experiment configuration and YAML parsing."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import Enum
from itertools import product
from pathlib import Path
from typing import Any


class RunMode(str, Enum):
    """Where an experiment should run."""

    LOCAL = "local"  # Run locally via experiment_runner
    AML = "aml"  # Submit to AML via Amulet


import yaml

# Add parent directory to path to import Magmathor modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from Magmathor.constants import FeedbackType, PreviousImageType, PromptVersion
from Magmathor.Utils.config_utils import EvaluationConfig


def _default_eval_config() -> EvaluationConfig:
    """Create default EvaluationConfig instance."""
    return EvaluationConfig()


@dataclass
class ExperimentSettings:
    """Settings for a single experiment configuration.

    Uses composition with EvaluationConfig to avoid field duplication.
    Experiment-specific fields (model, replicates, etc.) are defined here,
    while evaluation parameters are delegated to eval_config.
    """

    # Experiment-specific fields (not in EvaluationConfig)
    replicates: int = 1
    model: str = ""
    model_path: str = ""
    amlt_job_names: list[str] = field(default_factory=list)
    # Runtime-only field, not from YAML
    vllm_endpoints: list[str] = field(default_factory=list)

    # Evaluation config - contains all model evaluation parameters
    eval_config: EvaluationConfig = field(default_factory=_default_eval_config)

    # --- Convenience properties to access eval_config fields directly ---
    @property
    def text_only(self) -> bool:
        return self.eval_config.text_only

    @property
    def feedback_type(self) -> FeedbackType:
        return self.eval_config.feedback_type

    @property
    def hand_transparency(self) -> int:
        return self.eval_config.hand_transparency

    @property
    def include_common_sense(self) -> bool:
        return self.eval_config.include_common_sense

    @property
    def prompt_version(self) -> PromptVersion:
        return self.eval_config.prompt_version

    @property
    def previous_image(self) -> PreviousImageType:
        return self.eval_config.previous_image

    @property
    def use_memory(self) -> bool:
        return self.eval_config.use_memory

    @property
    def temperature(self) -> float:
        return self.eval_config.temperature

    @property
    def max_completion_tokens(self) -> int:
        return self.eval_config.max_completion_tokens

    def merge_with(self, base: ExperimentSettings) -> ExperimentSettings:
        """Merge with base settings. Self overrides base for non-default values.

        Uses reflection to automatically handle all fields, so new fields added
        to EvaluationConfig are automatically supported.
        """
        from dataclasses import fields as dataclass_fields

        # Create default instances for comparison
        default_settings = ExperimentSettings()
        default_eval = EvaluationConfig()

        # Helper to merge a single value
        def merge_value(self_val: Any, base_val: Any, default_val: Any) -> Any:
            # For Enums (check first since str-based enums pass isinstance str check)
            if isinstance(default_val, Enum):
                return self_val if self_val != default_val else base_val
            # For strings, use self if not empty
            if isinstance(default_val, str):
                return self_val if self_val else base_val
            # For lists, use self if not empty
            if isinstance(default_val, list):
                return self_val if self_val else base_val
            # For other types, use self if different from default
            return self_val if self_val != default_val else base_val

        # Merge experiment-specific fields using reflection
        merged_settings_kwargs: dict[str, Any] = {}
        for f in dataclass_fields(ExperimentSettings):
            if f.name == "eval_config":
                continue  # Handle separately
            self_val = getattr(self, f.name)
            base_val = getattr(base, f.name)
            default_val = getattr(default_settings, f.name)
            merged_settings_kwargs[f.name] = merge_value(
                self_val, base_val, default_val
            )

        # Merge eval_config fields using reflection
        merged_eval_kwargs: dict[str, Any] = {}
        for f in dataclass_fields(EvaluationConfig):
            self_val = getattr(self.eval_config, f.name)
            base_val = getattr(base.eval_config, f.name)
            default_val = getattr(default_eval, f.name)
            merged_eval_kwargs[f.name] = merge_value(self_val, base_val, default_val)

        merged_settings_kwargs["eval_config"] = EvaluationConfig(**merged_eval_kwargs)
        return ExperimentSettings(**merged_settings_kwargs)

    def get_output_suffix(self) -> str:
        """Generate output suffix using the EvaluationConfig directly."""
        return self.eval_config.get_output_suffix()

    def build_command_args(self, test_name: str, experiment_name: str) -> list[str]:
        """Build command line arguments for model_tester.py."""
        # For vLLM, use endpoint URLs instead of literal "vllm"
        model_arg = self.model
        if self.model == "vllm" and self.vllm_endpoints:
            model_arg = ",".join(self.vllm_endpoints)

        # Extract rep number from experiment_name (format: {name}--{suffix}--rep{n})
        rep_number = 1
        if "--rep" in experiment_name:
            import re

            match = re.search(r"--rep(\d+)$", experiment_name)
            if match:
                rep_number = int(match.group(1))

        args = [
            "run",
            "python",
            "Magmathor/Model/model_tester.py",
            "--model",
            model_arg,
            "--test",
            test_name,
            "--model-name",
            experiment_name.split("--")[0],
            "--rep",
            str(rep_number),
        ]

        # Add expected-model-path if set
        if self.model_path:
            args.extend(["--expected-model-path", self.model_path])

        if self.model == "vllm":
            args.extend(["--implementation", "vllm"])

        # Add boolean flags from eval_config
        if self.eval_config.text_only:
            args.append("--text_only")
        if self.eval_config.include_common_sense:
            args.append("--include_common_sense")
        if self.eval_config.use_memory:
            args.append("--use_memory")
        if self.eval_config.full_steps:
            args.append("--full_steps")

        args.extend(["--previous_image", self.eval_config.previous_image])

        # Add enum values from eval_config
        args.extend(["--feedback_type", self.eval_config.feedback_type.value])
        args.extend(["--prompt_version", self.eval_config.prompt_version.value])

        # Add numeric values from eval_config
        args.extend(["--hand_transparency", str(self.eval_config.hand_transparency)])
        args.extend(["--temperature", str(self.eval_config.temperature)])
        args.extend(
            ["--max_completion_tokens", str(self.eval_config.max_completion_tokens)]
        )

        return args


@dataclass
class ExperimentConfig:
    """A single experiment configuration (after expansion)."""

    name: str = ""
    settings: ExperimentSettings | None = None
    completed: bool = False
    run_mode: RunMode = RunMode.LOCAL  # Where to run: local or aml


@dataclass
class ExperimentCatalogue:
    """Collection of experiments with base settings."""

    base_settings: ExperimentSettings = field(default_factory=ExperimentSettings)
    experiments: list[ExperimentConfig] = field(default_factory=list)


# --- YAML parsing with dacite and list expansion ---

import dacite

# Dacite config for parsing - handles enum case-insensitivity
_DACITE_CONFIG = dacite.Config(
    cast=[FeedbackType, PromptVersion, PreviousImageType, int, float, bool],
    check_types=False,  # Allow flexible type coercion
)


def _get_eval_field_names() -> list[str]:
    """Get field names from EvaluationConfig using reflection."""
    from dataclasses import fields as dataclass_fields

    return [f.name for f in dataclass_fields(EvaluationConfig)]


def _settings_from_dict(data: dict[str, Any]) -> ExperimentSettings:
    """Create ExperimentSettings from a flat dict using dacite.

    The dict can contain both ExperimentSettings fields (model, replicates, etc.)
    and EvaluationConfig fields (temperature, feedback_type, etc.) - they'll be
    automatically separated and nested correctly.
    """
    eval_field_names = set(_get_eval_field_names())

    eval_data = {}
    settings_data = {}

    for key, value in data.items():
        if key in eval_field_names:
            eval_data[key] = value
        else:
            settings_data[key] = value

    # Parse eval_config with dacite
    eval_config = dacite.from_dict(
        data_class=EvaluationConfig,
        data=eval_data,
        config=_DACITE_CONFIG,
    )

    # Build ExperimentSettings
    settings_data["eval_config"] = eval_config
    return dacite.from_dict(
        data_class=ExperimentSettings,
        data=settings_data,
        config=_DACITE_CONFIG,
    )


def _expand_settings_dicts(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand a dict with possible list values into all combinations of dicts.

    Any field that has a list value will be expanded into multiple dicts.
    Non-list fields are passed through as-is.

    Returns raw dicts (not ExperimentSettings) to support proper merging.
    """
    eval_field_names = set(_get_eval_field_names())

    # Separate list-valued eval fields from other fields
    list_fields: dict[str, list[Any]] = {}
    scalar_fields: dict[str, Any] = {}

    for key, value in data.items():
        if key in eval_field_names and isinstance(value, list):
            list_fields[key] = value
        else:
            scalar_fields[key] = value

    # If no list fields, return single dict
    if not list_fields:
        return [dict(data)]

    # Generate all combinations of list field values
    field_names = list(list_fields.keys())
    field_value_lists = [list_fields[name] for name in field_names]

    results = []
    for values in product(*field_value_lists):
        combo_data = dict(scalar_fields)
        for name, value in zip(field_names, values):
            combo_data[name] = value
        results.append(combo_data)

    return results


def _expand_experiment(
    raw_exp: dict[str, Any],
    base_settings_dict: dict[str, Any],
) -> list[ExperimentConfig]:
    """Expand a raw experiment dict into one or more ExperimentConfigs.

    Each expanded config gets:
    - A suffix encoding its settings (e.g., T0_Fs_H00_C1_...)
    - A replicate suffix (e.g., --rep1, --rep2) if replicates > 1

    Final name format: {name}--{config_suffix}--rep{n}

    Args:
        raw_exp: Raw experiment dict from YAML
        base_settings_dict: Base settings as a raw dict (for proper merging)
    """
    name = raw_exp.get("name", "")
    settings_data = raw_exp.get("settings", {})
    completed = raw_exp.get("completed", False)

    # Parse run_mode (default: local)
    run_mode_str = raw_exp.get("run_mode", "local").lower()
    try:
        run_mode = RunMode(run_mode_str)
    except ValueError:
        print(f"[WARNING] Invalid run_mode '{run_mode_str}' for {name}, using 'local'")
        run_mode = RunMode.LOCAL

    if not settings_data:
        return [ExperimentConfig(name=name, completed=completed, run_mode=run_mode)]

    # Expand any list values into multiple dicts
    expanded_dicts = _expand_settings_dicts(settings_data)

    results = []
    for exp_dict in expanded_dicts:
        # Merge at dict level: base provides defaults, experiment overrides
        merged_dict = {**base_settings_dict, **exp_dict}
        merged_settings = _settings_from_dict(merged_dict)

        suffix = merged_settings.get_output_suffix()
        base_exp_name = f"{name}--{suffix}"

        # Expand replicates (use merged value to inherit from base_settings if not specified)
        num_replicates = max(1, merged_settings.replicates)
        for rep in range(1, num_replicates + 1):
            results.append(
                ExperimentConfig(
                    name=f"{base_exp_name}--rep{rep}",
                    settings=merged_settings,
                    completed=completed,
                    run_mode=run_mode,
                )
            )
    return results


def _expand_catalogue(raw_data: dict[str, Any]) -> ExperimentCatalogue:
    """Expand raw YAML data into a fully typed ExperimentCatalogue.

    This is the main entry point for converting raw YAML dicts into typed dataclasses.
    Handles list expansion for settings fields.
    """
    # Get raw base_settings dict (flatten any lists to first value for base)
    raw_base = raw_data.get("base_settings", {})
    base_settings_dict = {
        k: (v[0] if isinstance(v, list) else v) for k, v in raw_base.items()
    }
    # Also create a parsed base_settings for the catalogue
    base_settings = _settings_from_dict(base_settings_dict)

    raw_experiments = raw_data.get("experiments", [])

    experiments: list[ExperimentConfig] = []
    for raw_exp in raw_experiments:
        if raw_exp:
            experiments.extend(_expand_experiment(raw_exp, base_settings_dict))

    return ExperimentCatalogue(base_settings=base_settings, experiments=experiments)


def load_experiment_catalogue(catalogue_path: str | Path) -> ExperimentCatalogue:
    """Load and expand experiment catalogue from YAML file.

    Returns a fully typed ExperimentCatalogue with all list values expanded
    into individual experiments.
    """
    catalogue_path = Path(catalogue_path)
    if not catalogue_path.exists():
        raise FileNotFoundError(f"Experiment catalogue not found at {catalogue_path}")

    with open(catalogue_path) as f:
        raw_data = yaml.safe_load(f)

    raw_exp_count = len(raw_data.get("experiments", []))
    catalogue = _expand_catalogue(raw_data)

    print(
        f"[CONFIG] Loaded {raw_exp_count} experiment(s), "
        f"expanded to {len(catalogue.experiments)} experiment(s)"
    )

    # Validate: no duplicate experiment names (after suffixes are added)
    # This catches cases where the same experiment is registered twice with identical settings
    from collections import Counter

    name_counts = Counter(e.name for e in catalogue.experiments)
    duplicates = [(name, count) for name, count in name_counts.items() if count > 1]
    if duplicates:
        details = "\n  ".join(
            f"'{name}' appears {count} times" for name, count in duplicates
        )
        raise ValueError(
            f"Duplicate experiment names found:\n  {details}\n\n"
            "This usually means the same experiment (with identical settings) was registered twice in the YAML."
        )

    # Validate: no experiment has empty settings
    empty_settings = [e.name for e in catalogue.experiments if e.settings is None]
    if empty_settings:
        raise ValueError(
            f"Experiments with empty settings found: {', '.join(empty_settings)}"
        )

    return catalogue


# --- State persistence classes ---


@dataclass
class ExperimentState:
    """Persisted state for a running experiment."""

    experiment_name: str = ""
    pid: int = 0
    start_time: str = ""  # ISO format
    test_suite: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ExperimentName": self.experiment_name,
            "Pid": self.pid,
            "StartTime": self.start_time,
            "TestSuite": self.test_suite,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentState:
        return cls(
            experiment_name=data.get("ExperimentName", ""),
            pid=data.get("Pid", 0),
            start_time=data.get("StartTime", ""),
            test_suite=data.get("TestSuite", ""),
        )


@dataclass
class RunningExperiment:
    """Tracks a running experiment process."""

    experiment_name: str = ""
    test_suite: str = ""
    process: Any = None  # subprocess.Popen
    start_time: float = 0.0  # time.time()
    log_file_path: str = ""

    @property
    def key(self) -> str:
        return f"{self.experiment_name}--{self.test_suite}"


@dataclass
class TestResult:
    """Result of a completed experiment."""

    test_name: str = ""
    success: bool = False
    exit_code: int = 0
    output: str = ""
    errors: str = ""
