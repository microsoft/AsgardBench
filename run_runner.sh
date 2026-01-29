#!/usr/bin/env bash

# dotnet run --project experiment_runner/experiment_runner.csproj

# Set mounted storage path for experiment runner
export MOUNTED_STORAGE_PATH=~/mnt/magmardata_magmathor

uv run python -m experiment_runner --max-jobs 60
