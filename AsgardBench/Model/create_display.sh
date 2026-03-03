#!/bin/bash
set -ex

echo "--= Create Display =--"

display="90"

echo "Starting Xvfb :$display..."
Xvfb :$display -screen 0 1024x1024x24 -ac +extension GLX +render -noreset &

# Give Xvfb time to start
sleep 2

# Export DISPLAY so it's available to subprocesses
export DISPLAY=":$display"
echo "DISPLAY set to $DISPLAY"

# Run a tiny initialization to force Unity to start and create Player.log
echo "--= Running AI2Thor initialization... =--"
uv run python AsgardBench/Model/force_ai2thor_initialization.py || true
echo "--= Create Display Done =--"
