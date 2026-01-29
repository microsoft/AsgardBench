#!/usr/bin/env python3
"""
GPU Keep-Alive Monitor for vLLM Server
Monitors GPU utilization and sends dummy inference requests when utilization drops to 0%
to prevent compute cluster from killing the job.
"""

import json
import signal
import subprocess
import sys
import time
from typing import Optional

import requests

# Configuration
CHECK_INTERVAL_SECONDS = 10  # How often to check GPU utilization (normal)
CHECK_INTERVAL_AFTER_REQUEST = 3  # Check sooner after sending a keep-alive request
UTILIZATION_THRESHOLD = 5  # Send request if utilization below this percentage
VLLM_SERVER_URL = "http://localhost:43289/v1/completions"
MODEL_NAME = "vllm-model"

# Global flag for graceful shutdown
shutdown_flag = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global shutdown_flag
    print(f"\nReceived signal {signum}, shutting down GPU keep-alive monitor...")
    shutdown_flag = True


def get_gpu_utilization() -> Optional[float]:
    """
    Get current GPU utilization using nvidia-smi.
    Returns the average utilization across all GPUs, or None if error.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode != 0:
            print(f"Error running nvidia-smi: {result.stderr}")
            return None

        # Parse utilization values
        utilizations = [
            float(line.strip())
            for line in result.stdout.strip().split("\n")
            if line.strip()
        ]

        if not utilizations:
            return None

        # Return average utilization
        avg_util = sum(utilizations) / len(utilizations)
        return avg_util

    except subprocess.TimeoutExpired:
        print("nvidia-smi command timed out")
        return None
    except Exception as e:
        print(f"Error getting GPU utilization: {e}")
        return None


def send_dummy_request():
    """
    Send a minimal inference request to the vLLM server to keep GPUs active.
    Uses a very short prompt and minimal tokens to minimize overhead.
    """
    try:
        payload = {
            "model": MODEL_NAME,
            "prompt": "Hi",
            "max_tokens": 1,
            "temperature": 0.0,
        }

        response = requests.post(VLLM_SERVER_URL, json=payload, timeout=30)

        if response.status_code == 200:
            print(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Sent keep-alive request (GPU util was low)"
            )
            return True
        else:
            print(
                f"Keep-alive request failed with status {response.status_code}: {response.text}"
            )
            return False

    except requests.exceptions.RequestException as e:
        print(f"Error sending keep-alive request: {e}")
        return False


def check_server_health() -> bool:
    """Check if vLLM server is responding"""
    try:
        # Try to get models endpoint
        response = requests.get("http://localhost:43289/v1/models", timeout=5)
        return response.status_code == 200
    except:
        return False


def main():
    global shutdown_flag

    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("=" * 70)
    print("GPU Keep-Alive Monitor for vLLM Server")
    print("=" * 70)
    print(
        f"Check interval: {CHECK_INTERVAL_SECONDS}s (normal), {CHECK_INTERVAL_AFTER_REQUEST}s (after request)"
    )
    print(f"Utilization threshold: {UTILIZATION_THRESHOLD}%")
    print(f"vLLM server URL: {VLLM_SERVER_URL}")
    print("=" * 70)

    # Wait for server to be ready
    print("\nWaiting for vLLM server to be ready...", end="", flush=True)
    while not shutdown_flag:
        if check_server_health():
            print(" ✓ Server is ready!")
            break
        print(".", end="", flush=True)
        time.sleep(2)

    if shutdown_flag:
        return

    print("\nStarting GPU utilization monitoring...\n")

    consecutive_errors = 0
    max_consecutive_errors = 5
    next_check_interval = CHECK_INTERVAL_SECONDS  # Start with normal interval

    try:
        while not shutdown_flag:
            # Get current GPU utilization
            utilization = get_gpu_utilization()

            if utilization is None:
                consecutive_errors += 1
                print(
                    f"Warning: Could not get GPU utilization (error {consecutive_errors}/{max_consecutive_errors})"
                )

                if consecutive_errors >= max_consecutive_errors:
                    print("Too many consecutive errors, exiting...")
                    break

                time.sleep(next_check_interval)
                continue

            # Reset error counter on success
            consecutive_errors = 0

            # Check if we need to send a keep-alive request
            if utilization < UTILIZATION_THRESHOLD:
                print(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] GPU utilization: {utilization:.1f}% (below threshold)"
                )
                send_dummy_request()
                # Check sooner after sending request (utilization likely still low)
                next_check_interval = CHECK_INTERVAL_AFTER_REQUEST
            else:
                # Only print occasionally when things are fine
                if int(time.time()) % 60 < CHECK_INTERVAL_SECONDS:
                    print(
                        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] GPU utilization: {utilization:.1f}% (OK)"
                    )
                # Return to normal check interval when utilization is healthy
                next_check_interval = CHECK_INTERVAL_SECONDS

            # Wait before next check
            time.sleep(next_check_interval)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received")
    finally:
        print("GPU keep-alive monitor stopped.")


if __name__ == "__main__":
    main()
