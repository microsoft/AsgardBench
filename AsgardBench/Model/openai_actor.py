"""
Unified OpenAI-compatible actor for AsgardBench.

Supports any OpenAI-compatible API endpoint including:
- OpenAI (api.openai.com)
- Azure OpenAI (with API key auth)
- OpenRouter (openrouter.ai)
- VLLM and other local deployments
- Any other OpenAI-compatible endpoint

Configuration via environment variables:
- OPENAI_API_KEY: API key for authentication
- OPENAI_BASE_URL: Base URL for the API (default: https://api.openai.com/v1)
- OPENAI_API_VERSION: Optional API version (for Azure OpenAI)
- OPENAI_CACHE_CONTROL: Cache control behavior ("automatic" or "explicit")
"""

import base64
import logging
import os
import random
import time
from typing import Any, Final

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

from AsgardBench.Model.prompt_templates import split_prompt_for_caching
from AsgardBench.objects import ModelEmptyResponseError


def _needs_explicit_cache_control() -> bool:
    """
    Check if explicit cache_control should be added to messages.

    Controlled by OPENAI_CACHE_CONTROL environment variable:
    - "explicit": Add cache_control to messages (needed for Anthropic, Google)
    - "automatic" (default): Provider handles caching automatically (OpenAI, DeepSeek, etc.)
    """
    cache_control = os.getenv("OPENAI_CACHE_CONTROL", "automatic").lower()
    return cache_control == "explicit"


class OpenAIActor:
    """
    Unified actor for OpenAI-compatible APIs.

    Handles prompt caching, image encoding, and retry logic for any
    OpenAI-compatible endpoint.
    """

    def __init__(
        self,
        model_name: str,
        temperature: float,
        max_completion_tokens: int = 4096,
        api_key: str | None = None,
        base_url: str | None = None,
        extra_params: dict[str, Any] | None = None,
    ):
        """
        Initialize the OpenAI-compatible actor.

        Args:
            model_name: The model identifier (e.g., 'gpt-4o', 'claude-3-opus')
            temperature: Sampling temperature
            max_completion_tokens: Maximum tokens for completion
            api_key: API key (defaults to OPENAI_API_KEY env var)
            base_url: Base URL (defaults to OPENAI_BASE_URL env var or OpenAI default)
            extra_params: Additional parameters to pass to the API call
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.extra_params = extra_params or {}

        # Get API configuration from environment if not provided
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No API key provided. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.base_url = base_url or os.getenv(
            "OPENAI_BASE_URL", "https://api.openai.com/v1"
        )

        # Check if explicit cache_control is needed
        self._needs_cache_control = _needs_explicit_cache_control()

        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        print(f"Initialized OpenAI actor:")
        print(f"  Model: {model_name}")
        print(f"  Base URL: {self.base_url}")
        print(
            f"  Cache control: {'explicit' if self._needs_cache_control else 'automatic'}"
        )

    def get_response(
        self,
        current_image_path: str | None,
        previous_image_path: str | None,
        prompt: str,
    ) -> str:
        """
        Get AI response from the model.

        Args:
            current_image_path: Path to the current image file. None for text-only.
            previous_image_path: Path to the previous image file. None if not using.
            prompt: The rendered prompt string (may contain <<CACHE_BOUNDARY>> marker).

        Returns:
            The AI response content.

        Raises:
            ModelEmptyResponseError: If the model returns an empty response.
        """
        # Split prompt at cache boundary for optimal caching
        static_part, dynamic_part = split_prompt_for_caching(prompt)

        # Build messages
        messages = []

        # System message with static content
        if static_part:
            system_content: dict[str, Any] = {"type": "text", "text": static_part}

            # Add cache_control for providers that need it
            if self._needs_cache_control:
                system_content["cache_control"] = {"type": "ephemeral"}

            messages.append({"role": "system", "content": [system_content]})

        # User message with dynamic content and images
        user_content: list[dict[str, Any]] = [{"type": "text", "text": dynamic_part}]

        # Add previous image first if provided
        if previous_image_path is not None:
            with open(previous_image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_data}"},
                    }
                )

        # Add current image if provided
        if current_image_path is not None:
            with open(current_image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_data}"},
                    }
                )

        messages.append({"role": "user", "content": user_content})

        # Retry configuration
        MAX_RETRIES: Final = 10
        MAX_EMPTY_RESPONSE_RETRIES: Final = 3

        empty_response_count = 0

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_completion_tokens=self.max_completion_tokens,
                    **self.extra_params,
                )

                # Extract response content
                if response.choices and response.choices[0].message.content:
                    return response.choices[0].message.content

                # Empty response from model
                empty_response_count += 1
                if empty_response_count >= MAX_EMPTY_RESPONSE_RETRIES:
                    raise ModelEmptyResponseError(
                        f"Model returned empty response after {MAX_EMPTY_RESPONSE_RETRIES} attempts"
                    )

                logging.warning(
                    f"Empty response from model; retrying ({empty_response_count}/{MAX_EMPTY_RESPONSE_RETRIES})"
                )
                time.sleep(1 + random.random())
                continue

            except RateLimitError as e:
                wait_time = 5 + random.random() * 5
                logging.warning(
                    f"Rate limited; retrying after {wait_time:.1f}s (attempt {attempt}/{MAX_RETRIES})"
                )
                time.sleep(wait_time)
                continue

            except APITimeoutError as e:
                wait_time = 2 + random.random() * 3
                logging.warning(
                    f"API timeout; retrying after {wait_time:.1f}s (attempt {attempt}/{MAX_RETRIES})"
                )
                time.sleep(wait_time)
                continue

            except APIConnectionError as e:
                wait_time = 2 + random.random() * 3
                logging.warning(
                    f"Connection error; retrying after {wait_time:.1f}s (attempt {attempt}/{MAX_RETRIES}): {e}"
                )
                time.sleep(wait_time)
                continue

            except APIStatusError as e:
                # Check for transient server errors
                if e.status_code in {500, 502, 503, 504}:
                    wait_time = 2 + random.random() * 3
                    logging.warning(
                        f"Server error {e.status_code}; retrying after {wait_time:.1f}s (attempt {attempt}/{MAX_RETRIES})"
                    )
                    time.sleep(wait_time)
                    continue
                raise

        raise RuntimeError(f"Failed after {MAX_RETRIES} attempts")
