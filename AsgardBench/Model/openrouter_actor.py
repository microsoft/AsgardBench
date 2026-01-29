# Required packages
import base64
import logging
import os
import random
import time
import traceback
from typing import Any, Final

import requests
from PIL import Image

from AsgardBench.Model.prompt_templates import split_prompt_for_caching
from AsgardBench.objects import ModelEmptyResponseError
from AsgardBench.Utils.keyvault import get_openrouter_api_key

# OpenRouter API endpoint
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


def _normalize_model_name(model_name: str) -> str:
    """
    Convert model names from folder-safe format to OpenRouter API format.

    Converts "provider__model" to "provider/model" for OpenRouter API calls.
    This allows using model names like "z-ai__glm-4.6v" in experiment configs
    without creating subfolders.

    Args:
        model_name: The model name, possibly with "__" separator

    Returns:
        Model name with "/" separator for API calls
    """
    # Replace first occurrence of "__" with "/" (provider separator)
    if "__" in model_name:
        return model_name.replace("__", "/", 1)
    return model_name


# Model variant suffixes that map to reasoning_effort levels
# Format: "model-name-{effort}" where effort is: none, minimal, low, medium, high, xhigh
# These are treated as separate models in-code to make comparison easier
_REASONING_EFFORT_SUFFIXES = ["none", "minimal", "low", "medium", "high", "xhigh"]


def _parse_model_reasoning_effort(model_name: str) -> tuple[str, str | None]:
    """
    Parse model name to extract the base model and reasoning effort suffix.

    Args:
        model_name: The model name, possibly with a reasoning effort suffix
                   (e.g., "deepseek/deepseek-v3.2-medium" -> ("deepseek/deepseek-v3.2", "medium"))

    Returns:
        Tuple of (base_model_name, reasoning_effort or None)
    """
    model_lower = model_name.lower()
    for suffix in _REASONING_EFFORT_SUFFIXES:
        if model_lower.endswith(f"-{suffix}"):
            # Strip the suffix to get the base model name
            base_model = model_name[: -(len(suffix) + 1)]
            return base_model, suffix
    return model_name, None


# Model-specific default configurations
# Each entry maps a model pattern to extra parameters to pass to the API
MODEL_SPECIFIC_PARAMS: dict[str, dict[str, Any]] = {
    "deepseek/deepseek-v3.2": {
        "reasoning": {
            "enabled": True,
            "effort": "medium",
        }
    },
    "google/gemini-3-pro-preview": {
        "reasoning": {
            "enabled": True,
        }
    },
    "anthropic/claude-opus-4.5": {
        "reasoning": {
            "enabled": True,
        }
    },
    "anthropic/claude-sonnet-4.5": {
        "reasoning": {
            "enabled": True,
        }
    },
    "qwen/qwen3-vl-235b-a22b-thinking": {
        "reasoning": {
            "enabled": True,
        }
    },
    "z-ai/glm-4.6v": {
        "reasoning": {
            "enabled": True,
        }
    },
}


def get_model_specific_params(model_name: str) -> dict[str, Any]:
    """Get default extra parameters for a model based on its name."""
    for pattern, params in MODEL_SPECIFIC_PARAMS.items():
        if pattern in model_name.lower():
            return params.copy()
    return {}


class OpenRouterActor:

    def __init__(
        self,
        model_name: str,
        temperature: float,
        max_completion_tokens=4096,
        reasoning_config: dict[str, Any] | None = None,
        extra_params: dict[str, Any] | None = None,
        run_metadata: str | None = None,
    ):
        """Create the OpenRouter client instance.

        Args:
            model_name: The model identifier (e.g., 'deepseek/deepseek-chat-v3-0324')
                        Can include reasoning effort suffix (e.g., 'model-name-high')
                        Valid suffixes: none, minimal, low, medium, high, xhigh
            temperature: Sampling temperature
            max_completion_tokens: Maximum tokens for completion
            reasoning_config: Optional reasoning configuration for models that support it.
                              Example: {"effort": "medium", "summary": "auto"}
                              Valid effort values: xhigh, high, medium, low, minimal, none
                              Valid summary values: auto, concise, detailed
                              If not provided, will be auto-detected from model name suffix.
            extra_params: Additional parameters to pass to the API call
            run_metadata: Optional identifier for tracking requests (e.g., experiment config name).
                          This is sent to OpenRouter as the 'user' parameter and appears
                          as 'external_user' in generation stats for analytics.
        """

        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens

        # Normalize model name: convert "provider--model" to "provider/model"
        normalized_model_name = _normalize_model_name(model_name)

        # Parse model name to extract reasoning effort if specified
        # e.g., "deepseek/deepseek-v3.2-medium" -> base_model="deepseek/deepseek-v3.2", reasoning_effort="medium"
        base_model_name, reasoning_effort = _parse_model_reasoning_effort(
            normalized_model_name
        )
        self._base_model_name = base_model_name
        self._reasoning_effort = reasoning_effort
        self.model_name = model_name  # Keep original for logging

        # Get model-specific defaults and merge with provided params
        self.extra_params = get_model_specific_params(base_model_name)
        if extra_params:
            self.extra_params.update(extra_params)

        # Override reasoning config if explicitly provided
        if reasoning_config is not None:
            print(
                f"Using explicit reasoning config for model {model_name}: {reasoning_config}"
            )
            self.extra_params["reasoning"] = reasoning_config

        elif reasoning_effort is not None:
            # Auto-configure reasoning from model name suffix
            # Must set "enabled": True to activate reasoning
            print(
                f"Auto-configuring reasoning for model {model_name} with effort: {reasoning_effort}"
            )

            self.extra_params["reasoning"] = {
                "enabled": True,
                "effort": reasoning_effort,
            }

        # Get API key - uses Key Vault on AML, env var locally
        self.api_key = get_openrouter_api_key()

        # Optional headers for OpenRouter rankings
        self.site_url = os.getenv("OPENROUTER_SITE_URL", "")
        self.site_name = os.getenv("OPENROUTER_SITE_NAME", "")

        # Optional run metadata for tracking (sent as 'user' parameter)
        self.run_metadata = run_metadata

        print(f"Initialized OpenRouter client with model: {model_name}")
        if self.extra_params:
            print(f"Extra params: {self.extra_params}")
        if self.run_metadata:
            print(f"Run metadata: {self.run_metadata}")

    def _build_headers(self) -> dict[str, str]:
        """Build request headers including auth and optional site info."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            headers["X-Title"] = self.site_name
        return headers

    def get_response(
        self,
        current_image_path: str | None,
        previous_image_path: str | None,
        prompt: str,
    ):
        """
        Get AI response using OpenRouter.

        Args:
            current_image_path (str|None): Path to the current image file. None for text-only mode.
            previous_image_path (str|None): Path to the previous image file. None if not using previous image.
            prompt: The rendered prompt string (may contain <<CACHE_BOUNDARY>> marker).
                    The marker will be used to split static vs dynamic content for caching.

        Returns:
            str: The AI response content or error message
        """

        try:
            # Split prompt at cache boundary for optimal caching
            # Static content goes in system message, dynamic content in user message
            # This maximizes prefix caching across all providers:
            # - DeepSeek, OpenAI, Grok, etc.: automatic prefix-based caching
            # - Anthropic, Gemini: explicit cache_control breakpoints
            static_part, dynamic_part = split_prompt_for_caching(prompt)

            is_anthropic = "anthropic" in self._base_model_name.lower()
            is_gemini = "google" in self._base_model_name.lower()

            # Build system message content
            # Add cache_control for providers that support explicit caching:
            # - Anthropic: requires cache_control, supports up to 4 breakpoints, TTL 5min (default) or 1h
            # - Gemini: accepts cache_control, uses only the last breakpoint, 5min TTL
            # - Others (OpenAI, DeepSeek, Grok, etc.): automatic prefix caching, no cache_control needed
            messages = []

            if static_part:
                system_content = {
                    "type": "text",
                    "text": static_part,
                }

                if is_anthropic or is_gemini:
                    system_content["cache_control"] = {"type": "ephemeral"}

                messages.append({"role": "system", "content": [system_content]})

            # Build image content list
            image_content = []

            # Add previous image first if provided
            if previous_image_path is not None:
                with open(previous_image_path, "rb") as prev_file:
                    prev_data = base64.b64encode(prev_file.read()).decode("utf-8")
                    image_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{prev_data}"},
                        }
                    )

            # Add current image if provided
            if current_image_path is not None:
                with open(current_image_path, "rb") as image_file:
                    image_data = base64.b64encode(image_file.read()).decode("utf-8")
                    image_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_data}"},
                        }
                    )

            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": dynamic_part},
                        *image_content,
                    ],
                }
            )

            # Build request payload
            payload = {
                "model": self._base_model_name,  # Use base model name for API call
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_completion_tokens,
                **self.extra_params,
            }

            # Add run metadata as 'user' parameter for tracking
            # This appears as 'external_user' in OpenRouter's generation stats
            if self.run_metadata:
                payload["user"] = self.run_metadata

            headers = self._build_headers()

            # Do request with retry logic for transient errors
            query_start_time = time.time()
            attempt = 0

            max_policy_violation_retries: Final = 10
            policy_violation_count = 0

            max_dangerous_errors: Final = 10
            dangerous_errors_count = 0

            # Max retries for transient network errors (timeout, connection)
            max_transient_retries: Final = 80
            transient_error_count = 0

            # Dedicated counter for empty model responses (model didn't produce output)
            # Lower limit since this is likely a model issue, not transient
            max_empty_response_retries: Final = 3
            empty_response_count = 0

            while True:
                attempt += 1
                try:
                    response = requests.post(
                        OPENROUTER_API_URL,
                        json=payload,
                        headers=headers,
                        timeout=120,
                    )

                    # Check for rate limiting (429)
                    if response.status_code == 429:
                        # Respect Retry-After header if present
                        retry_after = response.headers.get("Retry-After")
                        if retry_after:
                            try:
                                wait_time = int(retry_after)
                            except ValueError:
                                wait_time = 5
                        else:
                            wait_time = 2 + random.random() * 3
                        logging.warning(
                            f"Rate limited (429); retrying after {wait_time:.1f}s (attempt {attempt}). "
                            f"Response: {response.text}"
                        )
                        time.sleep(wait_time)
                        continue

                    # Check for payment/credit issues (402 Payment Required)
                    # This happens when OpenRouter credits run out - wait for auto top-up
                    if response.status_code == 402:
                        # Wait 5 minutes before retrying - gives time for auto top-up
                        wait_time = 300
                        logging.warning(
                            f"Payment required (402) - credits exhausted; waiting {wait_time}s for auto top-up "
                            f"(attempt {attempt}). Response: {response.text[:200]}"
                        )
                        time.sleep(wait_time)
                        continue

                    # Check for CloudFlare DDoS protection blocks (52x errors)
                    # These are CloudFlare-specific errors that indicate the request was blocked
                    # 520: Unknown error, 521: Web server down, 522: Connection timed out,
                    # 523: Origin unreachable, 524: Timeout, 525: SSL handshake failed,
                    # 526: Invalid SSL cert, 527: Railgun error, 530: Origin DNS error
                    if 520 <= response.status_code <= 530:
                        # CloudFlare often sends Retry-After header
                        retry_after = response.headers.get("Retry-After")
                        if retry_after:
                            try:
                                wait_time = int(retry_after)
                            except ValueError:
                                wait_time = (
                                    10  # Default longer wait for CloudFlare blocks
                                )
                        else:
                            # Use exponential-ish backoff for CloudFlare errors
                            wait_time = min(5 + attempt * 2 + random.random() * 5, 60)
                        logging.warning(
                            f"CloudFlare error {response.status_code}; retrying after {wait_time:.1f}s "
                            f"(attempt {attempt}). Response: {response.text[:500]}"
                        )
                        time.sleep(wait_time)
                        continue

                    # Check for transient server errors
                    if response.status_code in {500, 502, 503, 504}:
                        # 503 may also be CloudFlare DDoS protection, check for Retry-After
                        retry_after = response.headers.get("Retry-After")
                        if retry_after:
                            try:
                                wait_time = int(retry_after)
                            except ValueError:
                                wait_time = 5
                        else:
                            wait_time = 2 + random.random() * 3
                        logging.warning(
                            f"Transient API error {response.status_code}; retrying after {wait_time:.1f}s "
                            f"(attempt {attempt}). Response: {response.text}"
                        )
                        time.sleep(wait_time)
                        continue

                    # Check for policy/content moderation violations (often 400 Bad Request)
                    if response.status_code == 400:
                        response_text = response.text.lower()
                        is_policy_violation = (
                            "content_filter" in response_text
                            or "content filter" in response_text
                            or "moderation" in response_text
                            or "policy" in response_text
                            or "flagged" in response_text
                        )
                        if is_policy_violation:
                            policy_violation_count += 1
                            if policy_violation_count < max_policy_violation_retries:
                                logging.warning(
                                    f"Policy violation; retrying ({policy_violation_count}/{max_policy_violation_retries}). "
                                    f"Response: {response.text}"
                                )
                                time.sleep(1 + random.random() * 2)
                                continue
                            else:
                                logging.error(
                                    f"Policy violation retry limit reached ({max_policy_violation_retries}). "
                                    f"Response: {response.text}"
                                )
                        else:
                            # Log the full response body for non-policy 400 errors
                            # These might be provider-specific issues that we should be aware of
                            logging.warning(
                                f"Non-policy 400 Bad Request. Response body: {response.text[:2000]}"
                            )

                    # Raise for other HTTP errors
                    response.raise_for_status()

                    # Parse response JSON - if malformed, retry the request
                    response_json = response.json()

                    # Check for wrapped upstream errors (HTTP 200 but error in body)
                    # These are transient server errors from upstream providers
                    error_obj = response_json.get("error")
                    if error_obj and isinstance(error_obj, dict):
                        error_code = error_obj.get("code")
                        error_message = error_obj.get("message", "")
                        metadata = error_obj.get("metadata", {})
                        provider_name = metadata.get("provider_name", "unknown")
                        raw_error = metadata.get("raw", "")

                        # Treat 5xx errors in body as transient (same as HTTP 5xx)
                        if isinstance(error_code, int) and 500 <= error_code < 600:
                            logging.warning(
                                f"Upstream provider returned {error_code} error; retrying. "
                                f"Error: {error_obj}"
                            )
                            time.sleep(2 + random.random() * 3)
                            continue

                        # Handle 4xx errors from upstream providers
                        # Some providers (like SiliconFlow) return 400 for transient issues
                        if isinstance(error_code, int) and 400 <= error_code < 500:
                            # Check if it's a potentially transient provider error
                            is_transient_provider_error = (
                                "parameter is invalid" in raw_error.lower()
                                or "20015"
                                in raw_error  # SiliconFlow specific error code
                                or "timeout" in raw_error.lower()
                                or "overloaded" in raw_error.lower()
                                or "temporarily" in raw_error.lower()
                            )

                            if is_transient_provider_error:
                                transient_error_count += 1
                                if transient_error_count < max_transient_retries:
                                    logging.warning(
                                        f"Upstream provider {provider_name} returned {error_code} "
                                        f"(possibly transient); retrying ({transient_error_count}/{max_transient_retries}). "
                                        f"Error: {error_obj}"
                                    )
                                    time.sleep(2 + random.random() * 3)
                                    continue
                                else:
                                    raise RuntimeError(
                                        f"Upstream provider {provider_name} error retry limit reached "
                                        f"({max_transient_retries}). Error: {error_obj}"
                                    )
                            else:
                                # Non-transient 4xx error from upstream - log and raise
                                logging.error(
                                    f"Upstream provider {provider_name} returned {error_code} error "
                                    f"(non-transient): {error_obj}"
                                )
                                raise RuntimeError(
                                    f"Upstream provider {provider_name} returned error: {error_obj}"
                                )

                    # Check for error response structure (missing or empty 'choices' key)
                    # This likely means the model didn't produce output (not a transient API error)
                    choices = response_json.get("choices")
                    if not choices:
                        empty_response_count += 1

                        if empty_response_count < max_empty_response_retries:
                            logging.warning(
                                f"Model returned empty response (no 'choices'); retrying "
                                f"({empty_response_count}/{max_empty_response_retries}). "
                                f"Full response: {response_json}"
                            )
                            time.sleep(2 + random.random() * 3)
                            continue
                        else:
                            error_info = response_json.get("error", response_json)
                            raise ModelEmptyResponseError(
                                f"Model produced no output after {max_empty_response_retries} retries. "
                                f"Response: {error_info}"
                            )

                    # Validate the structure of choices[0] - should have a 'message' dict
                    first_choice = choices[0]
                    message = (
                        first_choice.get("message")
                        if isinstance(first_choice, dict)
                        else None
                    )
                    if not isinstance(message, dict):
                        dangerous_errors_count += 1

                        if dangerous_errors_count < max_dangerous_errors:
                            logging.warning(
                                f"API returned malformed response (invalid 'message' structure); retrying "
                                f"({dangerous_errors_count}/{max_dangerous_errors}). "
                                f"choices[0]: {first_choice}"
                            )
                            time.sleep(2 + random.random() * 3)
                            continue
                        else:
                            raise RuntimeError(
                                f"Malformed response retry limit reached ({max_dangerous_errors}). "
                                f"Invalid 'message' structure in choices[0]: {first_choice}"
                            )

                    # Success - exit retry loop
                    break

                except requests.Timeout:
                    transient_error_count += 1
                    if transient_error_count >= max_transient_retries:
                        raise RuntimeError(
                            f"Timeout retry limit reached ({max_transient_retries}). "
                            f"Request keeps timing out."
                        )
                    logging.warning(
                        f"Timeout error; retrying after short delay "
                        f"({transient_error_count}/{max_transient_retries})."
                    )
                    time.sleep(2 + random.random() * 3)
                    continue

                except requests.ConnectionError as e:
                    transient_error_count += 1
                    if transient_error_count >= max_transient_retries:
                        raise RuntimeError(
                            f"Connection error retry limit reached ({max_transient_retries}). "
                            f"Last error: {e}"
                        )
                    logging.warning(
                        f"Connection error; retrying after short delay "
                        f"({transient_error_count}/{max_transient_retries}). Error: {e}"
                    )
                    time.sleep(2 + random.random() * 3)
                    continue

                except requests.exceptions.ChunkedEncodingError as e:
                    # Response stream was cut off prematurely (common with Gemini)
                    logging.warning(
                        f"Chunked encoding error (response ended prematurely); retrying. Error: {e}"
                    )
                    time.sleep(2 + random.random() * 2)
                    continue

                except requests.exceptions.JSONDecodeError as e:
                    dangerous_errors_count += 1

                    if dangerous_errors_count < max_dangerous_errors:
                        logging.warning(
                            f"JSON decode error (malformed response); retrying "
                            f"({dangerous_errors_count}/{max_dangerous_errors}). Error: {e}"
                        )
                        time.sleep(2 + random.random() * 2)
                        continue
                    else:
                        raise RuntimeError(
                            f"JSON decode error retry limit reached ({max_dangerous_errors}). "
                            f"Last error: {e}"
                        )

                except requests.HTTPError as e:
                    # HTTP errors (4xx, 5xx) that weren't handled above
                    # Log the response body for diagnosis - this helps identify
                    # provider-specific issues, context length errors, etc.
                    response_body = ""
                    if e.response is not None:
                        try:
                            response_body = e.response.text[:2000]
                        except Exception:
                            response_body = "(unable to read response body)"

                    dangerous_errors_count += 1

                    if dangerous_errors_count < max_dangerous_errors:
                        logging.warning(
                            f"HTTP error during API request; retrying "
                            f"({dangerous_errors_count}/{max_dangerous_errors}). "
                            f"Error type: {type(e).__name__}, Error: {e}, "
                            f"Response body: {response_body}"
                        )
                        time.sleep(2 + random.random() * 3)
                        continue
                    else:
                        raise RuntimeError(
                            f"Unexpected error retry limit reached ({max_dangerous_errors}). "
                            f"Error type: {type(e).__name__}, Last error: {e}, "
                            f"Response body: {response_body}"
                        )

                except Exception as e:
                    # Catch-all for unexpected errors during the request/response cycle
                    dangerous_errors_count += 1

                    if dangerous_errors_count < max_dangerous_errors:
                        logging.warning(
                            f"Unexpected error during API request; retrying "
                            f"({dangerous_errors_count}/{max_dangerous_errors}). "
                            f"Error type: {type(e).__name__}, Error: {e}"
                        )
                        time.sleep(2 + random.random() * 3)
                        continue
                    else:
                        raise RuntimeError(
                            f"Unexpected error retry limit reached ({max_dangerous_errors}). "
                            f"Error type: {type(e).__name__}, Last error: {e}"
                        )

            # Extract response content - use the validated 'message' dict from the loop
            response_content = message.get("content", "")
            reasoning_trace = message.get("reasoning", "")

            if reasoning_trace and "<think>" not in response_content:
                response_content = (
                    f"<think>\n{reasoning_trace}\n</think>\n{response_content}"
                )

            # Strip model-specific special tokens that may interfere with parsing
            # GLM models use <|begin_of_box|> and <|end_of_box|> for thinking markers
            response_content = response_content.replace("<|begin_of_box|>", "")
            response_content = response_content.replace("<|end_of_box|>", "")

            num_input_tokens = response_json["usage"]["prompt_tokens"]
            num_output_tokens = response_json["usage"]["completion_tokens"]
            cost_for_this_query = response_json["usage"].get("cost", 0.0)

            # Detailed logging for cost estimation
            if os.getenv("DETAILED_LOGGING", "0") == "1":
                print("===============PROMPT=================")
                print(f"[PROMPT_CHARS: {len(prompt)}]")
                print(f"[INPUT_TOKENS: {num_input_tokens}]")
                if current_image_path is not None:
                    img = Image.open(current_image_path)
                    image_bytes = os.path.getsize(current_image_path)
                    print(f"[IMAGE_SIZE: {img.size[0]}x{img.size[1]}]")
                    print(f"[IMAGE_BYTES: {image_bytes}]")
                else:
                    print("[IMAGE_SIZE: NONE]")
                    print("[IMAGE_BYTES: 0]")
                print(prompt)
                print("=============END PROMPT===============")

                if reasoning_trace:
                    print("===============REASONING TRACE=================")
                    print(f"[REASONING_TRACE_CHARS: {len(reasoning_trace)}]")
                    print(reasoning_trace)
                    print("=============END REASONING TRACE===============")

                print("===============RESPONSE=================")
                print(f"[RESPONSE_CHARS: {len(response_content)}]")
                print(f"[OUTPUT_TOKENS: {num_output_tokens}]")
                print(f"[QUERY_TIME: {time.time() - query_start_time:.2f} seconds]")
                print(f"[COST_THIS_QUERY: ${cost_for_this_query:.6f}]")
                print(response_content)
                print("=============END RESPONSE===============")

            return response_content

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            raise e
