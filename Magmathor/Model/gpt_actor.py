# Required packages
import base64
import logging
import os
import random
import threading
import time
import traceback
from typing import Final

from azure.identity import (
    AzureCliCredential,
    ChainedTokenCredential,
    DefaultAzureCredential,
    ManagedIdentityCredential,
    get_bearer_token_provider,
)
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AzureOpenAI,
    PermissionDeniedError,
    RateLimitError,
)
from PIL import Image

from Magmathor.Model.prompt_templates import strip_cache_marker
from Magmathor.objects import ModelEmptyResponseError

# Models that support the reasoning_effort parameter
_REASONING_MODELS = [
    "gpt-5",
    "gpt-5.1",
    "gpt-5.2",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5-pro",
    "o1",
    "o3",
    "o3-mini",
    "o3-pro",
    "o4-mini",
]

# Model variant suffixes that map to reasoning_effort levels for GPT-5 models
# Format: "gpt-5.2-{effort}" where effort is: none, minimal, low, medium, high, xhigh
# These are treated as separate models in-code to make comparison easier
_REASONING_EFFORT_SUFFIXES = ["none", "minimal", "low", "medium", "high", "xhigh"]


def _is_reasoning_model(model_name: str) -> bool:
    """Check if a model supports the reasoning_effort parameter."""
    model_lower = model_name.lower()
    return any(model_lower.startswith(m.lower()) for m in _REASONING_MODELS)


def _parse_model_reasoning_effort(model_name: str) -> tuple[str, str | None]:
    """
    Parse model name to extract the base model and reasoning effort suffix.

    Args:
        model_name: The model name, possibly with a reasoning effort suffix
                   (e.g., "gpt-5.2-medium" -> ("gpt-5.2", "medium"))

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


class GPTActor:

    def __init__(self, model_name: str, temperature: float, max_completion_tokens=4096):
        """Create the AzureOpenAI client instance with credential rotation."""

        self.temperature = temperature

        # Get API version from environment
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

        self.max_completion_tokens = max_completion_tokens

        # Parse model name to extract reasoning effort if specified
        # e.g., "gpt-5.2-medium" -> base_model="gpt-5.2", reasoning_effort="medium"
        base_model_name, reasoning_effort = _parse_model_reasoning_effort(model_name)
        self._base_model_name = base_model_name
        self._reasoning_effort = reasoning_effort

        # Initialize Azure resource rotation using the base model name
        self.model_name = model_name  # Keep original for logging
        self._init_azure_rotation(base_model_name)

        # Determine if this is a GPT model (uses max_completion_tokens) or not (uses max_tokens)
        self._is_gpt_model = base_model_name.lower().startswith("gpt-")

    def _build_chained_credential(self):
        """Build a chained credential for Azure authentication.

        On AML: Uses ManagedIdentityCredential with the job's user-assigned identity
        Locally: Uses AzureCliCredential (requires `az login`)
        """
        # Check if running on AML with managed identity
        managed_identity_client_id = os.environ.get("DEFAULT_IDENTITY_CLIENT_ID")

        if managed_identity_client_id:
            # On AML: prefer managed identity, fall back to CLI for debugging
            print(
                f"Using managed identity for Azure OpenAI: {managed_identity_client_id[:8]}..."
            )
            return ChainedTokenCredential(
                ManagedIdentityCredential(client_id=managed_identity_client_id),
                AzureCliCredential(),  # Fallback for local testing with AML env vars
            )
        else:
            # Local development: use CLI first, then default credential chain
            print("Using Azure CLI credential for Azure OpenAI (local mode)")
            return ChainedTokenCredential(
                AzureCliCredential(),
                DefaultAzureCredential(
                    exclude_cli_credential=True,
                    exclude_environment_credential=True,
                    exclude_shared_token_cache_credential=True,
                    exclude_developer_cli_credential=True,
                    exclude_powershell_credential=True,
                    exclude_interactive_browser_credential=True,
                    exclude_visual_studio_code_credentials=True,
                ),
            )

    def _azure_endpoint(self, resource_name: str) -> str:
        """Build Azure endpoint URL from resource name."""
        if resource_name.startswith("https://"):
            return resource_name

        return f"https://{resource_name}.openai.azure.com/"

    def _init_azure_rotation(self, model_name: str):
        """Initialize Azure credential and resource rotation."""
        # Get resource names from environment or use defaults

        resources_with_model = {
            "search-learn-2": ["gpt-4o", "gpt-5.2"],
            "west-us-3-aoai": ["gpt-4o", "gpt-4.1"],
            "sweden-aoai-3428": ["gpt-4o", "gpt-4.1"],
            "japan-aoai-423985": ["gpt-4o"],
            "west-us-3-aoai-9058433": ["gpt-4o", "gpt-4.1"],
            "japan-aoai-effai-543987": ["gpt-4o"],
            "sweden-aoai-effai-23457954": ["gpt-4o", "gpt-4.1", "gpt-4"],
            "https://validator-resource.cognitiveservices.azure.com/": [
                "gpt-5",
                "gpt-5.2",
                "gpt-4o",
                "Llama-4-Maverick-17B-128E-Instruct-FP8",
                "Mistral-Large-3",
            ],
        }

        self.azure_resources = []

        for resource, models in resources_with_model.items():
            if any(model_name.lower().startswith(m.lower()) for m in models):
                self.azure_resources.append(resource)

        assert (
            len(self.azure_resources) > 0
        ), f"No known Azure resources for model {model_name}"

        if not self.azure_resources:
            raise ValueError(
                "No Azure OpenAI resources configured. Set AZURE_OPENAI_RESOURCE_NAMES "
                "environment variable with comma-separated resource names."
            )

        # Build chained credential
        self.azure_credential = self._build_chained_credential()
        self.azure_token_provider = get_bearer_token_provider(
            self.azure_credential, "https://cognitiveservices.azure.com/.default"
        )

        # Client cache and rotation
        self.azure_client_cache = {}
        self.azure_lock = threading.Lock()
        self.azure_rr_counter = 0

        print(f"Initialized Azure rotation with {len(self.azure_resources)} resources")

    def _get_azure_client(self):
        """Return a rotated AzureOpenAI client (thread-safe)."""
        with self.azure_lock:
            resource = self.azure_resources[
                self.azure_rr_counter % len(self.azure_resources)
            ]
            self.azure_rr_counter += 1

            if resource not in self.azure_client_cache:
                azure_endpoint = self._azure_endpoint(resource)

                self.azure_client_cache[resource] = AzureOpenAI(
                    api_version=self.api_version,
                    azure_endpoint=azure_endpoint,
                    azure_ad_token_provider=self.azure_token_provider,
                )
                print(f"Created new Azure client for resource: {resource}")

            return self.azure_client_cache[resource]

    def get_response(
        self,
        current_image_path: str | None,
        previous_image_path: str | None,
        prompt: str,
    ):
        """
        Get AI response using rotated Azure OpenAI clients.

        Args:
            current_image_path (str|None): Path to the current image file. None for text-only mode.
            previous_image_path (str|None): Path to the previous image file. None if not using previous image.
            prompt (str): The user's message/prompt

        Returns:
            str: The AI response content or error message
        """

        try:
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

            # Build messages array
            # NOTE: Text comes BEFORE images to maximize prefix caching.
            # OpenAI/Azure uses automatic prefix-based caching, so putting
            # static text first means it can be cached across requests.
            # Strip the cache boundary marker (used by OpenRouter for explicit caching).
            clean_prompt = strip_cache_marker(prompt)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": clean_prompt},
                        *image_content,
                    ],
                }
            ]

            # Get a rotated Azure client
            client = self._get_azure_client()

            # Build completion kwargs - GPT models use max_completion_tokens, others use max_tokens
            completion_kwargs = {
                "model": self._base_model_name,  # Use base model name for API call
                "messages": messages,
                "temperature": self.temperature,
            }
            if self._is_gpt_model:
                completion_kwargs["max_completion_tokens"] = self.max_completion_tokens
            else:
                completion_kwargs["max_tokens"] = self.max_completion_tokens

            # Reasoning models don't support temperature
            if _is_reasoning_model(self._base_model_name):
                completion_kwargs.pop("temperature", None)

                # Add reasoning_effort if specified
                # Valid values: none, minimal, low, medium, high, xhigh
                if self._reasoning_effort is not None:
                    completion_kwargs["reasoning_effort"] = self._reasoning_effort

            # Do a chat completion with retry logic for transient errors
            query_start_time = time.time()
            attempt = 0

            max_policy_violation_retries: Final = 10
            policy_violation_count = 0

            max_dangerous_errors: Final = 20
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
                    response = client.chat.completions.create(**completion_kwargs)

                    # Check for empty response (no choices or empty content)
                    if not response.choices:
                        empty_response_count += 1

                        if empty_response_count < max_empty_response_retries:
                            logging.warning(
                                f"Model returned empty response (no 'choices'); retrying "
                                f"({empty_response_count}/{max_empty_response_retries})."
                            )
                            time.sleep(2 + random.random() * 3)
                            continue
                        else:
                            raise ModelEmptyResponseError(
                                f"Model produced no output after {max_empty_response_retries} retries."
                            )

                    break  # success
                except RateLimitError as e:  # 429
                    logging.warning(
                        f"Rate limited (429); retrying after short delay (attempt {attempt}). Error: {e}"
                    )
                    time.sleep(2 + random.random() * 3)
                    continue
                except APITimeoutError as e:
                    transient_error_count += 1
                    if transient_error_count >= max_transient_retries:
                        raise RuntimeError(
                            f"Timeout retry limit reached ({max_transient_retries}). "
                            f"Request keeps timing out."
                        )
                    logging.warning(
                        f"Timeout error; retrying after short delay "
                        f"({transient_error_count}/{max_transient_retries}). Error: {e}"
                    )
                    time.sleep(2 + random.random() * 3)
                    continue
                except APIConnectionError as e:
                    # Handles connection errors including httpx.RemoteProtocolError
                    # (e.g., "Server disconnected without sending a response")
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
                except PermissionDeniedError as e:
                    # Azure can return 403 "principal lacks data action" under heavy load
                    # Treat as transient and retry with the dangerous_errors counter
                    dangerous_errors_count += 1
                    if dangerous_errors_count >= max_dangerous_errors:
                        raise RuntimeError(
                            f"Permission denied retry limit reached ({max_dangerous_errors}). "
                            f"Last error: {e}"
                        )
                    logging.warning(
                        f"Permission denied (possibly transient); retrying "
                        f"({dangerous_errors_count}/{max_dangerous_errors}). Error: {e}"
                    )
                    time.sleep(2 + random.random() * 3)
                    continue
                except APIError as e:
                    # Retry on transient server errors (500/502/503/504) if status is available.
                    status = getattr(e, "status_code", None)
                    if status in {500, 502, 503, 504}:
                        logging.warning(
                            f"Transient API error {status}; retrying after short delay (attempt {attempt}). Error: {e}"
                        )
                        time.sleep(2 + random.random() * 3)
                        continue
                    error_str = str(e).lower()

                    # Retry on upstream request timeout (Azure backend timeout)
                    if "upstream request timeout" in error_str:
                        logging.warning(
                            f"Upstream request timeout; retrying after delay (attempt {attempt}). Error: {e}"
                        )
                        time.sleep(5)  # Longer delay for upstream timeouts
                        continue

                    # Retry on Azure policy/content filter violations (up to 10 times)
                    is_policy_violation = (
                        "content_filter" in error_str
                        or "content filter" in error_str
                        or "policy" in error_str
                        or "responsibleaipolicy" in error_str
                    )
                    if is_policy_violation:
                        policy_violation_count += 1
                        if policy_violation_count < max_policy_violation_retries:
                            logging.warning(
                                f"Policy violation; retrying ({policy_violation_count}/{max_policy_violation_retries}). Error: {e}"
                            )
                            time.sleep(1 + random.random() * 2)
                            continue
                        else:
                            logging.error(
                                f"Policy violation retry limit reached ({max_policy_violation_retries}). Error: {e}"
                            )
                            raise
                    # Non-retriable API error - re-raise
                    raise
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

            # Parse out the message
            response_content = response.choices[0].message.content or ""
            reasoning_trace = response.choices[0].message.model_extra.get(
                "reasoning_content", ""
            )

            if reasoning_trace and "<think>" not in response_content:
                response_content = (
                    f"<think>\n{reasoning_trace}\n</think>\n{response_content}"
                )

            # Handle case where usage is None (can happen with some Azure AI models)
            if response.usage is not None:
                num_input_tokens = response.usage.prompt_tokens
                num_output_tokens = response.usage.completion_tokens
            else:
                logging.warning(
                    "API response missing usage data (response.usage is None)"
                )
                num_input_tokens = 0
                num_output_tokens = 0

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
                print(response_content)
                print("=============END RESPONSE===============")

            return response_content

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            raise e
