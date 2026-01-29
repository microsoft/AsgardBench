import base64
import logging
import random
import threading
import time

from openai import APIError, APITimeoutError, OpenAI, RateLimitError
from PIL import Image

from Magmathor.Model.prompt_templates import strip_cache_marker


class VLLMActor:

    def __init__(
        self,
        endpoints: list[str],
        temperature: float = 0.0,
        max_completion_tokens: int = 4096,
        model_name: str | None = None,
        expected_model_path: str | None = None,
    ):
        """
        Create a vLLM client with endpoint rotation.

        Args:
            endpoints: List of vLLM server URLs (e.g., ["http://localhost:8000/v1", "http://localhost:8001/v1"]).
            temperature: Sampling temperature.
            max_completion_tokens: Maximum tokens in the response.
            model_name: Model name to use in requests. If None, will be auto-detected from the server.
            expected_model_path: If provided, validates that all endpoints serve this model path.
        """
        if not endpoints:
            raise ValueError("At least one vLLM endpoint must be provided.")

        self.endpoints = endpoints
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens

        # Validate model paths if expected_model_path is provided
        if expected_model_path:
            self._validate_model_paths(expected_model_path)

        # Client cache and rotation
        self._client_cache: dict[str, OpenAI] = {}
        self._lock = threading.Lock()
        self._rr_counter = random.randint(0, len(endpoints))

        # Auto-detect or use provided model name
        self.model_name = model_name or self._detect_model_name()

        print(f"Initialized VLLMActor with {len(self.endpoints)} endpoint(s)")
        print(f"Model name: {self.model_name}")

    def _validate_model_paths(self, expected_model_path: str) -> None:
        """
        Validate that all endpoints serve the expected model path.

        Args:
            expected_model_path: The model path that should be served by all endpoints.

        Raises:
            ValueError: If any endpoint serves a different model or multiple models.
        """
        import requests as http_requests

        # Normalize expected path (remove trailing slash for comparison)
        expected_normalized = expected_model_path.rstrip("/")

        print(f"Validating all endpoints serve model: {expected_model_path}")

        errors = []
        for endpoint in self.endpoints:
            # Strip /v1 suffix if present for the models endpoint
            base_url = endpoint.rstrip("/")
            if base_url.endswith("/v1"):
                base_url = base_url[:-3]

            try:
                response = http_requests.get(f"{base_url}/v1/models", timeout=10)
                response.raise_for_status()
                data = response.json()

                models = data.get("data", [])

                if len(models) == 0:
                    errors.append(f"  {endpoint}: No models found")
                    continue

                if len(models) > 1:
                    model_roots = [m.get("root", "unknown") for m in models]
                    errors.append(f"  {endpoint}: Multiple models found: {model_roots}")
                    continue

                actual_root = models[0].get("root", "").rstrip("/")

                if actual_root != expected_normalized:
                    errors.append(
                        f"  {endpoint}: Model mismatch!\n"
                        f"    Expected: {expected_normalized}\n"
                        f"    Actual:   {actual_root}"
                    )
                else:
                    print(f"  ✓ {endpoint}: Model path verified")

            except Exception as e:
                errors.append(f"  {endpoint}: Failed to query models endpoint: {e}")

        if errors:
            error_msg = "Model validation failed:\n" + "\n".join(errors)
            raise ValueError(error_msg)

    def _detect_model_name(self) -> str:
        """Query the first available endpoint to detect the served model name."""
        import requests as http_requests

        for endpoint in self.endpoints:
            # Strip /v1 suffix if present for the models endpoint
            base_url = endpoint.rstrip("/")
            if base_url.endswith("/v1"):
                base_url = base_url[:-3]

            try:
                response = http_requests.get(f"{base_url}/v1/models", timeout=10)
                response.raise_for_status()
                data = response.json()
                model_id = data["data"][0]["id"]
                print(f"Auto-detected model: {model_id} from {endpoint}")
                return model_id
            except Exception as e:
                logging.warning(f"Failed to detect model from {endpoint}: {e}")
                continue

        raise RuntimeError(
            f"Could not detect model name from any endpoint: {self.endpoints}"
        )

    def _get_client(self) -> OpenAI:
        """Return a rotated OpenAI client pointing to a vLLM endpoint (thread-safe)."""
        with self._lock:
            endpoint = self.endpoints[self._rr_counter % len(self.endpoints)]
            self._rr_counter += 1

            if endpoint not in self._client_cache:
                # Ensure endpoint has /v1 suffix for OpenAI client
                base_url = endpoint.rstrip("/")
                if not base_url.endswith("/v1"):
                    base_url = f"{base_url}/v1"

                self._client_cache[endpoint] = OpenAI(
                    base_url=base_url,
                    api_key="EMPTY",  # vLLM doesn't require a real API key
                )
                print(f"Created new vLLM client for endpoint: {endpoint}")

            return self._client_cache[endpoint]

    def get_response(
        self,
        current_image_path: str | None,
        previous_image_path: str | None,
        prompt: str,
    ) -> str:
        """
        Get AI response from the vLLM server.

        Args:
            current_image_path: Path to the current image file. None for text-only mode.
            previous_image_path: Path to the previous image file. None if not using previous image.
            prompt: The user's message/prompt.

        Returns:
            The AI response content or error message.
        """
        try:
            # Build image content if provided
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

            # Get a rotated client
            client = self._get_client()

            # Retry logic for transient errors
            attempt = 0
            while True:
                attempt += 1
                try:
                    response = client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_completion_tokens,
                    )
                    break  # success
                except RateLimitError as e:
                    logging.warning(
                        f"Rate limited (429); retrying after short delay (attempt {attempt}). Error: {e}"
                    )
                    time.sleep(2)
                    continue
                except APITimeoutError as e:
                    logging.warning(
                        f"Timeout error; retrying after short delay (attempt {attempt}). Error: {e}"
                    )
                    time.sleep(2)
                    continue
                except APIError as e:
                    status = getattr(e, "status_code", None)
                    if status in {500, 502, 503, 504}:
                        logging.warning(
                            f"Transient API error {status}; retrying after short delay (attempt {attempt}). Error: {e}"
                        )
                        time.sleep(2)
                        continue
                    raise

            response_content = response.choices[0].message.content

            # Detailed logging for cost estimation
            print("===============PROMPT=================")
            print(f"[PROMPT_CHARS: {len(prompt)}]")
            if image_path is not None:
                import os

                img = Image.open(image_path)
                image_bytes = os.path.getsize(image_path)
                print(f"[IMAGE_SIZE: {img.size[0]}x{img.size[1]}]")
                print(f"[IMAGE_BYTES: {image_bytes}]")
            else:
                print("[IMAGE_SIZE: NONE]")
                print("[IMAGE_BYTES: 0]")
            print(prompt)
            print("=============END PROMPT===============")

            assert response_content is not None, "Unexpected empty response content"

            print("===============RESPONSE=================")
            print(f"[RESPONSE_CHARS: {len(response_content)}]")
            print(response_content)
            print("=============END RESPONSE===============")

            return response_content

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(error_msg)
            raise e
