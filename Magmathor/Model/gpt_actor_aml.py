# https://dev.azure.com/msresearch/TRAPI/_wiki/wikis/TRAPI.wiki/16858/Getting-Started-Guide

# Required packages
import base64
import os
import sys

from azure.identity import (
    AzureCliCredential,
    ChainedTokenCredential,
    DefaultAzureCredential,
    ManagedIdentityCredential,
    get_bearer_token_provider,
)
from openai import AzureOpenAI

from Magmathor.Model.prompt_templates import strip_cache_marker

# Constants to avoid circular import
GPT_O3_BASE_NAME = "gpt-o3"
GPT_4O_BASE_NAME = "gpt-4o"


class GPTActorAML:

    def __init__(self, model_name: str, temperature: float):
        """Create the AzureOpenAI client instance."""

        # Ensure this is a valid API version:
        # https://dev.azure.com/msresearch/TRAPI/_wiki/wikis/TRAPI.wiki/15124/Deployment-Model-Information
        """if model_name == GPT_O3_BASE_NAME:
            print("--= Initializing GPT-o3 Model Actor =--")
            self.api_version = "2024-12-01-preview"
            self.deployment_name = "o3_2025-04-16"
        elif model_name == GPT_4O_BASE_NAME:
            print("--= Initializing GPT-4o Model Actor =--")
            self.api_version = "2024-10-21"
            self.deployment_name = "gpt-4o_2024-11-20"
        else:
            raise ValueError(f"Unsupported model name: {model_name}")"""

        self.endpoint = "https://validator-resource.cognitiveservices.azure.com/"

        if model_name == GPT_O3_BASE_NAME:
            print("--= Initializing GPT-o3 Model Actor =--")
            self.model_name = "gpt-4o"
            self.deployment = "gpt-4o"
        elif model_name == GPT_4O_BASE_NAME:
            print("--= Initializing GPT-o3 Model Actor =--")
            self.model_name = "o3"
            self.deployment = "o3"
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

        subscription_key = "{your key here}"

        api_version = "2024-12-01-preview"

        self.model = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=self.endpoint,
            api_key=subscription_key,
        )

        self.temperature = temperature

    def get_response(
        self,
        current_image_path: str | None,
        previous_image_path: str | None,
        prompt: str,
    ):
        """
        Get AI response with optional structured data extraction and debugging.

        Args:
            current_image_path (str|None): Path to the current image file. None for text-only mode.
            previous_image_path (str|None): Path to the previous image file. None if not using previous image.
            prompt (str): The user's message
            prompt_type (PromptType): The type of prompt for debugging categorization
            chat_history (list): Optional list of previous messages
            context_data (dict): Optional context data (can be all available topics or specific context)
            selected_context (dict): Optional selected context from user selections

        Returns:
            tuple: (ai_response_text, structured_data, prompt_info)
                - ai_response_text: Clean text response for display in chat
                - structured_data: Extracted JSON structure or None
                - prompt_info: Dict with info about the prompt sent to AI
        """

        try:
            if self.deployment == "o3":  # "o3_2025-04-16":
                # GPT-o3 Only supports a temprature of 1.0
                temperature = 1.0
            else:
                temperature = self.temperature

            # Build messages array
            # Strip the cache boundary marker (used by OpenRouter for explicit caching).
            clean_prompt = strip_cache_marker(prompt)
            messages = []

            # Text only mode
            if current_image_path is None:
                # Add the current user message
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": clean_prompt},
                        ],
                    }
                )
            else:
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

                # Add current image
                with open(current_image_path, "rb") as image_file:
                    image_data = base64.b64encode(image_file.read()).decode("utf-8")
                image_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_data}"},
                    }
                )

                # Add the current user message with images
                # NOTE: Text comes BEFORE images to maximize prefix caching.
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": clean_prompt},
                            *image_content,
                        ],
                    }
                )

            # Do a chat completion and capture the response
            response = self.model.chat.completions.create(
                model=self.deployment,
                messages=messages,
                temperature=temperature,
            )

            # Parse out the message
            response = response.choices[0].message.content

            return response

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(error_msg)
            raise e
