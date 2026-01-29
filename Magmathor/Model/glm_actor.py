from typing import Optional

import torch
from transformers import AutoModel, AutoTokenizer

from Magmathor.Model.prompt_templates import strip_cache_marker


class GLMActor:
    def __init__(self, model_path: str, temperature: float):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True
        )
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=dtype,
        ).eval()
        self.temperature = temperature

    def get_response(
        self,
        current_image_path: Optional[str],
        previous_image_path: Optional[str],
        prompt: str,
    ) -> str:
        try:
            from PIL import Image  # Lazy import to avoid dependency if unused

            images = []

            # Add previous image first if provided
            if previous_image_path:
                images.append(Image.open(previous_image_path).convert("RGB"))

            # Add current image if provided
            if current_image_path:
                images.append(Image.open(current_image_path).convert("RGB"))

            # GLM chat API - use single image or None
            # Note: If GLM doesn't support multiple images, we use only the current one
            image = images[-1] if images else None

            # GLM chat API (trust_remote_code=True) supports temperature and optional image
            # history unused for single-turn inference
            # Strip the cache boundary marker (used by OpenRouter for explicit caching).
            clean_prompt = strip_cache_marker(prompt)
            response, _ = self.model.chat(
                self.tokenizer,
                clean_prompt,
                image=image,
                history=None,
                temperature=self.temperature,
                max_new_tokens=1500,
            )
            return (
                response.strip() if isinstance(response, str) else str(response).strip()
            )
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(error_msg)
            raise e
