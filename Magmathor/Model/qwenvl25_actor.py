import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from Magmathor.Model.prompt_templates import strip_cache_marker


class QwenVLActor:
    def __init__(self, model_path: str, temperature: float):

        local_files_only = model_path != "Qwen/Qwen2.5-VL-7B-Instruct"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=local_files_only,
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.temperature = temperature

    def get_response(
        self,
        current_image_path: str | None,
        previous_image_path: str | None,
        prompt: str,
    ) -> str:
        try:
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
                    image_content.append(
                        {
                            "type": "image",
                            "image": previous_image_path,
                        }
                    )

                # Add current image
                image_content.append(
                    {
                        "type": "image",
                        "image": current_image_path,
                    }
                )

                messages.append(
                    {
                        "role": "user",
                        # NOTE: Text comes BEFORE images to maximize prefix caching.
                        "content": [
                            {"type": "text", "text": clean_prompt},
                            *image_content,
                        ],
                    }
                )

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=1500,
                temperature=self.temperature,
                do_sample=True if self.temperature > 0 else False,
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            return output_text.strip()

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(error_msg)
            raise e
