import torch
import librosa
from dataclasses import dataclass
from transformers import AutoProcessor
import os

@dataclass
class CustomDataCollator:
    def __init__(self, processor: AutoProcessor):
        self.processor = processor

    def __call__(self, samples: list[dict]) -> dict:
        batch_audio_data = []
        batch_text = []

        for sample in samples:
            conversation = sample["messages"]

            # --- find all audio paths ---
            found_audio_paths = []

            def find_audio_paths(content_list):
                paths = []
                if not isinstance(content_list, (list, tuple)):
                    content_list = [content_list]
                for item in content_list:
                    if isinstance(item, dict) and "audio_path" in item and item["audio_path"] is not None:
                        paths.append(item["audio_path"])
                    elif isinstance(item, dict) and "content" in item:
                        paths.extend(find_audio_paths(item["content"]))
                    elif isinstance(item, (list, tuple)):
                        paths.extend(find_audio_paths(item))
                return paths

            for role_entry in conversation:
                if "content" in role_entry and "role" in role_entry and role_entry["role"] != "assistant":
                    found_audio_paths.extend(find_audio_paths(role_entry["content"]))

            # --- load audio ---
            if found_audio_paths:
                audio_signals = []
                for path in found_audio_paths:
                    if os.path.exists(path):
                        audio, _ = librosa.load(
                            path,
                            sr=self.processor.feature_extractor.sampling_rate,
                        )
                        audio_signals.append(audio)
                    else:
                        print(f"File not found: {path}. Skipping this file.")

                if audio_signals:
                    batch_audio_data.extend(audio_signals)
                                # --- process text ---
                    text_formatted = self.processor.apply_chat_template(
                        conversation,
                        add_generation_prompt=False,
                        tokenize=False,
                    )
                    batch_text.append(text_formatted)
                    
            else:
                print("No audio path found in the sample. Skipping.")
                continue


        if not batch_text or not batch_audio_data:
            print("Warning: No valid samples in this batch. Skipping batch.")
            return {}
        # --- process with processor ---
        inputs = self.processor(
            text=batch_text,
            audio=batch_audio_data,
            return_tensors="pt",
            padding=True,
        )

        # --- build labels aligned with input_ids ---
        labels = inputs["input_ids"].clone()
        labels[inputs["attention_mask"] == 0] = -100  # ignore padding tokens

        # --- mask everything before assistant marker ---
        start_marker = self.processor.tokenizer.encode(
            "<|im_start|>assistant", add_special_tokens=False
        )
        start_marker_tensor = torch.tensor(
            start_marker, device=labels.device, dtype=labels.dtype
        )

        for i in range(len(batch_text)):
            input_ids_r = inputs["input_ids"][i]
            ass_start_idx = -1
            for j in range(len(input_ids_r) - len(start_marker_tensor) + 1):
                if torch.all(
                    input_ids_r[j : j + len(start_marker_tensor)]
                    == start_marker_tensor
                ):
                    ass_start_idx = j
                    break

            if ass_start_idx != -1:
                labels[i, : ass_start_idx + len(start_marker_tensor)] = -100
            else:
                labels[i, :] = -100
                print("Warning: assistant start marker not found!")

        inputs["labels"] = labels



        # --- sanity check printouts ---
        '''print(f'input_ids: {inputs["input_ids"].shape}')
        print(f'input_features: {inputs["input_features"].shape}')
        print(f'feature_attention_mask: {inputs["feature_attention_mask"].shape}')
        print(f'attention_mask: {inputs["attention_mask"].shape}')
        print(f'labels: {inputs["labels"].shape}')'''

        return inputs
