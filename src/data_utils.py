import json
import os
from torch.utils.data import IterableDataset
from datasets import load_dataset, Audio, Dataset
import torchaudio
import re # Added for regex operations
import pandas as pd # Added for CSV handling
from tqdm import tqdm

class SIFT50MDataset(IterableDataset):
    def __init__(self, sift_dataset: Dataset, base_datasets_paths):
        self.sift_dataset = sift_dataset
        self.base_datasets_paths = base_datasets_paths
        # print(self.sift_dataset) # Commented out for cleaner output
        self.base_dataset_references = self._load_base_dataset_references()

    def _build_common_voice_csv_mapping(self, lang, csv_path):
        # Load the Common Voice dataset
        dataset = load_dataset("mozilla-foundation/common_voice_15_0", lang, split="train", trust_remote_code=True)
        print('mapping csv')
        dataset = dataset.cast_column("audio", Audio(decode=False))
        # Create a list of dictionaries for the CSV
        mapping_data = []
        for entry in tqdm(dataset):
            filename_without_ext = os.path.splitext(os.path.basename(entry['path']))[0]
            mapping_data.append({
                'id': filename_without_ext,
                'audio_path': entry['audio']['path']
            })
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(mapping_data)
        df.to_csv(csv_path, index=False)
        print(f"Common Voice {lang} mapping saved to {csv_path}")
        return df

    def _build_vctk_mapping(self, csv_path,ds_path):
        # Load the Common Voice dataset
        vctk_dataset = torchaudio.datasets.VCTK_092(root=ds_path, download=False)
        vctk_mapping = []
        for i in tqdm(range(len(vctk_dataset))):
            wave, sr, _, speaker_id_vctk, utterance_id = vctk_dataset[i]
            sift_id = f"{speaker_id_vctk}_{utterance_id}"
            vctk_mapping.append({
                'id': sift_id,
                'audio_path': f'{ds_path}/VCTK-Corpus-0.92/wav48_silence_trimmed/{speaker_id_vctk}/{speaker_id_vctk}_{utterance_id}_mic2.flac'
            })
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(vctk_mapping)
        df.to_csv(csv_path, index=True)
        print(f"Vctk mapping saved to {csv_path}")
        return df

    def _build_mls_csv_mapping(csv_path):
        dataset = load_dataset(
            "facebook/multilingual_librispeech",
            "german",
            split="train"
        )
        dataset = dataset.cast_column("audio", Audio(decode=False))  # just get paths
        #print(dataset[0]['audio'])
        def extract_mapping(example):
            return {
                "id": example["id"],
                "audio_path": example["audio"]
            }
    
        # Run in parallel (adjust num_proc to your CPU cores)
        mapped = dataset.map(extract_mapping, num_proc=os.cpu_count(), remove_columns=dataset.column_names)
    
        # Save to CSV
        df = pd.DataFrame(mapped)
        df.to_csv(csv_path, index=False)
        print(f"MLS mapping saved to {csv_path}")
        return df

    def _load_base_dataset_references(self):
        references = {}
        for ds_name, ds_path in self.base_datasets_paths.items():
            print(ds_name, ds_path)
            if ds_name == "common_voice_de":
                # Build CSV mapping for German Common Voice
                csv_path = "./data/common_voice_de_mapping.csv"
                if not os.path.exists(csv_path):
                    self._build_common_voice_csv_mapping("de", csv_path)
                references[ds_name] = pd.read_csv(csv_path)
            elif ds_name == "common_voice_en":
                # Build CSV mapping for English Common Voice
                csv_path = "./data/common_voice_en_mapping.csv"
                if not os.path.exists(csv_path):
                    self._build_common_voice_csv_mapping("en", csv_path)
                references[ds_name] = pd.read_csv(csv_path)
            elif ds_name == "multilingual_librispeech_de":
                '''csv_path = "mls_mapping_en.csv"
                if not os.path.exists(csv_path):
                    self._build_mls_csv_mapping(csv_path)'''
                dataset = load_dataset("facebook/multilingual_librispeech", "german", split="train")
                df = dataset.to_pandas()
                df.set_index('id', inplace=True)
                references[ds_name] = df
            elif ds_name == "vctk_en":
                # For VCTK, pre-build a mapping for faster lookups
                csv_path = "./data/vctk_mapping.csv"
                if not os.path.exists(csv_path):
                    self._build_vctk_mapping(csv_path,ds_path)
                references[ds_name] = pd.read_csv(csv_path)
        return references

    def _get_audio_path_from_base_dataset(self, data_source, target_id):
        if data_source == "common_voice_de" or data_source == "common_voice_en":
            # Common Voice now uses a pre-built CSV mapping
            cv_df = self.base_dataset_references[data_source]
            matching_rows = cv_df[cv_df['id'] == target_id]
            if len(matching_rows) > 0:
                return matching_rows.iloc[0]['audio_path']
        elif data_source == "multilingual_librispeech_de":

            mls_df = self.base_dataset_references[data_source]
            if target_id in mls_df.index:
                return mls_df.loc[target_id]['audio']  # Return full audio data
            return None
        elif data_source == "vctk_en":
            # VCTK now uses a pre-built mapping
            vctk_mapping = self.base_dataset_references[data_source]
            matching_rows = vctk_mapping[vctk_mapping['id'] == target_id]
            if len(matching_rows) > 0:
                return matching_rows.iloc[0]['audio_path']
        return None

    def _process_content_list(self, content_list, data_source, target_ids):
        # This function recursively processes the list of dictionaries in 'content'
        found_path = None

        # Ensure we are always working with an iterable list of items
        if not isinstance(content_list, (list, tuple)):
            content_list = [content_list]

        for item in content_list:
            if isinstance(item, dict):
                for key, value in item.items():
                    if key == 'audio_path' and value is not None:
                        filename_without_ext = os.path.splitext(os.path.basename(value))[0]
                        for target_id in target_ids:
                            if filename_without_ext == target_id:
                                # Use the new helper function for lazy loading
                                mapped_audio_path = self._get_audio_path_from_base_dataset(data_source, target_id)
                                if mapped_audio_path:
                                    item[key] = mapped_audio_path
                                    found_path = mapped_audio_path
                                    break # Found a match, move to next item
                    elif isinstance(value, (list, tuple)):
                        # Recursively call for nested lists/tuples of dictionaries
                        nested_found_path = self._process_content_list(value, data_source, target_ids)
                        if nested_found_path: 
                            found_path = nested_found_path
                    elif isinstance(value, dict):
                        # Recursively call for nested dictionaries
                        nested_found_path = self._process_content_list([value], data_source, target_ids)
                        if nested_found_path: 
                            found_path = nested_found_path
        return found_path

    def __iter__(self):
        for entry in self.sift_dataset:
            
            data_source = entry['data_source']
            sift_entry_id = entry['id']

            # Process SIFT-50M ID to get target IDs for matching
            processed_sift_id_string = re.sub(r"^comparison_", "", sift_entry_id)
            target_ids = processed_sift_id_string.split("__")
            
            # Ensure 'message' is a list of dictionaries and create a mutable copy
            modified_message = entry['messages'].copy() if isinstance(entry['messages'], list) else []
            
            total_found_audio_path = None

            # Iterate through the top-level list (user/assistant roles)
            for role_entry in modified_message:
                if isinstance(role_entry, dict) and 'content' in role_entry and isinstance(role_entry['content'], (list, tuple)):
                    # Process the inner 'content' list of dictionaries
                    current_found_path = self._process_content_list(role_entry['content'], data_source, target_ids)
                    if current_found_path: 
                        total_found_audio_path = current_found_path
            entry['messages'] = modified_message # Update the entry with modified message

            yield {
                'audio_path': total_found_audio_path, 
                'metadata': entry
            }
