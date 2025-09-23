import json
import os
import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset, Audio, Dataset
import torchaudio
import re # Added for regex operations
import pandas as pd # Added for CSV handling
from tqdm import tqdm
import librosa

class SIFT50MDataset(IterableDataset):
    def __init__(self, sift_dataset: Dataset, base_datasets_paths, test_set= False):
        self.sift_dataset = sift_dataset
        self.base_datasets_paths = base_datasets_paths
        self.test_set = test_set
        # print(self.sift_dataset) # Commented out for cleaner output
        self.base_dataset_references = self._load_base_dataset_references()

    def _build_common_voice_csv_mapping(self, lang, csv_path):
        
        # Load the Common Voice dataset
        if self.test_set:
            dataset = load_dataset("mozilla-foundation/common_voice_15_0", lang, split="test", trust_remote_code=True) 
        else:
            dataset = load_dataset("mozilla-foundation/common_voice_15_0", lang, split="train", trust_remote_code=True)
        #print('mapping csv')
        dataset = dataset.cast_column("audio", Audio(decode=False))
        # Create a list of dictionaries for the CSV
        mapping_data = []
        for entry in tqdm(dataset):
            # Get the absolute path from the Hugging Face cache
            local_audio_path = os.path.abspath(entry['audio']['path'])
    
            # Verify if the file exists before adding it to the mapping
            if os.path.exists(local_audio_path):
                filename_without_ext = os.path.splitext(os.path.basename(entry['path']))[0]
                mapping_data.append({
                    'id': filename_without_ext,
                    'audio_path': local_audio_path
                })
            else:
                print(f"File not found: {local_audio_path}. Skipping.")
            
        # Create DataFrame and save to CSV
        df = pd.DataFrame(mapping_data)
        df.to_csv(csv_path, index=True)
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
            #print(ds_name, ds_path)
            if ds_name == "common_voice_de":
                # Build CSV mapping for German Common Voice
                if self.test_set:
                   csv_path = "./data/common_voice_de_test_mapping.csv" 
                else:
                    csv_path = "./data/common_voice_de_mapping.csv"
                if not os.path.exists(csv_path):
                    self._build_common_voice_csv_mapping("de", csv_path)
                references[ds_name] = pd.read_csv(csv_path)
                #print(csv_path)
            elif ds_name == "common_voice_en":
                # Build CSV mapping for English Common Voice
                if self.test_set:
                   csv_path = "./data/common_voice_en_test_mapping.csv" 
                else:
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
        #print(data_source)
        if data_source == "common_voice_de" or data_source == "common_voice_en":
            # Common Voice now uses a pre-built CSV mapping
            cv_df = self.base_dataset_references[data_source]
            #print(target_id)
            matching_rows = cv_df[cv_df['id'] == target_id]
            #print('matching row',matching_rows)
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
        found_urls = [] # Changed from found_paths to found_urls
    
        # Ensure iterable
        if not isinstance(content_list, (list, tuple)):
            content_list = [content_list]
        
        for item in content_list:
            if isinstance(item, dict):
                # Add 'type' for text items
                #print(item)
                if 'text' in item.keys() and item['text'] is not None:
                    item['type'] = 'text'
                
                # Process audio items
                if 'audio_path' in item.keys() and item['audio_path'] is not None:
                    filename_without_ext = os.path.splitext(os.path.basename(item['audio_path']))[0]
                    #print('inside audio loop',filename_without_ext)
                    
                    for target_id in target_ids:
                        if filename_without_ext == target_id:
                            #print('target_id',target_id)
                            mapped_audio_path = self._get_audio_path_from_base_dataset(data_source, target_id)
                            #print(mapped_audio_path)
                            #if os.path.exists(mapped_audio_path):
                            if mapped_audio_path:
                                # Rename the key and add type
                                item['audio_path'] = mapped_audio_path
                                item['type'] = 'audio'
                                found_urls.append(mapped_audio_path)
                            else:
                                print(f'Path not found for id: {target_id}')
                
                # Handle nested lists/dictionaries
                for key, value in item.items():
                    if isinstance(value, (list, tuple)):
                        nested_found_urls = self._process_content_list(value, data_source, target_ids)
                        found_urls.extend(nested_found_urls)
                    elif isinstance(value, dict):
                        nested_found_urls = self._process_content_list([value], data_source, target_ids)
                        found_urls.extend(nested_found_urls)
        
        return found_urls

    def __iter__(self):
        for entry in self.sift_dataset:
            
            data_source = entry['data_source']
            sift_entry_id = entry['id']

            # Process SIFT-50M ID to get target IDs for matching in comparison subset
            #processed_sift_id_string = re.sub(r"^comparison_", "", sift_entry_id)
            #target_ids = processed_sift_id_string.split("__")
            target_ids= [sift_entry_id]
            #print(target_ids)
            
            # Ensure 'message' is a list of dictionaries and create a mutable copy
            modified_message = entry['messages'].copy() if isinstance(entry['messages'], list) else []
            
            found_any_audio = False
            # Iterate through the top-level list (user/assistant roles)
            for role_entry in modified_message:
                if isinstance(role_entry, dict) and 'content' in role_entry.keys() and role_entry['role'] != 'assistant':
                    current_found_path = self._process_content_list(role_entry['content'], data_source, target_ids)
                    #print(current_found_path)
                    if len(current_found_path) > 0:
                        found_any_audio = True

            if found_any_audio:
                entry['messages'] = modified_message
                yield entry
            else:
                # Optional: print or log skipped entries
                print(f"Skipping entry {sift_entry_id}, no valid audio found.")
                continue


