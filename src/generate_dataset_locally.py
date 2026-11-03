import urllib.request
import zipfile
import json
import os

# Configuration matching finqa_hf.py
GIT_ZIP_URL = "https://github.com/czyssrs/FinQA/archive/refs/heads/main.zip"
CACHE_DIR = "./finqa_local_cache"
ZIP_DESTINATION = os.path.join(CACHE_DIR, "finqa_repo.zip")
EXTRACTION_DIR = os.path.join(CACHE_DIR, "extracted")

# Step 1: Create directories & download the source data zip file
os.makedirs(CACHE_DIR, exist_ok=True)
if not os.path.exists(ZIP_DESTINATION):
    print("Downloading raw FinQA source repository...")
    urllib.request.urlretrieve(GIT_ZIP_URL, ZIP_DESTINATION)

# Step 2: Extract files locally
if not os.path.exists(EXTRACTION_DIR):
    print("Extracting repository contents...")
    with zipfile.ZipFile(ZIP_DESTINATION, 'r') as zip_reference:
        zip_reference.extractall(EXTRACTION_DIR)

# Step 3: Parse train.json exactly how the original script intended
dataset_target_path = os.path.join(EXTRACTION_DIR, "FinQA-main", "dataset", "train.json")
parsed_examples_list = []

print(f"Parsing records from: {dataset_target_path}")
with open(dataset_target_path, encoding="utf-8") as file_stream:
    raw_json_records = json.load(file_stream)
    for index, example in enumerate(raw_json_records):
        # Extract fields identically to finqa_hf.py's format
        structured_record = { 
            "id": example['id'], 
            "pre_text": example['pre_text'], 
            "post_text": example['post_text'], 
            "table": example['table'], 
            "question": example['qa']['question'],
            "answer": example['qa']['answer'],
            'final_result': str(example['qa']['steps'][-1]['res']),
            "program_re": str(example['qa']['program']),                
            "gold_inds": list(example['qa']['gold_inds'].values())
        }
        parsed_examples_list.append(structured_record)

# Verification summary output
print(f"\n--- Success! ---")
print(f"Total Local Records Generated: {len(parsed_examples_list)}")
print("First record structure visualization:")
print(json.dumps(parsed_examples_list[0], indent=2))
