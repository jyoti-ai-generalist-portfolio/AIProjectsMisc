import os
import json
from pathlib import Path
from typing import List, Dict,Any

def load_all_documents(json_file:str = "finqa_local_cache/extracted/FinQA-main/dataset/train.json") -> List[Any]:
    documents = []

    # JSON files
    
    
    print(f"[DEBUG] Loading JSON: {json_file}")
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            content = json.load(f)
            print(f"[DEBUG] JSON content loaded successfully from {json_file}")
        print(f"[DEBUG] Loaded {len(content)} JSON docs from {json_file}")
        documents.extend(content)
        return documents
    except Exception as e:
            print(f"[ERROR] Failed to load JSON {json_file}: {e}")


if (__name__ == "__main__"):
    # Example usage
    json_file = "finqa_local_cache/extracted/FinQA-main/dataset/train.json"
    documents = load_all_documents(json_file)
    print(f"[DEBUG] Total documents loaded: {len(documents)}")