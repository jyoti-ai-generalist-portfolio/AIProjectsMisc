import os
import json
from pathlib import Path
from typing import List, Dict,Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, JSONLoader 

def load_all_documents(json_file:str = "../finqa_local_cache/extracted/FinQA-main/dataset/train.json") -> List[Any]:
    documents = []

    # JSON files
    
    #
    print(f"[DEBUG] Loading JSON: {json_file}")
    try:
        loader = JSONLoader(
                        file_path=json_file,
                        jq_schema=".messages[]",
                        text_content=False  # Required false when the schema returns an object/dict instead of a string
        )
        loaded = loader.load()
        print(f"[DEBUG] Loaded {len(loaded)} JSON docs from {json_file}")
        documents.extend(loaded)
    except Exception as e:
            print(f"[ERROR] Failed to load JSON {json_file}: {e}")


if (__name__ == "__main__"):
    # Example usage
    json_file = "../finqa_local_cache/extracted/FinQA-main/dataset/train.json"
    documents = load_all_documents(json_file)
    print(f"[DEBUG] Total documents loaded: {len(documents)}")