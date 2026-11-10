from vectorstorechromadb_4 import VectorStoreChromaDB
from embedding_4 import EmbeddingManager
from dataloader_4 import load_all_documents
import os
import csv
import datetime, time



LOG_FILE = "vectorstoredb_4.csv"
def init_log_file():
    """Initializes the CSV log file with headers if it doesn't exist."""
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "DEBUG Comment"])
            

def log_metrics(comment):
    """Appends execution metrics to the CSV file."""
    with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.datetime.fromtimestamp(time.time()).isoformat(),
            comment
        ])



        

def transform_to_chroma_format(raw_data: dict) -> list:
    chroma_docs = []
    file_name = raw_data.get("filename", "unknown_file")
    record_id = raw_data.get("id", "unknown_id")
     # --- 1. Process Text blocks (pre_text & post_text) ---
    # Combine pre_text and post_text into an iterable with descriptive sub-sources
    text_sections = [
        ("pre_text", raw_data.get("pre_text", [])),
        ("post_text", raw_data.get("post_text", []))
    ]

    global_idx = 0
    for section_name, lines in text_sections:
        for line in lines:
            line_str = line.strip()
            # Clean up empty lines or standalone periods
            if not line_str or line_str == ".":
                continue
            chroma_docs.append({
                "id": f"{record_id}_{section_name}_{global_idx}",
                "text": line,
                "metadata": {"source": file_name, "type": section_name}
            })
            global_idx += 1



    # --- 2. Process Tables (table_ori) ---
    # Convert matrix tables into structured descriptive text strings for semantic search
    table_data = raw_data.get("table_ori", [])
    if len(table_data) > 1:
        headers = [h.strip() for h in table_data[0]]
        
        for r_idx, row in enumerate(table_data[1:]):
            row_label = row[0].strip() if row[0] else "Metric"
            for c_idx, val in enumerate(row[1:]):
                # Safeguard index out of bounds matching header
                header_name = headers[c_idx + 1] if (c_idx + 1) < len(headers) else f"Column {c_idx+1}"
                
                table_text_representation = f"{row_label} for {header_name} is {val.strip()}"
                
                chroma_docs.append({
                    "id": f"{record_id}_table_r{r_idx}_c{c_idx}",
                    "text": table_text_representation,
                    "metadata": {
                        "source": file_name,
                        "type": "table_cell",
                        "row_metric": row_label,
                        "timeline": header_name
                    }
                })
    return chroma_docs


if __name__ == "__main__":

    init_log_file()
    # Initialize the embedding manager
    embedding_manager = EmbeddingManager(model_name="BAAI/bge-large-en-v1.5")

    # Initialize the vector store with the embedding manager
    vector_store = VectorStoreChromaDB(embedding_manager=embedding_manager, persist_directory="finqa_local_cache/chromadb")
    log_metrics(f"[DEBUG] Vector store initialized with persist directory: {vector_store.persist_directory}")
    # Load documents from JSON
    json_file = "finqa_local_cache/extracted/FinQA-main/dataset/train.json"
    documents = load_all_documents(json_file)
    log_metrics(f"[DEBUG] Total documents loaded: {len(documents)}")
    # Update the vector store with the loaded documents

    # Example documents for testing
    sample_documents = [
            {"id": "1", "text": "This is the first document.", "metadata": {"source": "doc1"}},
            {"id": "2", "text": "This is the second document.", "metadata": {"source": "doc2"}}
        ]

        
    if documents:
         # 1. Create a master list to hold all flat chunks
        all_chroma_docs = []
        
        # 2. Loop through every JSON object in the dataset
        for doc in documents:
            log_metrics(f"DEBUG: Processing document ID: {doc.get('id', 'unknown_id')}")
            transformed_chunks = transform_to_chroma_format(doc)
            log_metrics (f"[DEBUG] Transformed document ID: {doc.get('id', 'unknown_id')} into {len(transformed_chunks)} chunk segments.")
            all_chroma_docs.extend(transformed_chunks)
        
        log_metrics(f"[DEBUG] Transformed dataset into ChromaDB format: {len(all_chroma_docs)} total chunk segments ready for insertion.")
        total_chunks = len(all_chroma_docs)
          # Batch insertions to prevent memory allocation or timeout errors
        batch_size = 5000
        log_metrics(f"[DEBUG] Starting chunked data insertion (Batch size: {batch_size})...")
        # 3. Update the vector store with the master list
        for i in range(0, total_chunks, batch_size):
            batch = all_chroma_docs[i:i + batch_size]
            log_metrics(f"[DEBUG] Uploading segment slice: items {i} to {min(i + batch_size, total_chunks)}...")
            vector_store.updateVectorStore(batch)
        
        log_metrics (f"[DEBUG] Vector store update completed. Total documents processed: {len(documents)}")


