import os
from time import time
from google import genai
import time 
import numpy as np
from dotenv import load_dotenv
from google.genai import types

def write_similarity_results_to_file(model_name:str, results, output_file):
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    if not os.path.exists(output_file):
        with open(output_file, "w") as f:
            f.write(f"Output File for Similarity results using different models\n")
    with open(output_file, "a") as f:
        f.write(f"Similarity results for model: {model_name}\n")
        for line, similarity in results:
            f.write(f"{line.strip()} (Similarity: {similarity:.4f})\n")
    print(f"[INFO] Similarity results written to {output_file}")

def read_text_files_from_directory(input_dir):
    lines = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(input_dir, filename), "r", encoding="utf-8") as input_file:
                lines.extend(input_file.readlines())
    return lines


class EmbeddingManager:
    
    def __init__(self, api_key:str, model_name: str = "gemini-embedding-001"):
        self.api_key = api_key
        self.model_name = model_name    
        self.client = genai.Client(api_key=api_key)
        self.embeddings = []

        print(f"[INFO] Loaded embedding model: {model_name}")

    def generate_embeddings(self, lines):
        BATCH_SIZE = 10 
        self.embeddings = []
    
        for i in range(0, len(lines), BATCH_SIZE):
            batch = lines[i:i + BATCH_SIZE]
            doc_response = self.client.models.embed_content(
              model=self.model_name,
              contents=batch,
              config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
            )
        
            batch_embeddings = [item.values for item in doc_response.embeddings]
            self.embeddings.extend(batch_embeddings)
        
            print(f"[INFO] Processed batch {(i // BATCH_SIZE) + 1} of {len(lines) // BATCH_SIZE}")
        
            if (i + BATCH_SIZE) < len(lines):  
                time.sleep(4.5)
        
        print(f"Embedding Process completed successfully! {len(self.embeddings)} embeddings generated for {len(lines)} lines.")
        return

    @staticmethod
    def calculate_cosine_similarity(query_vec, doc_vecs):
        """
        Computes cosine similarity between a single query vector 
        and a matrix of document vectors.
        """
        dot_product = np.dot(doc_vecs, query_vec)
    
        query_norm = np.linalg.norm(query_vec)
        doc_norms = np.linalg.norm(doc_vecs, axis=1)
    
        # Guard against zero-division errors
        doc_norms[doc_norms == 0] = 1e-10
        if query_norm == 0:
            query_norm = 1e-10
            
        scores = dot_product / (query_norm * doc_norms)
        return scores
    
    def search_embeddings(self, query, top_k=5, model_name=None):
        if model_name is None:
            model_name = self.model_name
            
        query_response = self.client.models.embed_content(
                        model=model_name,
                        contents=query,
                        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
        )
        # FIXED: Extracting from the first item element inside the embeddings list attribute
        query_embedding = np.array(query_response.embeddings[0].values, dtype=np.float32)
        
        print("[INFO] Generated embedding for the query.")

        similarity_scores = self.calculate_cosine_similarity(query_embedding, np.array(self.embeddings, dtype=np.float32))
        print("[INFO] Calculated similarities.")
        top_indices = np.argsort(similarity_scores)[-top_k:][::-1]
        return [(lines[i], similarity_scores[i]) for i in top_indices]

# Execution Logic
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")    
input_dir = "data/semantic_data"
lines = read_text_files_from_directory(input_dir)
print(f"Successfully read {len(lines)} lines from text files in the '{input_dir}' directory!")

model_name = "gemini-embedding-001"  
print(f"Starting the Embedding Process with {model_name} model...")
embed_manager = EmbeddingManager(api_key, model_name)
embed_manager.generate_embeddings(lines)

emerging_query = "What is the impact of climate change on biodiversity?"
print(emerging_query)
print("Searching for the most relevant lines in the dataset...")
search_results = embed_manager.search_embeddings(emerging_query, top_k=5)
write_similarity_results_to_file(model_name, search_results, "output/similarity_results.txt")
print("Written to log file successfully! Check the output/similarity_results.txt file for the results.")
