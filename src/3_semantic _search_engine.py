import os
from sentence_transformers import SentenceTransformer
import numpy as np
"""
Model Name / Hugging Face ID
Vector Dimensions
Best For
Why Use It?

1. all-MiniLM-L6-v2,384,Fast prototyping & low memory,
The ultimate baseline model. It is incredibly lightweight, fast, and punches well above its weight class for semantic search.
2. bge-large-en-v1.5(BAAI/bge-large-en-v1.5),1024,High-accuracy English retrieval,
Consistently ranks near the top of the MTEB (Massive Text Embedding Benchmark) leaderboard for retrieval tasks.
3. gte-large,1024,General semantic search,
Excellent at capturing semantic similarity across diverse domains (tech, finance, legal).
4. all-mpnet-base-v2,768,High-quality general purpose,
Stronger overall performance than MiniLM, though it requires slightly more compute power.





"""
def write_similarity_results_to_file(model_name:str, results, output_file):
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
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
            with open(os.path.join(input_dir, filename), "r") as input_file:
                lines.extend(input_file.readlines())
    return lines


class EmbeddingManager:

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name  
        self.embeddings = []              
        self.embedding_model = SentenceTransformer(model_name)
        print(f"[INFO] Loaded embedding model: {model_name}")

    def generate_embeddings(self, lines):
        self.embeddings = self.embedding_model.encode(lines, show_progress_bar=True)
        print (f"Embedding Process completed successfully! {self.embeddings.shape} embeddings generated for {len(lines)} lines.")
        return 
    
    def search_embeddings(self, query, top_k=5):
        query_embedding = self.embedding_model.encode([query])
        print ("[INFO] Generated embedding for the query.")
        similarities = np.dot(self.embeddings, query_embedding[0]).flatten()
        print ("[INFO] Calculated similarities.")
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        print ("[INFO] Identified top relevant lines.")
        return [(lines[i], similarities[i]) for i in top_indices]


# Define the input directory for the text files
input_dir = "data/semantic_data"
lines = read_text_files_from_directory(input_dir)
print (f"Successfully read {len(lines)} lines from text files in the '{input_dir}' directory!")

#Embedding Process

#model_name = "sentence-transformers/all-MiniLM-L6-v2"  # You can change this to any of the models listed above
#model_name = "BAAI/bge-large-en-v1.5"  # You can change this to any of the models listed above
#model_name = "thenlper/gte-large"  # You can change this to any of the models listed above
model_name = "sentence-transformers/all-mpnet-base-v2"  # You can change this to any of the models listed above

print (f"Starting the Embedding Process with {model_name} model...")
embed_manager = EmbeddingManager(model_name)
embed_manager.generate_embeddings(lines)

emerging_query = "What is the impact of climate change on biodiversity?"
print (emerging_query)
print ("Searching for the most relevant lines in the dataset...")
search_results = embed_manager.search_embeddings(emerging_query, top_k=5)
write_similarity_results_to_file(model_name, search_results, "output/similarity_results.txt")
print ("Written to log file successfully! Check the output/similarity_results.txt file for the results.")



