from sentence_transformers import SentenceTransformer


class EmbeddingManager:
    def __init__(self, model_name: str= "BAAI/bge-large-en-v1.5"):
        self.model_name = model_name
        self.model = None
        self.load_model(model_name)
        self.embeddings = None  # Placeholder for embeddings, will be set after embedding generation

    def load_model(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        # Load the embedding model based on the provided model name
        # This is a placeholder for actual model loading logic
        try:
            # Example: Load a model using a hypothetical library
            # self.model = SomeEmbeddingLibrary.load_model(model_name)
            print(f"[DEBUG] Loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name)  # Example using SentenceTransformer
            # Simulate model loading
            print(f"[DEBUG] Loaded model: {model_name}")
        except Exception as e:
            print(f"[ERROR] Failed to load embedding model {model_name}: {e}")
        return 

    def embed_documents(self, texts: list):
        # Generate embeddings for the given texts using the loaded model
       
        
        if self.model is None:
            print("[ERROR] Embedding model is not loaded. Please load a model first.")
            return None
        self.embeddings =self.model.encode(texts).tolist()  
        #print(f"[DEBUG] Generated embeddings for texts: {texts}") 
        
        return self.embeddings  # Replace with actual embedding generation code


if __name__ == "__main__":
    # Example usage
    embedding_manager = EmbeddingManager(model_name="BAAI/bge-large-en-v1.5")
    sample_text = ["This is a sample text for embedding.","This is the second string for embedding."]
    embeddings = embedding_manager.embed_documents(sample_text)
    print(f"[DEBUG] Generated embeddings: {len(embeddings) if embeddings is not None else 'None'}")