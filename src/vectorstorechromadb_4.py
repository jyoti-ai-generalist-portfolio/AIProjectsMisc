import chromadb
from chromadb.config import Settings    

from langchain_core.documents import Document
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from langchain_core.documents import Document
from typing import List, Dict, Any
from embedding_4 import EmbeddingManager

class VectorStoreChromaDB:
    def __init__(self, embedding_manager,persist_directory: str = "finqa_local_cache/chromadb"):
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self.embedding_function = CustomEmbeddingWrapper(embedding_manager)
        self.initializeVectorStore()

    def initializeVectorStore(self):
        if (self.persist_directory is None) or (not self.persist_directory.strip()):
            raise ValueError("persist_directory must be a non-empty string.")
        try :
            # create the initial vectore store db
            self.client = chromadb.PersistentClient(path=self.persist_directory)
          
            self.collection = self.client.get_or_create_collection("finqa_collection_bge",
                                                                   embedding_function=self.embedding_function)
        except Exception as e:
            print(f"[ERROR] Failed to initialize vector store: {e}")

    def updateVectorStore(self, documents):
        # Update the vector store with the provided documents
        # This is a placeholder for actual vector store update logic
        try:
            self.collection.upsert(
                documents=[doc['text'] for doc in documents],
                metadatas=[doc.get('metadata', {}) for doc in documents],
                ids=[doc.get('id', None) for doc in documents]
            )
            print (f"[DEBUG] Updated vector store with {len(documents)} documents.")
        except Exception as e:
            print(f"[ERROR] Failed to update vector store: {e}")


    def hybridSearch(self,
            queryList: Optional[list[str]] = None,      # 3rd: Optional (safely defaults to empty list)
            metaList: Optional[dict[str, Any]] = None,  # 2nd: Optional (safely defaults to empty dict)
            top_k: int = 5,                     # 4th: Optional
            searchFullText: bool = False,        # 5th: Optional
            searchWord: str = ""                    # 1st: Mandatory
        ):

        
        """  
        Use .get to retrieve records by ID and/or filters without similarity ranking:
        You can query a collection to run a similarity search using .query:


        For Metadata filtering we use the query with a where clause
            collection.query(query_texts=["first query", "second query"],
                            where={"page": 10}
                            )
            The Where filter dictionary must have the following structure
            {
                "metadata_field": {
                        <Operator>: <Value>
                }
            }

            Equals : 
            {
                "metadata_field": "search_string"
            }

             is equivalent to

            {
                "metadata_field": {
                    "$eq": "search_string"
                }
            }
            # Greater than
            collection.query(
                query_texts=["first query", "second query"],
                where={"page": { "$gt": 10 }}
            )

            #Using And / Or logical operators
            {
                "$and": [
                {
                    "metadata_field": {
                         <Operator>: <Value>
                        }
                 },
                 {
                       "metadata_field": {
                          <Operator>: <Value>
                        }
                 }
                ]
            }
        # In or Not In operators

            {
            "metadata_field": {
                "$in": ["value1", "value2", "value3"]
            }
            }

            {
            "metadata_field": {
                "$nin": ["value1", "value2", "value3"]
            }
            }
        
            collection.query(
                 query_texts=["doc10", "thus spake zarathustra", ...],
                n_results=10,
                where={"metadata_field": "is_equal_to_this"},
                where_document={"$contains":"search_string"}
            )
            
        """
        results = []

        if queryList is None:
            queryList = []
        
        if metaList is None:
            metaList = {}


        if (searchFullText):
            results = self.collection.query(
                query_texts=queryList , # list of queries 
                n_results=top_k,
                where=metaList ,  # Metadata filter as a dictionary
                where_document={"$contains": searchWord},  # Full-text content filter
            )
        else:
            results = self.collection.query(
                            query_texts=queryList , # list of queries 
                            n_results=top_k,
                            where=metaList ,  # Metadata filter as a dictionary
            )


        return results
        



class CustomEmbeddingWrapper(EmbeddingFunction):
    def __init__(self, embedding_manager):
        # Pass your existing embedding manager here
        self.embedding_manager = embedding_manager

    def __call__(self, input: Documents) -> Embeddings:
        # ChromaDB passes texts in the 'input' variable (which is a list of strings)
        # Call your manager's method here (e.g., embed_documents, get_embeddings, etc.)
        return self.embedding_manager.embed_documents(input)

if __name__ == "__main__":
    # Example usage
    embedding_manager = EmbeddingManager(model_name="BAAI/bge-large-en-v1.5")

    vector_store = VectorStoreChromaDB(embedding_manager=embedding_manager, persist_directory="finqa_local_cache/chromadb")
    sample_documents = [
        {"id": "1", "text": "This is the first document.", "metadata": {"source": "doc1"}},
        {"id": "2", "text": "This is the second document.", "metadata": {"source": "doc2"}}
    ]
    
    vector_store.updateVectorStore(sample_documents)
    queryList = ["find which is the first document"]
    metaList = {
                "source": "doc1"
    }
    top_k=2
    results = vector_store.hybridSearch(queryList, metaList, top_k, False)
    print ("Got the results below : ")
    print (results)




