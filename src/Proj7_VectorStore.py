# Class to manage the vector store in ChromaDB, store documents and retrieve results
import chromadb
import logging

class VectorStore:
    def __init__(self, persistent_directory: str = "chroma_book_store"):
        self.persistent_directory = persistent_directory
        self.client = None
        self.collection = None
        self.createVectorStore(self.persistent_directory)
        

    def createVectorStore(self,persistent_directory: str = "chroma_book_store"):
        try :
            # create the initial vectore store db
            self.client = chromadb.PersistentClient(path=persistent_directory)
            self.collection = self.client.get_or_create_collection("bookstore")
            
        except Exception as e:
            logging.exception(f"[ERROR] Failed to initialize vector store: {e}")


    def updateVectorStore(self, chunks, batch_size: int = 250):
        total_chunks = len(chunks)
        try:
            for i in range(0, total_chunks, batch_size):
                batch = chunks[i : i + batch_size]
                self.collection.upsert(
                documents=[doc["text"] for doc in batch],
                metadatas=[doc.get("metadata", {}) for doc in batch],
                ids=[doc.get("id") for doc in batch],
                )
                processed = min(i + batch_size, total_chunks)
                print(f"Inserted {processed}/{total_chunks} chunks into ChromaDB...")
                logging.info(
                "UpdatingVectorStore, [DEBUG] Inserted"
                f" {processed}/{total_chunks} chunks."
                )

            print(f"Total documents in collection: {self.collection.count()}")
        except Exception as e:
            logging.exception(
            f"UpdatingVectorStore,[ERROR] Failed to update vector store: {e}"
            )

    def hybridSearch(self,
                queryList: Optional[list[str]] = None,      # 3rd: Optional (safely defaults to empty list)
                metaList: Optional[dict[str, Any]] = None,  # 2nd: Optional (safely defaults to empty dict)
                top_k: int = 5,                     # 4th: Optional
                searchFullText: bool = False,        # 5th: Optional
                searchWord: str = ""                    # 1st: Mandatory
            ):
        results = []
        if queryList is None:
            queryList = []
                
        where_filter = metaList if metaList else None
        
        if (searchFullText):
            results = self.collection.query(
                        query_texts=queryList , # list of queries 
                        n_results=top_k,
                        where=where_filter ,  # Metadata filter as a dictionary
                        where_document={"$contains": searchWord},  # Full-text content filter
            )
        else:
            results = self.collection.query(
                        query_texts=queryList , # list of queries 
                        n_results=top_k,
                        where=where_filter   # Metadata filter as a dictionary
            )

        return results
    

if __name__ == "__main__":
    logging.basicConfig(
                filename="output/log_bookstore.log",
                level=logging.INFO,
                format="%(asctime)s, %(message)s",
            )
    query_texts = ["Who is the partner of Sherlock Holmes ?"]
    vectorstore = VectorStore("chroma_book_store")
    results = vectorstore.hybridSearch(queryList=  query_texts ,top_k = 3, searchFullText=False)
    print ("Got results")
    print (results)