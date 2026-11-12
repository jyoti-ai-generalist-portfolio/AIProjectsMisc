#This python program implements a simple Chatbot that can answer Literature related queries.
#There is a RAG implemented with selected documents
#We send the query to RAG, retrieve relevant passages and pass it on to Gemini LLM for further work


import os
from langchain_core.documents import Document
from typing import List, Any

from pathlib import Path
import logging
from chonkie import RecursiveChunker
from Proj7_VectorStore import VectorStore

# Step 1 Data Ingestion
def load_documents(dir_path:str="data/literature_docs") -> List[Any]:
    data_path = Path(dir_path).resolve()
    logging.info(f"LoadingDocuments, [DEBUG] Data path: {data_path}")
    documents = []
    # text files
    listofFiles = list(data_path.glob('**/*.txt'))
    logging.info(f"LoadingDocuments, [DEBUG] Found {len(listofFiles)} Text files: {[str(f) for f in listofFiles]}")
    for file in listofFiles:
        try:
            text = file.read_text(encoding="utf-8")
            doc = Document(page_content=text, metadata={"source": str(file)})
            documents.append(doc)
        except Exception as e:
            logging.exception(f"LoadingDocuments,[ERROR] Failed to load Text {file}: {e}")
    return documents

def chunk_documents(documents: list[Any]) -> list[Any]:
    all_chunks =[]
    for doc in documents:
        chunker = RecursiveChunker(chunk_size=1000)
        chunks = chunker.chunk(doc.page_content)
        chunk_idx = 0
        for chunk in chunks:
            #print (f"{doc.metadata["source"]}_{chunk_idx}")
            all_chunks.append({ 
                            "id": f"{doc.metadata["source"]}_{chunk_idx}",
                            "text":chunk.text,
                           "metadata": doc.metadata,
                            "start_index": chunk.start_index,
                            "end_index": chunk.end_index          
                })
            chunk_idx = chunk_idx + 1

        print (f"Completed chunking for {doc.metadata['source']}")

    logging.info(f"Chunking, Chunked {len(documents)} into {len(all_chunks)} ")
    return all_chunks







if __name__ == "__main__":

    # Configure once at the beginning of your program (appends by default)
    logging.basicConfig(
        filename="output/log_bookstore.log",
        level=logging.INFO,
        format="%(asctime)s, %(message)s",
    )

    # Call from anywhere without passing file variables
    logging.info("Process, Comment")
    # Load data
    documents = load_documents("data/literature_docs")
    # chunking
    all_chunks = chunk_documents(documents)
    logging.info(f"Chunking, Completed")
    # Embedding and storing in vector store
    vectorstore = VectorStore("chroma_book_store")
    vectorstore.updateVectorStore(all_chunks)


    logging.info("DataIngestion, Completed Data Ingestion")

    









