
from typing import Any, Optional

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

from Proj7_VectorStore import VectorStore

class MyChromaRetriever(BaseRetriever):

    vector_store: Any
    top_k: int = 5
    metadata_filter: Optional[dict[str, Any]] = None

    def _get_relevant_documents(self,query: str, *,run_manager) -> list[Document]:
        results = self.vector_store.hybridSearch(
            queryList=[query],
            metaList=self.metadata_filter,
            top_k=self.top_k,
            searchFullText=False
        )
        documents = results.get("documents") or [[]]
        metadatas = results.get("metadatas") or [[]]
        ids = results.get("ids") or [[]]
        retrieved_docs = []

        for i, chunk_text in enumerate(documents[0]):

            metadata = (
                metadatas[0][i]
                        if i < len(metadatas[0])
                        else {}
                    )

            metadata = dict(metadata or {})

            # Preserve original ChromaDB document ID
            chunk_id = ids[0][i] if i < len(ids[0]) else None

            if chunk_id:
                metadata["chunk_index"] = chunk_id
            else:
                metadata["chunk_index"] = 0

            # Create LangChain Document
            retrieved_doc = Document(
                page_content=chunk_text,
                metadata=metadata
            )

            retrieved_docs.append(retrieved_doc)

        return retrieved_docs

if __name__ == "__main__":

    vectorstore = VectorStore("chroma_book_store")

    retriever = MyChromaRetriever(
        vector_store=vectorstore,
        top_k=3
    )

    question = "Who is the partner of Sherlock Holmes?"

    results = retriever.invoke(question)

    print("\nRetrieved Documents:\n")

    for doc in results:
        print("Content:", doc.page_content)
        print("Source:", doc.metadata["source"])
        print("Chunk:", doc.metadata["chunk_index"])
        #print("Start:", doc.metadata["start_index"])
        #print("End:", doc.metadata["end_index"])