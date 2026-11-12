from langchain_core.documents import Document
from typing import List, Any
from google.genai import types
import logging
from Proj7_VectorStore import VectorStore
from google import genai
from dotenv import load_dotenv
import os

def ask_gemini(client: genai.client,query: str, retrieved_docs: list[str]) -> str:
    context_text = "\n\n---\n\n".join(retrieved_docs)

    prompt = f"""Context passages:
                {context_text}

                User Question: {query}"""

    response = client.models.generate_content(
         model="gemini-3.6-flash",
         contents=prompt,
         config=types.GenerateContentConfig(
             system_instruction= ("You are a helpful literature assistant. Answer the user's question "
                "strictly using the provided context passages. If the answer cannot be "
                "found in the context, clearly state that the provided excerpts do not "
                "contain enough information."),
            temperature=0.2,
            # Timeout in milliseconds (e.g., 30,000 ms = 30 seconds)
            http_options=types.HttpOptions(timeout=30_000),
            ),
        )
    return response.text



if __name__ == "__main__":
    load_dotenv()
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    model_name = "gemini-3.6-flash"

    gemini_client = genai.Client (api_key=gemini_api_key)
    logging.basicConfig(
            filename="output/log_bookstore.log",
            level=logging.INFO,
            format="%(asctime)s, %(message)s",
        )
    vectorstore = VectorStore("chroma_book_store")

    while True:
        print ("Chatbot: Hello ! How may I help you ?")
        userquery = input("You:")
        if userquery.lower() in ["exit", "quit"]:
            break


        query_texts = [userquery]
        results = vectorstore.hybridSearch(queryList=  query_texts ,top_k = 3, searchFullText=False)
        print ("Chatbot : Here are the relevant passages \n")
        result_documents = results['documents'][0]
        result_metadatas= results['metadatas'][0]
        result_distances = results['distances'][0]
        for doc,meta,dist in zip(result_documents,result_metadatas,result_distances):
            print (f"Source File : {meta['source']}")
            print(f"Distance    : {dist:.4f}")
            print (f"Relevant Passage : \n{doc}")
            print ("_____________")
        print ("Chatbot : Going to ask gemini now...")
        answer = ask_gemini(gemini_client, userquery, result_documents)
        print(f"\nChatbot: {answer}\n")

        