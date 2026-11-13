from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from Proj7_VectorStore import VectorStore
from Proj8_LangChainRetriever import MyChromaRetriever
from dotenv import load_dotenv
import os

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY




# ------------------------------------
# 1. Initialize VectorStore
# ------------------------------------

vectorstore = VectorStore("chroma_book_store")
# ------------------------------------
# 2. Initialize LangChain Retriever
# ------------------------------------

retriever = MyChromaRetriever(
    vector_store=vectorstore,
    top_k=3
)


# ------------------------------------
# 3. Initialize Gemini LLM
# ------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0
)

# ------------------------------------
# 4. Create Prompt Template
# ------------------------------------

prompt = PromptTemplate.from_template("""
You are a helpful assistant.

Answer the question using ONLY the
provided context.

If the answer is not available in the
context, say that you don't know.

Context:
{context}

Question:
{question}

Answer:
""")


# ------------------------------------
# 5. Format Retrieved Documents
# ------------------------------------
def format_docs(docs):

    return "\n\n".join(
        doc.page_content for doc in docs
    )

# ------------------------------------
# 6. Create LangChain Chain
# ------------------------------------
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)


# ------------------------------------
# 7. Execute RAG Chain
# ------------------------------------

if __name__ == "__main__":

    question = "Who is the author of Pride and Prejudice?"

    answer = rag_chain.invoke(question)

    print("\nQuestion:", question)
    print("\nAnswer:", answer)