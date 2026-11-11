import time
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings


sample_text = """
Artificial intelligence has transformed modern software engineering. Machine learning models now assist developers with code completion, automated testing, and bug detection. These tools reduce boilerplate work and speed up release cycles.

On the other hand, baking sourdough bread requires patience and fermentation science. A healthy starter relies on wild yeast and lactobacilli working together. Temperature and hydration levels determine the final crumb structure and crust thickness.

Finally, orbital mechanics governs how satellites remain in stable paths around Earth. Low Earth Orbit satellites travel at roughly 7.8 kilometers per second to balance gravitational pull. Geostationary satellites orbit much higher to match Earth's rotation period.
""".strip()


def evaluate_chunks(name: str, chunks: list[str], elapsed_time: float):
    lengths = [len(c) for c in chunks]
    avg_len = sum(lengths) / len(lengths) if lengths else 0
    print(f"=== {name} ===")
    print(f"Total Chunks : {len(chunks)}")
    print(f"Time Taken   : {elapsed_time:.4f}s")
    print(f"Chunk Sizes  : min={min(lengths)}, max={max(lengths)}, avg={avg_len:.1f} chars")
    for i, chunk in enumerate(chunks, 1):
        preview = chunk.replace("\n", " ")
        print(f"  [{i}] ({len(chunk)} chars): {preview}")
    print()

# 1. Fixed-Size Chunking (splits strictly by character count)
start = time.perf_counter()
fixed_splitter = CharacterTextSplitter(
    separator="",
    chunk_size=250,
    chunk_overlap=30
)
fixed_chunks = fixed_splitter.split_text(sample_text)
evaluate_chunks("Fixed-Size Chunking", fixed_chunks, time.perf_counter() - start)


# 2. Recursive Chunking (splits hierarchically: paragraphs -> lines -> words)
start = time.perf_counter()
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=30,
    separators=["\n\n", "\n", ". ", " ", ""]
)
recursive_chunks = recursive_splitter.split_text(sample_text)
evaluate_chunks("Recursive Chunking", recursive_chunks, time.perf_counter() - start)

# 3. Semantic Chunking (splits when embedding similarity between sentences drops)
"""
 Instead of splitting after a fixed number of characters, the algorithm measures the semantic distance (usually derived from cosine similarity) between consecutive sentences. Wherever that distance crosses your breakpoint threshold, a split occurs.

How It Works in Practice

1. Sentence embedding. The text is broken into individual sentences (often grouped with neighbors in a small sliding window) and converted into numerical vectors using an embedding model.
2. Distance calculation. The splitter computes the cosine distance between each consecutive pair of sentence embeddings. A low distance means the sentences are about the same topic, while a spike in distance signals a topic shift.
3. Threshold comparison. The chunker checks every distance score against your breakpoint threshold. Any spike above the threshold becomes a "breakpoint" where one chunk ends and the next begins.

"""
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
start = time.perf_counter()
semantic_splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=80
)
semantic_chunks = semantic_splitter.split_text(sample_text)
evaluate_chunks("Semantic Chunking", semantic_chunks, time.perf_counter() - start)

