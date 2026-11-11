#In Python, Recall@K and Mean Reciprocal Rank (MRR) are evaluation metrics—rather than retrieval strategies—that you can implement in a few lines of standard Python using sets and list indexing. They measure how well your retriever (like a vector database or BM25) finds and ranks the right documents for a set of test queries.
#• Recall@K measures the proportion of all relevant documents that appear in your top K retrieved results.
#• MRR (Mean Reciprocal Rank) measures how high up the first relevant document appears across all queries, averaging 1 / rank for each query.


#In the code above, recall_at_k(retrieved, relevant, k) is a function that calculates what
# fraction of all truly relevant documents were successfully found inside the top k results returned by your search
#  system.
#It answers one simple question: "Out of every correct document that exists in my database 
# for this question, how many did my retriever manage to grab in its first k picks?"


def recall_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    """Calculates Recall@K for a single query."""
    if not relevant_ids:
        return 0.0
    
    # Slice the top K retrieved results
    top_k = set(retrieved_ids[:k])
    
    # Count how many relevant documents were found in the top K
    hits = len(top_k.intersection(relevant_ids))
    return hits / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list, relevant_ids: set, k: int = None) -> float:
    """Calculates Reciprocal Rank (RR) for a single query."""
    search_space = retrieved_ids[:k] if k else retrieved_ids
    
    # Find the 1-based rank of the FIRST relevant document
    for rank, doc_id in enumerate(search_space, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
            
    return 0.0


def evaluate_retriever(queries_results: list[tuple[list, set]], k: int) -> dict:
    """Computes average Recall@K and MRR@K across a batch of queries."""
    total_recall = 0.0
    total_rr = 0.0
    num_queries = len(queries_results)

    for retrieved, relevant in queries_results:
        total_recall += recall_at_k(retrieved, relevant, k)
        total_rr += reciprocal_rank(retrieved, relevant, k)

    return {
        f"Recall@{k}": round(total_recall / num_queries, 4),
        f"MRR@{k}": round(total_rr / num_queries, 4)
    }


# Format: (ordered_retrieved_doc_ids, set_of_ground_truth_doc_ids)
test_data = [
    (["doc_1", "doc_2", "doc_3", "doc_4"], {"doc_1", "doc_3"}),  # Found 2/2 in top 3; first hit at rank 1 (RR = 1.0)
    (["doc_5", "doc_2", "doc_8", "doc_9"], {"doc_2"}),           # Found 1/1 in top 3; first hit at rank 2 (RR = 0.5)
    (["doc_4", "doc_7", "doc_9", "doc_1"], {"doc_1"})            # Found 0/1 in top 3; first hit at rank 4 (RR@3 = 0.0)
]

scores = evaluate_retriever(test_data, k=3)
print(scores)
# Output: {'Recall@3': 0.6667, 'MRR@3': 0.5}



#Metric	What It Rewards	Best Use Case
#Recall@K	Finding all relevant chunks anywhere inside the top K context window.	Multi-hop RAG pipelines where the LLM needs several pieces of evidence to synthesize an answer.
#MRR	Pushing the single best match to rank #1.	FAQ bots, web search, or single-answer lookups where the top result matters most.

