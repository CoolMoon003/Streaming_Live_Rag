from backend.app.retrieval.bm25 import BM25Retriever
from backend.app.retrieval.dense import DenseRetriever
from backend.app.retrieval.fusion import reciprocal_rank_fusion
from backend.app.retrieval.reranker import CrossEncoderReranker


CHUNKS_PATH = "data/processed/chunks.jsonl"


def print_results(title, results):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    for result in results:
        chunk = result["chunk"]

        print(
            f"\n#{result['rank']} "
            f"{chunk['chunk_id']}"
        )

        print(f"Section: {chunk['section']}")

        # Show the correct score depending on retrieval stage
        if "reranker_score" in result:
            print(
                f"Reranker Score: "
                f"{result['reranker_score']:.4f}"
            )
        elif "rrf_score" in result:
            print(
                f"RRF Score: "
                f"{result['rrf_score']:.4f}"
            )
        elif "score" in result:
            print(
                f"Score: "
                f"{result['score']:.4f}"
            )

        print(f"Text: {chunk['text']}")


def main():

    print("Loading retrievers...")

    bm25 = BM25Retriever(CHUNKS_PATH)

    dense = DenseRetriever(CHUNKS_PATH)

    reranker = CrossEncoderReranker()

    print("Retrievers loaded successfully.")

    queries = [
        "What is required for international travel?",

        "Can an employee claim hotel costs when travelling abroad?",

        "What documents are needed for international business expenses?",

        "What determines whether a workshop venue is suitable?",

        "What is the maximum international airfare amount an employee can claim?",
    ]

    for query in queries:

        print("\n\n")
        print("#" * 80)
        print(f"QUERY: {query}")
        print("#" * 80)

        # ---------------------------------------------------------
        # 1. BM25
        # ---------------------------------------------------------

        bm25_results = bm25.search(
            query,
            top_k=10,
        )

        print_results(
            "BM25 RESULTS",
            bm25_results,
        )

        # ---------------------------------------------------------
        # 2. Dense FAISS
        # ---------------------------------------------------------

        dense_results = dense.search(
            query,
            top_k=10,
        )

        print_results(
            "DENSE FAISS RESULTS",
            dense_results,
        )

        # ---------------------------------------------------------
        # 3. Reciprocal Rank Fusion
        # ---------------------------------------------------------

        fused_results = reciprocal_rank_fusion(
            [
                bm25_results,
                dense_results,
            ],
            top_k=10,
        )

        print_results(
            "RRF FUSED RESULTS",
            fused_results,
        )

        # ---------------------------------------------------------
        # 4. Cross-Encoder Reranking
        # ---------------------------------------------------------

        reranked_results = reranker.rerank(
            query=query,
            results=fused_results,
            top_k=5,
        )

        print_results(
            "FINAL CROSS-ENCODER RESULTS",
            reranked_results,
        )


if __name__ == "__main__":
    main()