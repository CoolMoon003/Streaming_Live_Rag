from backend.app.retrieval.bm25 import BM25Retriever
from backend.app.retrieval.dense import DenseRetriever
from backend.app.retrieval.fusion import reciprocal_rank_fusion
from backend.app.retrieval.reranker import CrossEncoderReranker


class StreamingRetriever:
    """
    Connects the existing Phase 1 retrieval pipeline into
    the Streaming Live RAG architecture.

    Pipeline:

        BM25
          +
        Dense FAISS
          ↓
        RRF
          ↓
        Cross-Encoder
    """

    def __init__(
        self,
        chunks_path: str,
        bm25_top_k: int = 10,
        dense_top_k: int = 10,
        rrf_top_k: int = 10,
        final_top_k: int = 5,
    ):

        # clamp so a top_k larger than the corpus never breaks a stage
        self.bm25_top_k = max(1, int(bm25_top_k))
        self.dense_top_k = max(1, int(dense_top_k))
        self.rrf_top_k = max(1, int(rrf_top_k))
        self.final_top_k = max(1, int(final_top_k))

        print("Initializing Streaming Retriever...")

        self.bm25 = BM25Retriever(
            chunks_path
        )

        self.dense = DenseRetriever(
            chunks_path
        )

        self.reranker = CrossEncoderReranker()

        print("Streaming Retriever ready.")

    def retrieve(
        self,
        query: str,
    ) -> dict:

        # ---------------------------------------------------------
        # BM25
        # ---------------------------------------------------------

        bm25_results = self.bm25.search(
            query,
            top_k=self.bm25_top_k,
        )

        # ---------------------------------------------------------
        # Dense FAISS
        # ---------------------------------------------------------

        dense_results = self.dense.search(
            query,
            top_k=self.dense_top_k,
        )

        # ---------------------------------------------------------
        # Reciprocal Rank Fusion
        # ---------------------------------------------------------

        fused_results = reciprocal_rank_fusion(
            [
                bm25_results,
                dense_results,
            ],
            top_k=self.rrf_top_k,
        )

        # ---------------------------------------------------------
        # Cross Encoder
        # ---------------------------------------------------------

        reranked_results = self.reranker.rerank(
            query=query,
            results=fused_results,
            top_k=self.final_top_k,
        )

        return {
            "query": query,
            "bm25_results": bm25_results,
            "dense_results": dense_results,
            "fused_results": fused_results,
            "reranked_results": reranked_results,
        }