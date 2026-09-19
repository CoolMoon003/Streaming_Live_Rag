import asyncio

from backend.app.retrieval.streaming_retriever import (
    StreamingRetriever,
)


class AsyncStreamingRetriever:
    """
    Async orchestration layer around the existing synchronous
    BM25 + Dense FAISS + RRF + Cross-Encoder pipeline.

    The retrieval itself remains synchronous.

    asyncio.to_thread() prevents the retrieval pipeline from
    blocking the event loop while it is running.
    """

    def __init__(
        self,
        chunks_path: str,
    ):
        self.retriever = StreamingRetriever(
            chunks_path
        )

    async def retrieve(
        self,
        query: str,
        generation_id: int,
    ):

        print(
            f"[RETRIEVAL START] "
            f"generation={generation_id} | "
            f"query='{query}'"
        )

        results = await asyncio.to_thread(
            self.retriever.retrieve,
            query,
        )

        print(
            f"[RETRIEVAL COMPLETE] "
            f"generation={generation_id}"
        )

        return {
            "generation_id": generation_id,
            "query": query,
            "results": results,
        }