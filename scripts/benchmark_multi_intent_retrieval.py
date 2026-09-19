import asyncio
import time

from backend.app.retrieval.async_streaming_retriever import AsyncStreamingRetriever


CHUNKS_PATH = "data/processed/chunks.jsonl"

QUERIES = [
    "What approval is needed for international travel?",
    "What documentation is needed for reimbursement?",
]


async def sequential(retriever):
    results = []

    start = time.perf_counter()

    for query in QUERIES:
        result = await retriever.retrieve(
            query=query,
            generation_id=1,
        )
        results.append(result)

    elapsed = time.perf_counter() - start

    return elapsed, results


async def parallel(retriever):
    start = time.perf_counter()

    results = await asyncio.gather(
        *[
            retriever.retrieve(
                query=query,
                generation_id=2,
            )
            for query in QUERIES
        ]
    )

    elapsed = time.perf_counter() - start

    return elapsed, results


def get_result_ids(results):
    output = []

    for result in results:
        ids = [
            item["chunk"]["chunk_id"]
            for item in result["results"]["reranked_results"]
        ]
        output.append(ids)

    return output


async def main():
    print("=" * 80)
    print("MULTI-INTENT SEQUENTIAL VS PARALLEL BENCHMARK")
    print("=" * 80)

    retriever = AsyncStreamingRetriever(CHUNKS_PATH)

    # Warm-up: model loading/inference initialization is excluded
    print("\nWarming up...")
    await retriever.retrieve(
        query=QUERIES[0],
        generation_id=0,
    )

    print("\nRunning sequential retrieval...")
    sequential_time, sequential_results = await sequential(retriever)

    print("\nRunning parallel retrieval...")
    parallel_time, parallel_results = await parallel(retriever)

    sequential_ids = get_result_ids(sequential_results)
    parallel_ids = get_result_ids(parallel_results)

    print("\n" + "=" * 80)
    print("RESULT")
    print("=" * 80)

    print(f"Sequential time : {sequential_time:.3f} sec")
    print(f"Parallel time   : {parallel_time:.3f} sec")

    if parallel_time > 0:
        speedup = sequential_time / parallel_time
        print(f"Speedup         : {speedup:.2f}x")

    print(f"Results same    : {sequential_ids == parallel_ids}")

    print("\nSequential result IDs:")
    for i, ids in enumerate(sequential_ids, start=1):
        print(f"  Query {i}: {ids}")

    print("\nParallel result IDs:")
    for i, ids in enumerate(parallel_ids, start=1):
        print(f"  Query {i}: {ids}")


if __name__ == "__main__":
    asyncio.run(main())