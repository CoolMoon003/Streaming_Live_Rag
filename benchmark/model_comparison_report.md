# Streaming-Live-RAG Model Comparison

_Generated 2026-09-20T13:58:33+00:00 (UTC) by `scripts/benchmark_models.py`. All numbers are copied from `benchmark/model_comparison_results.json`; this report ranks nothing and declares no winner._

## Experimental setup

- **Environment:** Windows-10-10.0.26200-SP0 | CPU: AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD (12 logical cores) | RAM: 16.5 GB | Python 3.11.9
- **Ollama:** local instance at `http://localhost:11434`, version 0.34.2; models run **sequentially** (one loaded at a time, unloaded afterwards).
- **Corpus:** `data/processed/phase6_chunks.jsonl` (89 chunks, unchanged).
- **Phase 6 query set:** `data/processed/phase6_eval_queries.json`, 18 queries in file order (P6-Q001, P6-Q002, P6-Q003, P6-Q004, P6-Q005, P6-Q006, P6-Q007, P6-Q008, P6-Q009, P6-Q010, P6-Q011, P6-Q012, P6-Q013, P6-Q014, P6-Q015, P6-Q016, P6-Q017, P6-Q018); unchanged. One cold commit per query, fresh session per query.
- **Retrieval pipeline (identical for every model):** rule-based controller and multi-intent decomposition, BM25 + FAISS dense retrieval, RRF fusion, CrossEncoder reranking, evidence gate/selector, grounded generation prompt, citation validator, deterministic sentence attribution.
- **Ollama options (production configuration, unchanged):** temperature 0.0, num_predict 300, think=False, client timeout 60.0 s.
- **Models tested:** `llama3.2:3b`, `qwen2.5:3b`, `gemma3:4b`, `qwen3:4b`, `qwen3:8b`.
- **Default production model:** `llama3.2:3b` (unchanged by this benchmark).
- **Model-dependent stages:** generation only. A scan of `backend/` found Ollama/LLM-client references only in: `backend/app/llm/grounded_generator.py`, `backend/app/llm/ollama_client.py`, `backend/app/orchestration/streaming_rag_orchestrator.py`. Controller, decomposition, retrieval, reranking, evidence selection and validation do not call the LLM.

## Model configuration

| Model | Size (on disk) | Parameters | Quantization | Role | Run status |
|---|---|---|---|---|---|
| llama3.2:3b | 2.02 GB | 3.2B | Q4_K_M | Production default / demo model | complete |
| qwen2.5:3b | 1.93 GB | 3.1B | Q4_K_M | Comparison candidate | complete |
| gemma3:4b | 3.34 GB | 4.3B | Q4_K_M | Comparison candidate | complete |
| qwen3:4b | 2.50 GB | 4.0B | Q4_K_M | Comparison candidate | complete |
| qwen3:8b | 5.23 GB | 8.2B | Q4_K_M | Comparison candidate | complete |

## Latency comparison

Milliseconds, over queries that generated tokens (server latency: all completed queries). Values are Ollama/pipeline-reported; null = not available.

**TTFT, pipeline (commit → first token)**

| Model | n | mean | median | min | max |
|---|---|---|---|---|---|
| llama3.2:3b | 18 | 7498 | 5702 | 2289 | 17537 |
| qwen2.5:3b | 18 | 7370 | 5467 | 2318 | 17345 |
| gemma3:4b | 18 | 11035 | 9213 | 2505 | 22402 |
| qwen3:4b | 18 | 9193 | 7011 | 2349 | 21757 |
| qwen3:8b | 18 | 15135 | 10965 | 2378 | 38892 |

**TTFT, LLM only (Ollama request → first token)**

| Model | n | mean | median | min | max |
|---|---|---|---|---|---|
| llama3.2:3b | 18 | 7331 | 5565 | 2156 | 17279 |
| qwen2.5:3b | 18 | 7208 | 5316 | 2130 | 17083 |
| gemma3:4b | 18 | 10884 | 9079 | 2370 | 22199 |
| qwen3:4b | 18 | 9043 | 6866 | 2210 | 21546 |
| qwen3:8b | 18 | 14988 | 10833 | 2256 | 38669 |

**LLM generation latency**

| Model | n | mean | median | min | max |
|---|---|---|---|---|---|
| llama3.2:3b | 18 | 11765 | 9149 | 5024 | 27791 |
| qwen2.5:3b | 18 | 9799 | 8187 | 3155 | 20459 |
| gemma3:4b | 18 | 17401 | 12874 | 5605 | 40161 |
| qwen3:4b | 18 | 45980 | 43969 | 38645 | 57609 |
| qwen3:8b | 18 | 26529 | 19904 | 14328 | 66243 |

**Pre-generation (retrieval + select + gate)**

| Model | n | mean | median | min | max |
|---|---|---|---|---|---|
| llama3.2:3b | 18 | 166 | 149 | 122 | 261 |
| qwen2.5:3b | 18 | 162 | 147 | 117 | 261 |
| gemma3:4b | 18 | 151 | 139 | 110 | 230 |
| qwen3:4b | 18 | 150 | 149 | 109 | 210 |
| qwen3:8b | 18 | 147 | 134 | 107 | 223 |

**Server processing per turn**

| Model | n | mean | median | min | max |
|---|---|---|---|---|---|
| llama3.2:3b | 18 | 11932 | 9317 | 5158 | 28007 |
| qwen2.5:3b | 18 | 9961 | 8335 | 3344 | 20657 |
| gemma3:4b | 18 | 17552 | 13007 | 5741 | 40392 |
| qwen3:4b | 18 | 46131 | 44085 | 38785 | 57817 |
| qwen3:8b | 18 | 26677 | 20033 | 14450 | 66445 |

## Token usage

Ollama-reported counts (`prompt_eval_count` / `eval_count`); never estimated.

| Model | turns with counts | mean prompt | mean completion | mean total |
|---|---|---|---|---|
| llama3.2:3b | 18 | 311.9 | 48.2 | 360.2 |
| qwen2.5:3b | 18 | 315.9 | 32.1 | 348.1 |
| gemma3:4b | 18 | 324.0 | 60.2 | 384.2 |
| qwen3:4b | 18 | 296.9 | 300.0 | 596.9 |
| qwen3:8b | 18 | 302.9 | 53.9 | 356.8 |

## Grounding and citation behavior

Groundedness is the project's citation/evidence proxy (`evaluation/metrics.py`), **not** semantic factual truth. Micro = supported claims / all claims; macro = mean of per-turn scores over turns with claims.

| Model | micro | macro | claims supported/total | single-intent micro | multi-intent micro | invalid citations | validator rejections | whole-query refusals | queries with intent-level abstention | errored queries |
|---|---|---|---|---|---|---|---|---|---|---|
| llama3.2:3b | 0.897 | 0.935 | 26/29 | 0.842 | 1.000 | 0 | 0 | 0 | 1 | 0 |
| qwen2.5:3b | 0.833 | 0.889 | 10/12 | 0.818 | 1.000 | 0 | 2 | 0 | 8 | 0 |
| gemma3:4b | 1.000 | 1.000 | 35/35 | 1.000 | 1.000 | 0 | 0 | 0 | 1 | 0 |
| qwen3:4b | 0.160 | 0.169 | 35/219 | 0.126 | 0.289 | 0 | 5 | 0 | 5 | 0 |
| qwen3:8b | 0.882 | 0.917 | 30/34 | 0.917 | 0.800 | 0 | 0 | 0 | 1 | 0 |

Refusal / abstention detail (the Phase 6 file carries no per-query 'should refuse' label, so these are observed behaviors, not scored expectations):

- `llama3.2:3b`: whole-query refusals: none; intent-level abstentions in: P6-Q018.
- `qwen2.5:3b`: whole-query refusals: none; intent-level abstentions in: P6-Q001, P6-Q002, P6-Q003, P6-Q007, P6-Q008, P6-Q013, P6-Q014, P6-Q018.
- `gemma3:4b`: whole-query refusals: none; intent-level abstentions in: P6-Q018.
- `qwen3:4b`: whole-query refusals: none; intent-level abstentions in: P6-Q003, P6-Q009, P6-Q012, P6-Q013, P6-Q015.
- `qwen3:8b`: whole-query refusals: none; intent-level abstentions in: P6-Q018.

## Query-level results

| Query | Model | Type | Status | Groundedness (supported/total) | Invalid cit. | Citation valid | Pipeline TTFT ms | LLM gen ms | Server ms | Error |
|---|---|---|---|---|---|---|---|---|---|---|
| P6-Q001 | llama3.2:3b | single | answered | 1.00 (1/1) | 0 | True | 2289 | 5024 | 5158 |  |
| P6-Q001 | qwen2.5:3b | single | answered | null | 0 | True | 2318 | 3155 | 3344 |  |
| P6-Q001 | gemma3:4b | single | answered | 1.00 (1/1) | 0 | True | 2505 | 5605 | 5741 |  |
| P6-Q001 | qwen3:4b | single | answered | 0.24 (4/17) | 0 | True | 2349 | 38645 | 38785 |  |
| P6-Q001 | qwen3:8b | single | answered | 1.00 (2/2) | 0 | True | 2378 | 14328 | 14450 |  |
| P6-Q002 | llama3.2:3b | single | answered | 0.33 (1/3) | 0 | True | 5721 | 10657 | 10807 |  |
| P6-Q002 | qwen2.5:3b | single | answered | null | 0 | True | 6231 | 7006 | 7228 |  |
| P6-Q002 | gemma3:4b | single | answered | 1.00 (1/1) | 0 | True | 9490 | 12408 | 12609 |  |
| P6-Q002 | qwen3:4b | single | answered | 0.12 (2/16) | 0 | True | 7197 | 42969 | 43125 |  |
| P6-Q002 | qwen3:8b | single | answered | 1.00 (2/2) | 0 | True | 11320 | 24223 | 24375 |  |
| P6-Q003 | llama3.2:3b | single | answered | 1.00 (1/1) | 0 | True | 5593 | 8428 | 8570 |  |
| P6-Q003 | qwen2.5:3b | single | answered | null | 0 | True | 5427 | 6240 | 6400 |  |
| P6-Q003 | gemma3:4b | single | answered | 1.00 (1/1) | 0 | True | 9297 | 12423 | 12572 |  |
| P6-Q003 | qwen3:4b | single | answered | null | 0 | False | 6663 | 40486 | 40642 |  |
| P6-Q003 | qwen3:8b | single | answered | 1.00 (1/1) | 0 | True | 10519 | 16612 | 16762 |  |
| P6-Q004 | llama3.2:3b | single | answered | 1.00 (1/1) | 0 | True | 5684 | 8198 | 8326 |  |
| P6-Q004 | qwen2.5:3b | single | answered | 0.50 (1/2) | 0 | True | 5933 | 9755 | 9892 |  |
| P6-Q004 | gemma3:4b | single | answered | 1.00 (1/1) | 0 | True | 9268 | 12894 | 13028 |  |
| P6-Q004 | qwen3:4b | single | answered | 0.12 (2/16) | 0 | True | 6748 | 40706 | 40845 |  |
| P6-Q004 | qwen3:8b | single | answered | 0.50 (1/2) | 0 | True | 11141 | 20565 | 20698 |  |
| P6-Q005 | llama3.2:3b | single | answered | 1.00 (1/1) | 0 | True | 5918 | 9470 | 9654 |  |
| P6-Q005 | qwen2.5:3b | single | answered | 1.00 (1/1) | 0 | True | 5269 | 8017 | 8142 |  |
| P6-Q005 | gemma3:4b | single | answered | 1.00 (2/2) | 0 | True | 9020 | 14466 | 14596 |  |
| P6-Q005 | qwen3:4b | single | answered | 0.13 (2/15) | 0 | True | 6606 | 42309 | 42459 |  |
| P6-Q005 | qwen3:8b | single | answered | 0.50 (1/2) | 0 | True | 10247 | 18066 | 18173 |  |
| P6-Q006 | llama3.2:3b | single | answered | 1.00 (1/1) | 0 | True | 6036 | 10743 | 11005 |  |
| P6-Q006 | qwen2.5:3b | single | answered | 0.50 (1/2) | 0 | True | 5507 | 8674 | 8818 |  |
| P6-Q006 | gemma3:4b | single | answered | 1.00 (2/2) | 0 | True | 9159 | 14608 | 14743 |  |
| P6-Q006 | qwen3:4b | single | answered | 0.18 (3/17) | 0 | True | 6934 | 42989 | 43138 |  |
| P6-Q006 | qwen3:8b | single | answered | 1.00 (2/2) | 0 | True | 10788 | 21247 | 21378 |  |
| P6-Q007 | llama3.2:3b | single | answered | 1.00 (1/1) | 0 | True | 5365 | 8091 | 8265 |  |
| P6-Q007 | qwen2.5:3b | single | answered | null | 0 | False | 5241 | 6670 | 6803 |  |
| P6-Q007 | gemma3:4b | single | answered | 1.00 (1/1) | 0 | True | 8812 | 11576 | 11714 |  |
| P6-Q007 | qwen3:4b | single | answered | 0.05 (1/21) | 0 | True | 6243 | 42763 | 42888 |  |
| P6-Q007 | qwen3:8b | single | answered | 1.00 (1/1) | 0 | True | 9875 | 15177 | 15299 |  |
| P6-Q008 | llama3.2:3b | single | answered | 1.00 (1/1) | 0 | True | 7413 | 11251 | 11423 |  |
| P6-Q008 | qwen2.5:3b | single | answered | null | 0 | True | 7248 | 8357 | 8527 |  |
| P6-Q008 | gemma3:4b | single | answered | 1.00 (2/2) | 0 | True | 11182 | 19331 | 19472 |  |
| P6-Q008 | qwen3:4b | single | answered | 0.12 (2/17) | 0 | True | 8941 | 45574 | 45727 |  |
| P6-Q008 | qwen3:8b | single | answered | 1.00 (2/2) | 0 | True | 14609 | 30970 | 31129 |  |
| P6-Q009 | llama3.2:3b | single | answered | 1.00 (1/1) | 0 | True | 5529 | 8827 | 8980 |  |
| P6-Q009 | qwen2.5:3b | single | answered | 1.00 (1/1) | 0 | True | 5424 | 7862 | 8027 |  |
| P6-Q009 | gemma3:4b | single | answered | 1.00 (1/1) | 0 | True | 9113 | 12854 | 12986 |  |
| P6-Q009 | qwen3:4b | single | answered | null | 0 | False | 6647 | 44810 | 44949 |  |
| P6-Q009 | qwen3:8b | single | answered | 1.00 (1/1) | 0 | True | 11350 | 19243 | 19367 |  |
| P6-Q010 | llama3.2:3b | single | answered | 1.00 (1/1) | 0 | True | 5301 | 7703 | 7840 |  |
| P6-Q010 | qwen2.5:3b | single | answered | 1.00 (1/1) | 0 | True | 5370 | 7357 | 7488 |  |
| P6-Q010 | gemma3:4b | single | answered | 1.00 (1/1) | 0 | True | 8999 | 12669 | 12814 |  |
| P6-Q010 | qwen3:4b | single | answered | 0.19 (3/16) | 0 | True | 7089 | 48769 | 48912 |  |
| P6-Q010 | qwen3:8b | single | answered | 1.00 (1/1) | 0 | True | 10323 | 15673 | 15861 |  |
| P6-Q011 | llama3.2:3b | single | answered | 1.00 (2/2) | 0 | True | 6660 | 13184 | 13334 |  |
| P6-Q011 | qwen2.5:3b | single | answered | 1.00 (2/2) | 0 | True | 6300 | 10994 | 11140 |  |
| P6-Q011 | gemma3:4b | single | answered | 1.00 (2/2) | 0 | True | 10385 | 17241 | 17385 |  |
| P6-Q011 | qwen3:4b | single | answered | 0.05 (1/19) | 0 | True | 8714 | 50784 | 50937 |  |
| P6-Q011 | qwen3:8b | single | answered | 1.00 (2/2) | 0 | True | 12780 | 24865 | 25012 |  |
| P6-Q012 | llama3.2:3b | single | answered | 1.00 (1/1) | 0 | True | 5273 | 7595 | 7744 |  |
| P6-Q012 | qwen2.5:3b | single | answered | 1.00 (1/1) | 0 | True | 5344 | 7533 | 7675 |  |
| P6-Q012 | gemma3:4b | single | answered | 1.00 (1/1) | 0 | True | 8802 | 12024 | 12171 |  |
| P6-Q012 | qwen3:4b | single | answered | null | 0 | False | 7126 | 50213 | 50347 |  |
| P6-Q012 | qwen3:8b | single | answered | 1.00 (1/1) | 0 | True | 10042 | 16593 | 16717 |  |
| P6-Q013 | llama3.2:3b | single | answered | 0.50 (1/2) | 0 | True | 5178 | 7587 | 7732 |  |
| P6-Q013 | qwen2.5:3b | single | answered | null | 0 | False | 5275 | 6727 | 6865 |  |
| P6-Q013 | gemma3:4b | single | answered | 1.00 (1/1) | 0 | True | 8816 | 11345 | 11475 |  |
| P6-Q013 | qwen3:4b | single | answered | null | 0 | False | 6296 | 42369 | 42518 |  |
| P6-Q013 | qwen3:8b | single | answered | 1.00 (1/1) | 0 | True | 10276 | 14856 | 14991 |  |
| P6-Q014 | llama3.2:3b | single | answered | 1.00 (1/1) | 0 | True | 5108 | 7344 | 7477 |  |
| P6-Q014 | qwen2.5:3b | single | answered | null | 0 | True | 5184 | 8544 | 8693 |  |
| P6-Q014 | gemma3:4b | single | answered | 1.00 (1/1) | 0 | True | 8609 | 11332 | 11462 |  |
| P6-Q014 | qwen3:4b | single | answered | 0.10 (2/20) | 0 | True | 6626 | 43897 | 44019 |  |
| P6-Q014 | qwen3:8b | single | answered | 1.00 (1/1) | 0 | True | 10128 | 15916 | 16050 |  |
| P6-Q015 | llama3.2:3b | single | answered | 1.00 (1/1) | 0 | True | 7997 | 13511 | 13634 |  |
| P6-Q015 | qwen2.5:3b | single | answered | 1.00 (1/1) | 0 | True | 7730 | 9886 | 10004 |  |
| P6-Q015 | gemma3:4b | single | answered | 1.00 (3/3) | 0 | True | 12184 | 23765 | 23876 |  |
| P6-Q015 | qwen3:4b | single | answered | null | 0 | False | 9779 | 44042 | 44151 |  |
| P6-Q015 | qwen3:8b | single | answered | 1.00 (3/3) | 0 | True | 16881 | 39565 | 39684 |  |
| P6-Q016 | llama3.2:3b | multi | answered | 1.00 (5/5) | 0 | True | 17537 | 26398 | 26658 |  |
| P6-Q016 | qwen2.5:3b | multi | answered | null | 0 | True | 17345 | 20081 | 20343 |  |
| P6-Q016 | gemma3:4b | multi | answered | 1.00 (6/6) | 0 | True | 21246 | 40161 | 40392 |  |
| P6-Q016 | qwen3:4b | multi | answered | 0.33 (5/15) | 0 | True | 21757 | 56124 | 56335 |  |
| P6-Q016 | qwen3:8b | multi | answered | 0.50 (2/4) | 0 | True | 38892 | 58289 | 58513 |  |
| P6-Q017 | llama3.2:3b | multi | answered | 1.00 (4/4) | 0 | True | 17358 | 27791 | 28007 |  |
| P6-Q017 | qwen2.5:3b | multi | answered | null | 0 | True | 16930 | 20459 | 20657 |  |
| P6-Q017 | gemma3:4b | multi | answered | 1.00 (5/5) | 0 | True | 22402 | 37659 | 37863 |  |
| P6-Q017 | qwen3:4b | multi | answered | 0.18 (3/17) | 0 | True | 21182 | 57609 | 57817 |  |
| P6-Q017 | qwen3:8b | multi | answered | 1.00 (5/5) | 0 | True | 38102 | 66243 | 66445 |  |
| P6-Q018 | llama3.2:3b | multi | answered | 1.00 (1/1) | 0 | True | 15005 | 19961 | 20160 |  |
| P6-Q018 | qwen2.5:3b | multi | answered | 1.00 (1/1) | 0 | True | 14581 | 19062 | 19254 |  |
| P6-Q018 | gemma3:4b | multi | answered | 1.00 (3/3) | 0 | True | 19342 | 30851 | 31039 |  |
| P6-Q018 | qwen3:4b | multi | answered | 0.38 (5/13) | 0 | True | 18580 | 52586 | 52768 |  |
| P6-Q018 | qwen3:8b | multi | answered | 1.00 (1/1) | 0 | True | 32779 | 45089 | 45276 |  |

Full answers, controller/reuse fields, retrieval calls, intent split and evidence IDs are in the JSON.

## Observed trade-offs

- Median LLM generation latency ranged from 8187 ms (qwen2.5:3b) to 43969 ms (qwen3:4b) across the models measured in this run.
- Median pipeline TTFT ranged from 5467 ms (qwen2.5:3b) to 10965 ms (qwen3:8b) across the models measured in this run.
- Median server latency per turn ranged from 8335 ms (qwen2.5:3b) to 44085 ms (qwen3:4b) across the models measured in this run.
- Mean completion tokens per turn ranged from 32.1 (qwen2.5:3b) to 300.0 (qwen3:4b) across the models measured in this run.
- Mean total tokens per turn ranged from 348.1 (qwen2.5:3b) to 596.9 (qwen3:4b) across the models measured in this run.
- Groundedness (micro) ranged from 0.160 (qwen3:4b) to 1.000 (gemma3:4b) across the models measured in this run.
- Groundedness (macro) ranged from 0.169 (qwen3:4b) to 1.000 (gemma3:4b) across the models measured in this run.
- Total invalid citations was identical across models (0).
- Each line above describes one metric in this run only. Metrics are not combined, and no overall ordering is implied.

## Limitations

- Single machine, local Ollama inference, one run per model: latency reflects this hardware and any concurrent load (CPU/RAM/GPU contention was not controlled beyond running models one at a time). Run-to-run variance was **not** measured.
- Small evaluation set (18 queries); a single claim changes a model's micro score noticeably. Differences should not be read as statistically significant.
- Groundedness is a citation/evidence proxy, not a semantic correctness judgement; answers were not human-graded.
- Temperature 0 makes decoding near-deterministic but not guaranteed bit-identical across runs.
- **Fairness / model-dependent stages:** only generation depends on the Ollama model. Retrieval evidence and intent decomposition identical across models in this run: **True** (18 queries compared).
- The production client timeout (default 60 s) was left as-is unless overridden above; a slow model can therefore fail with a timeout, which is recorded as an infrastructure error, not as an incorrect RAG answer.
- `think=False` is sent for every model exactly as production does; how each model family honors it was not altered.
- No model run failed or errored.

## Reproducibility

```
ollama serve
# models must already be installed: llama3.2:3b qwen2.5:3b gemma3:4b qwen3:4b qwen3:8b
.venv\Scripts\python -m scripts.benchmark_models          # full run (add --force to replace existing outputs)
.venv\Scripts\python -m scripts.benchmark_models --report-only --force   # rebuild this report from the JSON
```

Protected-file integrity (sha256 before/after): {"added": [], "removed": [], "modified": []}.
