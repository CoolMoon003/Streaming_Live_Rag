# Streaming Live RAG - Architecture Ablation Matrix

> **NOTE**: Full benchmark run.Not a final live generation benchmark.

Evaluation comparing the four explicit pipeline configurations on the standard streaming benchmark cases.

| Configuration | Early Retrieval | Reuse / Delta | Multi-Intent | Early Retr. Calls | Exact Reuses | Fresh Commit Calls | Multi-Intent Handled | Mean Pre-Gen (ms) | Groundedness |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **full** | ON | ON | ON | 15 | 3 | 7 | 3 | 99.12 | requires_live_generation |
| **baseline** | OFF | OFF | OFF | 0 | 0 | 8 | 0 | 125.14 | requires_live_generation |
| **no_reuse** | ON | OFF | ON | 15 | 0 | 11 | 3 | 130.26 | requires_live_generation |
| **no_multi_intent** | ON | ON | OFF | 15 | 3 | 5 | 0 | 87.9 | requires_live_generation |

## Fairness and Invariant Verification

- **baseline**: Confirmed 0 early retrieval calls, 0 exact reuses, 0 multi-intent branches.
- **no_reuse**: Confirmed early retrieval executes during partials, but commit performs fresh retrieval.
- **no_multi_intent**: Confirmed early retrieval and reuse operate normally, but multi-intent decomposition is bypassed.
- **full**: Confirmed all streaming advantages (early retrieval, cache/delta reuse, and multi-intent subquerying) operate simultaneously.