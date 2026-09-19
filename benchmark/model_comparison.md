# Local LLM Model Comparison

## Samsung Streaming Live RAG

All models were evaluated using the same corpus-grounded benchmark prompts.

| Model | Speed Test | Grounded | Missing Fact | Multi-Fact | Adversarial |
|---|---:|---|---|---|---|
| Llama 3.2 3B | 18.81 tok/s | PASS | PASS | PASS | PASS |
| Qwen 2.5 3B | 17.49 tok/s | PASS | PASS | PASS | PASS* |
| Gemma 3 4B | 14.51 tok/s | PASS | PASS | PASS | Not tested |
| Qwen 3 8B | 6.93 tok/s | PASS | PASS | PASS | Not tested |

## Selected Model

Llama 3.2 3B

### Why?

Llama 3.2 3B provided the best overall balance of:

- Corpus grounding
- Rejection of unsupported facts
- Instruction following
- Multi-evidence answering
- Low latency
- Local/offline execution

### Important observation

Qwen 2.5 3B was slightly faster in some tests, but during the adversarial
grounding test it introduced an unsupported inference regarding accommodation
expenses and prior approval.

Therefore, raw generation speed was not used as the only model-selection
criterion.

## Benchmark Environment

- Ollama 0.34.0
- Temperature: 0.0
- Thinking: disabled
- num_predict: 120–150 depending on benchmark
- Local execution
- Same synthetic development corpus
- Same benchmark prompts

## Notes

These results are engineering/model-selection experiments. The final
Samsung evaluation will use the supplied corpus and held-out benchmark.