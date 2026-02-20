# Investigation: NV-Embed-v2 and E5-base-v2 Benchmark Failures

## 1. nvidia/NV-Embed-v2

### Observed Error

Both RTX6000PRO and H100 runs failed with the same error in `server.log`:

```
pydantic_core._pydantic_core.ValidationError: 1 validation error for ModelConfig
  Value error, Model architectures ['NVEmbedModel'] are not supported for now.
```

The vLLM server dies before becoming healthy. No benchmark results are produced.

### Root Cause: Custom Architecture Not Supported by vLLM

NV-Embed-v2 uses a custom `NVEmbedModel` architecture that is **not in vLLM's supported model list**. This is not a configuration issue — it is a fundamental incompatibility.

**Why vLLM cannot support it:**

| Technical Blocker | Details |
|---|---|
| Custom architecture class | Registers as `NVEmbedModel`, not any standard HuggingFace architecture |
| Latent-Attention pooling | Uses a non-standard pooling mechanism where the LLM attends to learned latent vectors — vLLM's embedding pipeline only supports mean/CLS pooling |
| Non-standard forward API | Requires a `prompt` parameter in `model.encode()` and a `latent attention mask` in `model.forward()`, neither of which vLLM's embedding API provides |
| Custom remote code | Depends on `trust_remote_code=True` with custom modeling code on HuggingFace |

**Community has asked, NVIDIA/vLLM have not acted:**

- [vLLM Issue #5826](https://github.com/vllm-project/vllm/issues/5826) — closed as "not planned"
- [vLLM Issue #9868](https://github.com/vllm-project/vllm/issues/9868) — 9 thumbs-up, closed as "not planned"
- [vLLM Issue #12137](https://github.com/vllm-project/vllm/issues/12137) — closed as "not planned"
- [vLLM Issue #15849](https://github.com/vllm-project/vllm/issues/15849) — closed as "not planned"

All issues were closed automatically by stale-bot after 90+ days with no developer engagement.

Additionally, [NVIDIA confirmed on HuggingFace](https://huggingface.co/nvidia/NV-Embed-v2/discussions/6) that NV-Embed-v2 will **not** be supported by NVIDIA NIM either, due to its CC-BY-NC-4.0 (non-commercial) license.

### Model Specs (for reference)

| Property | Value |
|---|---|
| Base LLM | Mistral-7B-v0.1 (decoder-only) |
| Parameters | ~8B |
| Embedding Dimension | 4096 |
| Max Sequence Length | 32,768 tokens |
| MTEB Score | 72.31 (#1 at time of release, Aug 2024) |
| License | CC-BY-NC-4.0 (non-commercial) |

### Verdict

**NV-Embed-v2 cannot be benchmarked with vLLM (any version) or sglang.** There is no workaround short of writing a custom vLLM model implementation. The model should be removed from the sweep config, or benchmarked using a different serving approach (e.g., direct HuggingFace inference with `sentence-transformers`).

---

## 2. intfloat/e5-base-v2

### Observed Error

The server starts successfully, but the benchmark client gets `400 Bad Request`:

```
vllm.exceptions.VLLMValidationError: This model's maximum context length is 512 tokens.
However, your request has 945 input tokens. Please reduce the length of the input messages.
```

### Root Cause: 512-Token Hard Limit vs. Benchmark's Text Generation

**Two factors combine to cause the failure:**

#### Factor 1: E5-base-v2 has a 512-token max sequence length

This is a hard architectural limit inherited from BERT. From the [model card](https://huggingface.co/intfloat/e5-base-v2):

> "Long texts will be truncated to at most 512 tokens."

The model's tokenizer and position embeddings were trained for a maximum of 512 tokens. This cannot be extended.

#### Factor 2: The benchmark's `generate_text()` produces more tokens than expected

In `benchmark_embedding.py`, `generate_text(num_tokens=256)` creates 256 random words of 3-8 characters each. Each word averages ~5.5 characters. With BERT's WordPiece tokenizer, random character strings tokenize to **~1.5-2 subword tokens per word**, so:

```
256 words x ~1.5-2 tokens/word = ~384-512 tokens
```

At `chunk_size=512`, this becomes:

```
512 words x ~1.5-2 tokens/word = ~768-1024 tokens
```

The server log confirms a request with **945 tokens** was sent, which is consistent with `chunk_size=512` generating ~945 actual tokens — nearly 2x the model's 512 limit.

### Is 512 Tokens a Practical Limitation?

**Yes.** 512 tokens (~380-400 English words) is widely recognized as insufficient for modern embedding workloads:

- The [LongEmbed paper (arXiv 2404.12096)](https://arxiv.org/html/2404.12096v3) explicitly states: *"existing embedding models are limited to encoding short documents of typically 512 tokens... this narrow context window has greatly hindered their application."*
- [Jina AI](https://arxiv.org/html/2310.19923v4) built Jina Embeddings v2 specifically to overcome "the 512-token limit of traditional models like BERT"
- Real-world documents (web pages, papers, legal text, code) routinely exceed 512 tokens

### Comparison with Other Models in the Sweep

| Model | Max Seq Length | Factor vs E5 |
|---|---|---|
| **e5-base-v2** | **512** | 1x |
| BGE-M3 | 8,192 | 16x |
| Qwen3-Embedding-8B | 32,768 | 64x |

E5-base-v2 is a 2023-era BERT-based model (110M params). The other models in the sweep are 2024-era models with 16-64x longer context windows. Benchmarking them side-by-side at `chunk_size=256` and `chunk_size=512` creates an uneven comparison because:

- At `chunk_size=256`: E5 barely fits (if token generation is fixed), but the benchmark doesn't test the long-context capabilities of BGE-M3 and Qwen3
- At `chunk_size=512`: E5 overflows entirely

### Is E5-base-v2 Outdated?

Not officially deprecated, but effectively superseded within its own family:

| Model | Max Tokens | BEIR Score |
|---|---|---|
| e5-base-v2 | 512 | 50.3 |
| multilingual-e5-large-instruct | 512 | 52.5 |
| e5-mistral-7b-instruct | 4,096 | 56.9 |

### Verdict

**E5-base-v2 can technically be benchmarked, but only at very short chunk sizes.** The `generate_text()` function inflates token counts by ~1.5-2x vs its `num_tokens` parameter, so even `chunk_size=256` is borderline. Two options:

1. **Fix `generate_text()`** to produce text that actually tokenizes to ~`num_tokens` tokens (divide word count by ~1.5-2)
2. **Remove e5-base-v2 from the sweep** — its 512-token limit makes it non-representative of modern embedding workloads, and comparing it against 8K-32K context models at short chunk sizes is not particularly informative

---

## Summary

| Model | Can Benchmark? | Issue | Resolution |
|---|---|---|---|
| nvidia/NV-Embed-v2 | **No** | Architecture unsupported by vLLM, no fix possible | Remove from sweep |
| intfloat/e5-base-v2 | **Partially** | 512-token limit exceeded by text generator | Fix text generator or remove from sweep |

## Sources

- [nvidia/NV-Embed-v2 Model Card](https://huggingface.co/nvidia/NV-Embed-v2)
- [intfloat/e5-base-v2 Model Card](https://huggingface.co/intfloat/e5-base-v2)
- [vLLM Issue #5826](https://github.com/vllm-project/vllm/issues/5826), [#9868](https://github.com/vllm-project/vllm/issues/9868), [#12137](https://github.com/vllm-project/vllm/issues/12137), [#15849](https://github.com/vllm-project/vllm/issues/15849)
- [HuggingFace Discussion: NV-Embed-v2 Inference Frameworks](https://huggingface.co/nvidia/NV-Embed-v2/discussions/6)
- [LongEmbed: Extending Embedding Models for Long Context Retrieval (arXiv 2404.12096)](https://arxiv.org/html/2404.12096v3)
- [Jina Embeddings 2: 8192-Token Embeddings (arXiv 2310.19923)](https://arxiv.org/html/2310.19923v4)
- [Microsoft unilm E5 README](https://github.com/microsoft/unilm/blob/master/e5/README.md)
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
