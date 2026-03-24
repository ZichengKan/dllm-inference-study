## vLLM Serving: PagedAttention vs HuggingFace Inference

Model: Qwen2.5-0.5B-Instruct  
Hardware: NVIDIA RTX 4060 Laptop GPU (8GB VRAM)  
Environment: WSL2 Ubuntu 24.04

### Results

| Method | Latency | Throughput | Speedup |
|--------|---------|-----------|---------|
| HuggingFace (sequential) | 4.97s | 50.3 tok/s | 1x |
| vLLM (PagedAttention) | 0.31s | 817.0 tok/s | **16x** |

### Why vLLM is 16x Faster

**Batching**: HuggingFace processes requests sequentially (one by one).
vLLM processes all 5 requests simultaneously, maximizing GPU utilization.

**Flash Attention 2**: vLLM automatically uses FlashAttention backend,
which reduces attention computation from O(N²) memory to O(N) by
processing in tiles that fit in SRAM, avoiding slow HBM reads/writes.

**torch.compile**: vLLM compiles the model's compute graph to optimized
machine code via PyTorch's JIT compiler. First run takes ~10s to compile,
subsequent runs use the cache.

### PagedAttention Key Stats

- GPU KV cache pool: 355,840 tokens managed
- Max concurrency at 512 tokens/request: 695 simultaneous requests
- Memory waste from fragmentation: near 0% (vs 60-80% in naive KV Cache)

### Core Concepts

**KV Cache problem**: Standard KV Cache pre-allocates fixed contiguous
memory per request. Since sequence lengths vary, most allocated memory
goes unused → 60-80% VRAM wasted on fragmentation.

**PagedAttention solution**: Divides KV Cache into fixed-size blocks
(like OS virtual memory pages). Blocks are allocated on demand and
don't need to be contiguous. A block table maps logical to physical
blocks. Result: near-zero fragmentation, same VRAM serves far more
requests.

**Continuous Batching**: Instead of waiting for all requests in a batch
to finish before starting new ones, vLLM slots in new requests the
moment a slot frees up. GPU stays fully utilized at all times.
