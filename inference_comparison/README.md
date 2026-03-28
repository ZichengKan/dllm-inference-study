## Inference Acceleration Comparison: Quantization vs Serving Framework

Model: Qwen2.5-0.5B-Instruct  
Hardware: NVIDIA RTX 4060 Laptop GPU (8GB VRAM)  
Config: batch=8, max_tokens=50, 5 runs averaged

### Results

| Method | Throughput (tok/s) | Latency (ms/req) | Speedup |
|--------|-------------------|-----------------|---------|
| HF + FP16 (Baseline) | 348 | 143.8 | 1.0x |
| HF + INT4 (Quantization only) | 172 | 291.2 | 0.5x |
| vLLM + FP16 (Serving only) | 1,639 | 30.5 | **4.7x** |
| vLLM + INT4 (Both) | 1,290 | 38.8 | 3.7x |

![Inference Comparison](inference_comparison.png)

### Key Findings

**INT4 quantization alone hurts performance on small models**  
HF + INT4 is 2x slower than HF + FP16 (172 vs 348 tok/s). With a 0.5B
model at batch=8, the bitsandbytes dequantization overhead (INT4→FP16
before matrix multiply) dominates over any memory bandwidth savings.
This confirms the finding from the earlier quantization experiment.

**vLLM alone provides the largest gain (4.7x)**  
PagedAttention + Continuous Batching + Flash Attention 2 together
eliminate the primary bottleneck at this batch size: scheduling overhead
and sequential request processing. The compute pattern is already
memory-bound, and vLLM's batching amortizes weight reads across 8
concurrent requests.

**vLLM + INT4 is slower than vLLM + FP16**  
bitsandbytes INT4 is not fully compatible with vLLM's fused CUDA kernels,
adding dequantization overhead that offsets any bandwidth savings. In
production systems, AWQ or GPTQ quantization formats are preferred for
serving because they are designed for inference-time efficiency and
integrate cleanly with vLLM's kernel fusion.

### Why Optimization Effect Depends on the Bottleneck

These results illustrate a general principle:
- **Bottleneck = scheduling / request batching** → vLLM helps most
- **Bottleneck = VRAM capacity** (model too large to load) → quantization helps most
- **Bottleneck = memory bandwidth** (large model, small batch) → quantization + vLLM together

For a 0.5B model at batch=8 on 8GB VRAM, the bottleneck is batching,
not capacity. Quantization's value becomes clear only for larger models
(7B+) where FP16 would exceed available VRAM entirely.

### Practical Takeaway

For deployment on consumer GPUs:
1. Use vLLM (or similar serving framework) first — always beneficial
2. Add quantization only when the model doesn't fit in VRAM at FP16
3. Prefer AWQ/GPTQ over bitsandbytes NF4 for serving workloads
4. For training/fine-tuning on consumer GPUs, QLoRA (bitsandbytes +
   LoRA) remains the right choice — the overhead is acceptable during
   training where throughput matters less than VRAM fit
