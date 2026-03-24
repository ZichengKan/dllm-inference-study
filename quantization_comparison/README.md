## Quantization Comparison: FP16 vs INT8 vs INT4

Model: Qwen2.5-0.5B-Instruct  
Hardware: NVIDIA RTX 4060 Laptop GPU (8GB VRAM)  
Library: bitsandbytes

### Results

| Precision | VRAM Usage | Speed (tok/s) | Answer Quality |
|-----------|-----------|---------------|----------------|
| FP16      | 0.92 GB   | 39.9          | Baseline       |
| INT8      | 0.60 GB   | 12.0          | Comparable     |
| INT4 (NF4)| 0.44 GB   | 32.8          | Comparable     |

### Key Findings

**Memory**: INT4 reduces VRAM by ~52% vs FP16. The reduction is less than the
theoretical 4x because activations and KV cache remain in FP16.

**Speed**: INT8 is surprisingly slower than both FP16 and INT4 on a 0.5B model.
This is because bitsandbytes dequantizes INT8 weights back to FP16 before
matrix multiplication, adding overhead that dominates for small models.
INT4 (NF4) has better kernel optimization and avoids this bottleneck.

**Quality**: All three precisions produce comparable outputs on simple factual
and reasoning tasks, confirming that quantization loss is negligible for
inference on this model scale.

**Implication**: For deployment on memory-constrained devices, INT4 (NF4) is
the best trade-off — it uses the least memory, runs faster than INT8, and
preserves output quality. The INT8 speed disadvantage diminishes for larger
models (7B+) where memory bandwidth savings dominate.

### Where Quantization Intervenes in the Data Flow

The full inference pipeline for a causal LM looks like:

    Input text
        → tokenizer → input_ids (integers)
        → Embedding layer → X: (batch, seq_len, d_model) in fp16
        → Transformer Block x N
            each block: LayerNorm → Q/K/V proj → Attention
                        → O proj → FFN (up/gate/down proj)
        → final LayerNorm + lm_head
        → logits → argmax/sample → output token

Quantization only changes how the weight matrices (W) inside each
nn.Linear are stored in VRAM. The data flow (activations X) remains fp16
throughout. At inference time, INT4/INT8 weights are dequantized on-the-fly
back to fp16 before matrix multiplication.

The key insight: quantization saves memory bandwidth (smaller W means
faster transfer from VRAM to compute units), not computation itself.

### About the Model: Qwen2.5-0.5B-Instruct

Qwen2.5 is Alibaba's open-source LLM series. The 0.5B-Instruct variant:
- Parameters: 500M
- Architecture: same as GPT (Transformer decoder-only)
- d_model: 896, n_layers: 24, n_heads: 14, vocab_size: 151,936
- Modern improvements over vanilla GPT:
  RoPE positional encoding, RMSNorm, SwiGLU activation, GQA

Used here because: small enough to run fp16 on 8GB VRAM (0.92GB),
representative of the Qwen2.5 family (0.5B to 72B same architecture),
conclusions scale directly to larger models.
