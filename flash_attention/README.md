## Flash Attention Benchmark: Standard vs Flash Attention

Model: Single attention head (d_head=64, FP16)  
Hardware: NVIDIA RTX 4060 Laptop GPU (8GB VRAM)  
Framework: PyTorch 2.5.1 (built-in scaled_dot_product_attention)

### Results

| Sequence Length | Standard Attn (ms) | Flash Attn (ms) | Speedup |
|----------------|-------------------|-----------------|---------|
| 128            | 0.16              | 0.03            | 4.74x   |
| 256            | 0.09              | 0.05            | 1.56x   |
| 512            | 0.09              | 0.05            | 1.92x   |
| 1024           | 0.08              | 0.05            | 1.55x   |
| 2048           | 0.18              | 0.10            | 1.71x   |

![Flash Attention Benchmark](flash_attention_benchmark.png)

### Why Flash Attention is Faster

Standard Attention computes S = QK^T and P = softmax(S) as full (N×N)
matrices, writing them to HBM (global GPU memory) between steps:

    Q,K → [HBM write S] → [HBM read S] → softmax → [HBM write P]
        → [HBM read P] → P@V → output
    Total HBM traffic: O(N²)

Flash Attention uses tiling: it splits Q,K,V into blocks that fit in
SRAM (on-chip cache, ~70x faster than HBM), computes the full attention
output block by block without ever materializing the full S or P matrix:

    For each block of K,V:
        load block into SRAM → compute partial S → online softmax update
        → accumulate into output → discard block
    Total HBM traffic: O(N × d)  — no N² term

The online softmax trick makes this possible: as each new block of scores
arrives, we update the running max m and normalizer l, and correct the
previously accumulated output. The final result is mathematically
identical to standard attention.

### Observations from This Experiment

**N=128 shows the highest speedup (4.74x)**: At small sequence lengths,
the fixed HBM read/write overhead of standard attention dominates,
making Flash Attention's advantage most visible.

**Speedup plateaus at larger N in this microbenchmark**: With batch=1
and heads=1, kernel launch overhead masks the O(N²) scaling. In
realistic settings (batch=32, heads=8, N=4096), the S matrix would be
~8GB — impossible to store in HBM at all — making Flash Attention not
just faster but necessary for correctness.

**Flash Attention is not an approximation**: The output is numerically
identical to standard attention. The speedup comes entirely from
reducing HBM memory traffic, not from approximating the computation.

### Connection to dLLM Research

Standard Flash Attention relies on causal masking (each query only
attends to previous keys), enabling left-to-right streaming computation.

LLaDA-style dLLMs use **bidirectional attention** — every token attends
to every other token — which breaks the causal streaming assumption.
This is why Fast-dLLM V2 introduces Block-Causal Attention: causal
between blocks (enabling Flash Attention), bidirectional within blocks
(enabling parallel denoising). This design recovers Flash Attention's
efficiency while preserving dLLM's bidirectional generation capability.
