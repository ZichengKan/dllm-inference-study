# LLM Inference Efficiency: Experiments and Analysis

Systematic study of LLM inference efficiency across two parallel tracks:
general optimization techniques for transformer models, and inference
acceleration for diffusion-based LLMs (dLLMs).

## Key Findings

| Experiment | Result |
|---|---|
| Quantization | INT8 slower than INT4 on small models (dequantization overhead) |
| Knowledge Distillation | Soft labels transfer Teacher response *style*, not just correctness |
| Early Exit | Every layer contributes in 0.5B models; limited exit benefit |
| Flash Attention | 4.74× speedup at N=128 |
| vLLM | 16× throughput at batch=8; serving framework dominates over quantization |
| dLLM Activation Similarity | Sharp similarity drop at Step 6→7 on LLaDA-8B, motivating adaptive cache refresh |
| dLLM Compute Skipping | τ=0.99 → 56.6% MLP FLOPs ↓ while **+6% accuracy** on GSM8K (7B, 100 samples) |
| Layer-level Max Skipping | Complete failure: max aggregation dominated by stable tokens → model collapse |

## Track 1: General LLM Inference Optimization

| Module | Description |
|---|---|
| `quantization_comparison/` | FP16 / INT8 / INT4 via bitsandbytes on Qwen2.5 |
| `lora_finetuning/` | QLoRA fine-tuning with PEFT |
| `flash_attention/` | Speedup benchmark across sequence lengths |
| `vllm_serving/` | PagedAttention + Continuous Batching vs HuggingFace; 4-way comparison |

## Track 2: Diffusion LLM (dLLM) Inference

| Module | Description |
|---|---|
| `activation_analysis/` | dLLM-Cache reproduction on LLaDA-8B-Instruct; Step 6→7 critical transition |
| `compute_skipping/` | Token-level and layer-level skipping on Fast-dLLM V2; 18 settings, GSM8K 7B |