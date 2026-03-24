## LoRA Fine-tuning: Parameter-Efficient SFT on Qwen2.5-0.5B

Model: Qwen2.5-0.5B-Instruct  
Hardware: NVIDIA RTX 4060 Laptop GPU (8GB VRAM)  
Method: QLoRA (INT4 quantization + LoRA)  
Data: tatsu-lab/alpaca (200 samples, 100 training steps)

### Parameter Comparison

| | Parameters | % of Total |
|--|-----------|------------|
| Full model | 315,119,488 | 100% |
| LoRA trainable (r=8, q/v proj) | 540,672 | 0.17% |

Only 0.17% of parameters are trained, yet the model learns the
instruction-following format effectively.

### Training Results

| Step | Loss |
|------|------|
| 0    | 14.48 |
| 20   | 4.25 |
| 40   | 0.81 |
| 100  | 0.71 |

Loss drops from 14.48 to 0.71 in 100 steps, showing rapid convergence
on the instruction-following format.

### Output Comparison

**Before training:**
```
Q: What is the capital of Japan?
A: The capital of Japan is Tokyo. It is located on the island of Honshu,
   which is part of the main island group of Japan. Tokyo...
```
(Model continues generating without stopping — no instruction-following format)

**After training:**
```
Q: What is the capital of Japan?
A: The capital of Japan is Tokyo.
```
(Concise, instruction-following style learned from Alpaca)

### Key Concepts

**Why LoRA works**: Weight update matrix ΔW is low-rank during fine-tuning.
Instead of updating the full W (d×d parameters), LoRA approximates
ΔW ≈ A×B where A is (d×r) and B is (r×d), r<<d.
With r=8, d=896: full ΔW = 802,816 params → LoRA = 14,336 params (56x compression).

**QLoRA = Quantization + LoRA**: Load model in INT4 (saves VRAM),
attach LoRA adapters (only 0.17% params trained). Enables fine-tuning
large models on consumer GPUs.

**Note on labels**: In this experiment, labels=input_ids (full sequence loss).
Proper SFT should set prompt tokens to -100 in labels so loss is computed
only on response tokens. This matters for longer training runs.

### LoRA Configuration
- r=8, lora_alpha=16
- target_modules: q_proj, v_proj
- lora_dropout=0.05
- optimizer: AdamW, lr=2e-4
- batch_size=4
