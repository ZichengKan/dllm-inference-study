\## dLLM Activation Similarity over Denoising Steps



Model: LLaDA-8B-Instruct  

Hardware: NVIDIA RTX 4060 Laptop GPU (8GB VRAM, CPU offload)  

Prompt: "What is the capital of France?" (7 tokens)  

Response length: 8 tokens, 10 denoising steps



\### Experiment Design



This experiment extends the dLLM-Cache Figure 1 reproduction by tracking

activation similarity \*\*across all adjacent step pairs\*\* (0→1, 1→2, ..., 8→9),

rather than a single step transition. Two activation types are measured:

Attention Output and FFN Output, across all 32 Transformer layers.



For each step, forward hooks collect activations from every layer. Cosine

similarity is computed per token position between adjacent steps, then

averaged separately over prompt tokens and response tokens.



!\[Similarity over Steps](similarity\_over\_steps.png)



\### Key Findings



\*\*1. Prompt tokens are more stable than response tokens throughout\*\*  

Prompt token similarity (green) is consistently higher than response token

similarity (red), supporting dLLM-Cache's core assumption that prompt

activations can be cached with longer refresh intervals.



\*\*2. A sharp transition occurs at Step 6→7\*\*  

Both prompt and response similarity drop significantly at this step pair,

suggesting the model undergoes a critical state transition — likely when

the majority of masked tokens receive their first committed prediction.

This non-monotonic behavior challenges the assumption of uniform caching

intervals across all denoising steps.



\*\*3. Final steps converge to near-perfect similarity (→1.0)\*\*  

Steps 8→9 show similarity approaching 1.0 for both token types, indicating

the model's internal representations stabilize once all tokens are determined.



\*\*4. Deeper layers show lower similarity than shallow layers\*\*  

The heatmap reveals that layers 24-31 exhibit lower similarity (more red)

than layers 0-8, consistent with the intuition that deeper layers handle

token prediction decisions while shallower layers extract stable features.



\### Implications for Caching Strategy



These results partially support dLLM-Cache's design (prompt caching) but

also reveal its limitation: a fixed cache refresh interval cannot adapt to

the abrupt transition at Step 6→7. This observation directly motivates

DLLM-CACHE's adaptive refresh mechanism, which dynamically decides when

to invalidate cached activations based on measured activation change.



\### Method



\- Hook targets: `attn\_out` and `ff\_out` in each of 32 LLaDALlamaBlock layers

\- Similarity metric: cosine similarity per token along d\_model=4096 dimension

\- Denoising simulation: sequential unmasking of response tokens across steps

