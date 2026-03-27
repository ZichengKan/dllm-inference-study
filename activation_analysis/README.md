\## LLaDA-8B Activation Similarity Analysis



Model: LLaDA-8B-Instruct  

Hardware: NVIDIA RTX 4060 Laptop GPU (8GB VRAM, CPU offload)  

Prompt: "What is the capital of France?" (7 tokens, 8 response tokens)



\### Overview



This module analyzes how LLaDA-8B-Instruct's internal representations

evolve across denoising steps, using PyTorch forward hooks to collect

activations from all 32 Transformer layers.



Two complementary analyses are performed:

1\. \*\*Single-step similarity\*\* (Step 0→1): reproduces the core finding

&#x20;  of dLLM-Cache, showing prompt vs response activation stability

2\. \*\*Full-trajectory similarity\*\* (Steps 0→9): tracks activation dynamics

&#x20;  across all adjacent step pairs, revealing temporal structure in the

&#x20;  denoising process



\### Part 1: Single-Step Analysis



!\[Figure 1 Reproduction](figure1\_reproduction.png)



Registered hooks on all 32 layers to collect Key, Value, Attention Output,

and FFN Output activations at Step 0 and Step 1. Computed cosine similarity

per token position along the d\_model=4096 dimension.



\*\*Finding\*\*:

\- Prompt tokens: ρ̄ ≈ 0.93–0.98 across all layers → quasi-static

\- Response tokens: ρ̄ ≈ 0.67–0.77 (Value vectors lowest at 0.67)



This validates dLLM-Cache's core design: prompt activations change very

little between denoising steps and can be safely cached, while response

token activations require selective recomputation.



\### Part 2: Full Denoising Trajectory



!\[Similarity over Steps](similarity\_over\_steps.png)



Extended the analysis to all 9 adjacent step pairs, tracking mean cosine

similarity separately for prompt and response tokens across Attention Output

and FFN Output features.



\*\*Key findings\*\*:



\*\*Prompt tokens consistently more stable than response tokens\*\* — the gap

between green (prompt) and red (response) lines persists throughout

denoising, confirming that differentiated caching is warranted across

the entire generation process, not just the first step.



\*\*Sharp transition at Step 6→7\*\* — both prompt and response similarity

drop significantly at this step pair, indicating a critical state transition

where the model commits to major token decisions. This non-monotonic

behavior cannot be handled by fixed-interval caching, directly motivating

DLLM-CACHE's adaptive refresh strategy.



\*\*Final steps converge to near-1.0 similarity\*\* — once all tokens are

determined (Step 8→9), internal representations stabilize. Caching could

be aggressive in these final steps.



\*\*Deeper layers show lower similarity\*\* — layers 24–31 exhibit more

variation than layers 0–8, consistent with shallow layers extracting

stable syntactic features while deep layers handle token prediction.



\### Method

```python

\# Hook registration

for i, layer in enumerate(model.model.transformer.blocks):

&#x20;   layer.attn\_out.register\_forward\_hook(make\_hook(i, "attn\_out"))

&#x20;   layer.ff\_out.register\_forward\_hook(make\_hook(i, "ffn\_out"))



\# Similarity computation

def cosine\_sim(a, b):  # a, b: (seq\_len, d\_model)

&#x20;   a = a / (a.norm(dim=-1, keepdim=True) + 1e-8)

&#x20;   b = b / (b.norm(dim=-1, keepdim=True) + 1e-8)

&#x20;   return (a \* b).sum(dim=-1)  # (seq\_len,)

```



\### Connection to Papers



| Finding | Relevant Paper |

|---------|---------------|

| Prompt activations quasi-static | dLLM-Cache (motivation for prompt caching) |

| Response activations dynamic | dLLM-Cache (motivation for selective recompute) |

| Non-monotonic similarity at Step 6→7 | DLLM-CACHE (motivation for adaptive refresh) |

| Deep layers more dynamic | Fast-dLLM (layer-wise caching granularity) |

