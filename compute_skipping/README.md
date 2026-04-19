\# Compute Skipping on Fast-dLLM V2



Token-level and layer-level compute-skipping strategies implemented on the

Fast-dLLM V2 codebase, evaluated on GSM8K with a 7B model (RTX 4090).



\## Key Findings



| Setting | Accuracy | FLOPs Reduction | Avg Steps |

|---|---|---|---|

| Baseline | 79.0% | 0.0% | 10.7 |

| token\\\_thr\\\_0.99 | \*\*85.0%\*\* | \*\*56.6%\*\* | 24.4 |

| token\\\_thr\\\_0.96 | 84.0% | 78.0% | 24.5 |

| layer\\\_avg\\\_0.99 | 82.0% | 61.5% | 9.9 |

| layer\\\_max\\\_\\\* | 0.0% | 69–80% | 31.0 |



\- Token threshold τ=0.99 \*\*exceeds baseline accuracy by +6%\*\* while reducing MLP FLOPs by 56.6%.

\- Layer-level max aggregation \*\*completely fails\*\*: dominated by stable tokens, causing near-total MLP bypass and model collapse.

\- Layer-level skipping must preserve Attention (for KV cache correctness); only MLP can be skipped.



\## Implementation



\- `skip\_patch.py` — Core logic. Monkey-patches `Fast\_dLLM\_QwenDecoderLayer.forward` without modifying original model files. Supports 5 modes: `none`, `token\_threshold`, `token\_topk`, `layer\_avg`, `layer\_max`.

\- `run\_experiments.py` — Runs all 18 settings and saves results.

\- `collect\_attn.py` — Collects attention weight statistics for Figure C3.

\- `plot\_figures.py` / `plot\_c3.py` — Generate all analysis figures.



\## Setup



```bash

pip install transformers==4.53.1 einops datasets matplotlib

python run\_experiments.py --model 7B --samples 100

```



\## Figures



!\[Acc vs FLOPs](figures/D\_acc\_vs\_flops.png)

!\[Layer Similarity One Sample](figures/A1\_layer\_sim\_one\_sample.png)

!\[Token Similarity One Sample](figures/B1\_token\_sim\_one\_sample.png)

!\[H vs Output Scatter](figures/B4\_h\_vs\_output\_scatter.png)

