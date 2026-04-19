## Early Exit Analysis: Layer-wise Representation Stability

Model: Qwen2.5-0.5B-Instruct (24 layers)  
Hardware: NVIDIA RTX 4060 Laptop GPU (8GB VRAM)  
Questions: 5 diverse prompts (factual, arithmetic, conceptual)

### What is Early Exit?

Standard Transformer inference runs all N layers for every input.
Early Exit proposes adding exit points at intermediate layers: if the
model's representation is already "stable" at layer k < N, skip the
remaining layers and output early, saving compute.

The key challenge is defining "stable enough to exit." Two metrics
are used here:

1. **Inter-layer cosine similarity**: high similarity between layer i
   and i+1 means that layer i+1 changed the representation very little
   → candidate exit point

2. **L2 norm of hidden state change**: small L2 change means the layer
   had little impact → candidate exit point

### Why not use lm_head confidence directly?

A naive approach applies the final lm_head to intermediate hidden states
to get prediction confidence. This fails because lm_head was trained
specifically on the final layer's output distribution — applying it to
intermediate layers produces meaningless logits with near-zero confidence
for all layers except the last 1-2. The confidence subplot confirms this:
only layers 23-24 produce meaningful predictions.

![Early Exit Analysis](early_exit_analysis.png)

### Key Findings

**Every layer makes a meaningful contribution**: Inter-layer cosine
similarity stays between 0.83-0.97 throughout, never reaching the 0.99
stability threshold. No layer is clearly redundant for this model size.

**Final two layers are critical decision layers**: L2 change spikes
dramatically at layers 23-24 (~300 vs ~15 for earlier layers), and
prediction confidence only becomes meaningful at layer 23+. These layers
cannot be skipped without significant quality loss.

**Early Exit is more valuable for large models**: In 70B+ models, many
middle layers show near-1.0 inter-layer similarity — genuine redundancy
that Early Exit can exploit. For 0.5B models, the network is already
compact and every layer carries information.

### Connection to Efficient LLM Research

Early Exit connects to several active research directions:
- **Adaptive computation**: allocate more compute to harder inputs
- **Speculative decoding**: use a shallow exit as the draft model,
  verify with the full model (similar to Teacher-Student in distillation)
- **dLLM inference**: in diffusion LLMs, each denoising step runs the
  full model — Early Exit could reduce per-step cost for easy steps

### Method
```python
# Collect all hidden states in one forward pass
outputs = model(**inputs, output_hidden_states=True)
hidden_states = outputs.hidden_states  # tuple of (n_layers+1) tensors

# Inter-layer similarity
sim = F.cosine_similarity(hidden_states[i][:, -1, :],
                          hidden_states[i+1][:, -1, :])

# L2 change
change = (hidden_states[i+1] - hidden_states[i]).norm()
```
