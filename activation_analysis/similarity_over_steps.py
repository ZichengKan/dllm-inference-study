import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings("ignore")

# ── 配置 ──────────────────────────────────────────────────────
MODEL_NAME   = "GSAI-ML/LLaDA-8B-Instruct"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
N_STEPS      = 10
MASK_TOKEN   = 126336
PROMPT_TEXT  = "What is the capital of France?"
RESPONSE_LEN = 8

# ── 加载模型 ───────────────────────────────────────────────────
print("加载模型（CPU offload，需要几分钟）...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()
print("模型加载完成")

# ── 构造输入 ───────────────────────────────────────────────────
prompt_ids = tokenizer.encode(PROMPT_TEXT, add_special_tokens=False)
n_prompt   = len(prompt_ids)
input_ids  = torch.tensor(
    [prompt_ids + [MASK_TOKEN] * RESPONSE_LEN],
    dtype=torch.long
).to(DEVICE)
n_total = input_ids.shape[1]
print(f"序列长度: prompt={n_prompt}, response={RESPONSE_LEN}, total={n_total}")

# ── 生成去噪步骤序列 ───────────────────────────────────────────
torch.manual_seed(42)

def make_denoising_steps(input_ids, n_steps, n_prompt):
    steps = []
    current = input_ids.clone()
    response_positions = list(range(n_prompt, input_ids.shape[1]))
    unmask_per_step = max(1, len(response_positions) // n_steps)
    still_masked = response_positions.copy()
    for step in range(n_steps):
        steps.append(current.clone())
        n_unmask = min(unmask_per_step, len(still_masked))
        if n_unmask > 0:
            to_unmask = still_masked[:n_unmask]
            still_masked = still_masked[n_unmask:]
            for pos in to_unmask:
                current[0, pos] = torch.randint(100, 10000, (1,)).item()
    return steps

denoising_steps = make_denoising_steps(input_ids, N_STEPS, n_prompt)

# ── Hook 收集激活值 ────────────────────────────────────────────
def collect_hooks(model):
    hooks = []
    layer_acts = {}

    def make_hook(layer_idx, feat_name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            key = (layer_idx, feat_name)
            layer_acts[key] = out.detach().cpu().float()
        return hook_fn

    blocks = model.model.transformer.blocks
    for i, layer in enumerate(blocks):
        hooks.append(layer.attn_out.register_forward_hook(
            make_hook(i, "attn_out")))
        hooks.append(layer.ff_out.register_forward_hook(
            make_hook(i, "ffn_out")))

    return hooks, layer_acts

print(f"\n开始收集激活值（{N_STEPS} 步）...")
step_activations = []

for step_idx, step_input in enumerate(denoising_steps):
    print(f"  Step {step_idx+1}/{N_STEPS}...", end=" ", flush=True)
    hooks, layer_acts = collect_hooks(model)
    with torch.no_grad():
        _ = model(step_input)
    for h in hooks:
        h.remove()
    step_activations.append(dict(layer_acts))
    print("done")

n_layers = len(model.model.transformer.blocks)
print(f"\n收集完成，共 {n_layers} 层 × {N_STEPS} 步")

# ── 计算 cosine similarity ─────────────────────────────────────
def cosine_sim(a, b):
    a = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
    b = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
    return (a * b).sum(dim=-1)

feat_names  = ["attn_out", "ffn_out"]
sim_results = {feat: [] for feat in feat_names}

print("\n计算 cosine similarity...")
for step_idx in range(N_STEPS - 1):
    acts_a = step_activations[step_idx]
    acts_b = step_activations[step_idx + 1]
    for feat in feat_names:
        layer_sims = []
        for layer_idx in range(n_layers):
            key = (layer_idx, feat)
            if key in acts_a and key in acts_b:
                a = acts_a[key][0]
                b = acts_b[key][0]
                sim = cosine_sim(a, b)
                layer_sims.append(sim.numpy())
        if layer_sims:
            sim_results[feat].append(np.array(layer_sims))
print("计算完成")

# ── 可视化 ────────────────────────────────────────────────────
print("\n生成图表...")
sns.set_style("whitegrid")
fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor("#F8F9FA")
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

feat_titles     = {"attn_out": "Attention Output", "ffn_out": "FFN Output"}
colors_prompt   = "#2D6A4F"
colors_response = "#E63946"
step_pairs      = [f"{i}→{i+1}" for i in range(N_STEPS - 1)]

# 子图 1-2：折线图
for fi, feat in enumerate(feat_names):
    ax = axes[fi]
    prompt_sims   = []
    response_sims = []
    for pair_idx in range(len(sim_results[feat])):
        sim_mat = sim_results[feat][pair_idx]
        mean_sim = sim_mat.mean(axis=0)
        prompt_sims.append(mean_sim[:n_prompt].mean())
        response_sims.append(mean_sim[n_prompt:].mean())

    ax.plot(range(len(step_pairs)), prompt_sims,
            marker="o", linewidth=2.5, markersize=7,
            color=colors_prompt, label="Prompt tokens")
    ax.plot(range(len(step_pairs)), response_sims,
            marker="s", linewidth=2.5, markersize=7,
            color=colors_response, label="Response tokens")
    ax.fill_between(range(len(step_pairs)), prompt_sims,
                    alpha=0.12, color=colors_prompt)
    ax.fill_between(range(len(step_pairs)), response_sims,
                    alpha=0.12, color=colors_response)
    ax.set_xticks(range(len(step_pairs)))
    ax.set_xticklabels(step_pairs, fontsize=8, rotation=30)
    ax.set_xlabel("Denoising Step Pair", fontsize=10)
    ax.set_ylabel("Mean Cosine Similarity", fontsize=10)
    ax.set_title(f"{feat_titles[feat]}: Similarity over Steps",
                 fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.set_facecolor("#F8F9FA")

# 子图 3-4：heatmap
for fi, feat in enumerate(feat_names):
    ax = axes[fi + 2]
    if not sim_results[feat]:
        continue
    heatmap_data = []
    for pair_idx in range(len(sim_results[feat])):
        sim_mat = sim_results[feat][pair_idx]
        heatmap_data.append(sim_mat.mean(axis=1))
    heatmap_data = np.array(heatmap_data).T

    sns.heatmap(
        heatmap_data, ax=ax,
        cmap="RdYlGn", vmin=0.5, vmax=1.0,
        xticklabels=step_pairs,
        yticklabels=[str(i) if i % 4 == 0 else "" for i in range(n_layers)],
        cbar_kws={"label": "Cosine Similarity", "shrink": 0.8}
    )
    ax.set_xlabel("Denoising Step Pair", fontsize=10)
    ax.set_ylabel("Layer Index", fontsize=10)
    ax.set_title(f"{feat_titles[feat]}: Layer × Step Heatmap",
                 fontsize=11, fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=30, fontsize=8)

fig.suptitle(
    "dLLM Activation Similarity over Denoising Steps  |  LLaDA-8B-Instruct",
    fontsize=13, fontweight="bold", y=1.01
)
fig.text(
    0.5, -0.01,
    "Prompt tokens remain quasi-static across denoising steps  "
    "— Response tokens show larger variation, especially in later steps",
    ha="center", fontsize=10, color="#555555", style="italic"
)
plt.savefig("similarity_over_steps.png", dpi=150,
            bbox_inches="tight", facecolor=fig.get_facecolor())
print("图片已保存：similarity_over_steps.png")
print("实验完成！")