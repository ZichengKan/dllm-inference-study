import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── 配置 ──────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE     = "cuda"

QUESTIONS = [
    "What is the capital of France?",
    "What is 12 multiplied by 12?",
    "Explain what a neural network is in one sentence.",
    "Who wrote Romeo and Juliet?",
    "What is the chemical formula for water?",
]

# ── 加载模型 ───────────────────────────────────────
print("加载模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    output_hidden_states=True,
)
model.eval()
n_layers = model.config.num_hidden_layers
print(f"模型加载完成，共 {n_layers} 层")

# ── 分析函数 ───────────────────────────────────────
def analyze_question(question):
    prompt = f"### Instruction:\n{question}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    hidden_states = outputs.hidden_states
    # hidden_states[0] = embedding, [1..n_layers] = transformer layers

    # 1. 层间 cosine similarity（相邻层 hidden state 的相似度）
    #    越高说明这一层变化越小，越接近"稳定"
    layer_similarities = []
    for i in range(1, n_layers):
        hs_prev = hidden_states[i][0, -1, :].float()    # 上一层最后token
        hs_curr = hidden_states[i+1][0, -1, :].float()  # 当前层最后token
        sim = F.cosine_similarity(
            hs_prev.unsqueeze(0),
            hs_curr.unsqueeze(0)
        ).item()
        layer_similarities.append(sim)

    # 2. 层间 L2 变化量（越小说明这层改变越少）
    layer_changes = []
    for i in range(1, n_layers):
        hs_prev = hidden_states[i][0, -1, :].float()
        hs_curr = hidden_states[i+1][0, -1, :].float()
        change = (hs_curr - hs_prev).norm().item()
        layer_changes.append(change)

    # 3. 最后几层的真实置信度（用 lm_head 预测，只在最后8层用才有意义）
    last_n = 8
    layer_confidences = []
    layer_predictions = []
    for i in range(n_layers - last_n + 1, n_layers + 1):
        hs = hidden_states[i][0, -1, :].unsqueeze(0)
        logits = model.lm_head(hs)
        probs  = F.softmax(logits.float(), dim=-1)
        confidence   = probs.max().item()
        predicted_id = probs.argmax().item()
        predicted_tok = tokenizer.decode([predicted_id]).strip()
        layer_confidences.append(confidence)
        layer_predictions.append(predicted_tok)

    return layer_similarities, layer_changes, layer_confidences, layer_predictions

# ── 对所有问题运行 ─────────────────────────────────
print("\n分析各层...")
all_sims    = []
all_changes = []
all_confs   = []
all_preds   = []

for q in QUESTIONS:
    print(f"  {q[:45]}...")
    sims, changes, confs, preds = analyze_question(q)
    all_sims.append(sims)
    all_changes.append(changes)
    all_confs.append(confs)
    all_preds.append(preds)

# ── 打印最后8层的预测 ──────────────────────────────
last_n = 8
last_layers = list(range(n_layers - last_n + 1, n_layers + 1))
print(f"\n最后 {last_n} 层的预测 token：")
print(f"{'问题':<38}", end="")
for l in last_layers:
    print(f"L{l:<4}", end=" ")
print()
print("-" * 100)
for i, q in enumerate(QUESTIONS):
    print(f"{q[:37]:<38}", end="")
    for j, pred in enumerate(all_preds[i]):
        print(f"{pred[:4]:<5}", end=" ")
    print()

# ── 可视化 ────────────────────────────────────────
print("\n生成图表...")
sns.set_style("whitegrid")
fig = plt.figure(figsize=(16, 14))
fig.patch.set_facecolor("#F8F9FA")
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.35)

ax1 = fig.add_subplot(gs[0, :])   # 层间相似度曲线（横跨）
ax2 = fig.add_subplot(gs[1, 0])   # 层间变化量
ax3 = fig.add_subplot(gs[1, 1])   # 最后几层置信度
ax4 = fig.add_subplot(gs[2, :])   # heatmap（横跨）

colors = plt.cm.Set2(np.linspace(0, 1, len(QUESTIONS)))
sim_layers    = list(range(2, n_layers + 1))    # 层间 similarity 的 x 轴
change_layers = list(range(2, n_layers + 1))

# ── 上图：层间 cosine similarity ──────────────────
for i, (q, sims) in enumerate(zip(QUESTIONS, all_sims)):
    label = q[:40] + "..." if len(q) > 40 else q
    ax1.plot(sim_layers, sims, linewidth=1.8, marker="o", markersize=3,
             color=colors[i], label=label, alpha=0.85)

ax1.axhline(y=0.99, color="red", linestyle="--",
            linewidth=1.2, label="Stability threshold (0.99)", alpha=0.7)
ax1.set_xlabel("Layer Index", fontsize=11)
ax1.set_ylabel("Cosine Similarity (layer i vs i+1)", fontsize=11)
ax1.set_title("Inter-Layer Hidden State Similarity\n"
              "(High similarity = small change = potential early exit point)",
              fontsize=12, fontweight="bold")
ax1.legend(fontsize=8, loc="lower right")
ax1.set_xlim(2, n_layers)
ax1.set_ylim(0.8, 1.01)
ax1.set_facecolor("#F8F9FA")

# ── 中左：层间 L2 变化量 ──────────────────────────
mean_change = np.mean(all_changes, axis=0)
for i, (q, changes) in enumerate(zip(QUESTIONS, all_changes)):
    ax2.plot(change_layers, changes, linewidth=1.2, alpha=0.4,
             color=colors[i])
ax2.plot(change_layers, mean_change, linewidth=2.5,
         color="#264653", label="Mean change", zorder=5)
ax2.set_xlabel("Layer Index", fontsize=11)
ax2.set_ylabel("L2 Norm of Hidden State Change", fontsize=11)
ax2.set_title("Hidden State Change per Layer\n"
              "(Smaller = layer has less impact)",
              fontsize=12, fontweight="bold")
ax2.legend(fontsize=9)
ax2.set_facecolor("#F8F9FA")

# ── 中右：最后几层的真实置信度 ───────────────────
for i, (q, confs) in enumerate(zip(QUESTIONS, all_confs)):
    label = q[:30] + "..." if len(q) > 30 else q
    ax3.plot(last_layers, confs, linewidth=2, marker="s", markersize=5,
             color=colors[i], label=label, alpha=0.85)
ax3.axhline(y=0.9, color="red", linestyle="--",
            linewidth=1.2, alpha=0.7, label="Threshold (0.9)")
ax3.set_xlabel("Layer Index", fontsize=11)
ax3.set_ylabel("Confidence (max softmax prob)", fontsize=11)
ax3.set_title(f"Prediction Confidence (Last {last_n} Layers)\n"
              "(lm_head valid only near final layer)",
              fontsize=12, fontweight="bold")
ax3.legend(fontsize=7)
ax3.set_facecolor("#F8F9FA")

# ── 下图：相似度 heatmap ───────────────────────────
sim_matrix = np.array(all_sims)
short_qs   = [q[:28] + "..." if len(q) > 28 else q for q in QUESTIONS]

sns.heatmap(
    sim_matrix,
    ax=ax4,
    cmap="RdYlGn",
    vmin=0.85, vmax=1.0,
    xticklabels=[str(i) if i % 4 == 0 else "" for i in sim_layers],
    yticklabels=short_qs,
    cbar_kws={"label": "Cosine Similarity", "shrink": 0.6}
)
ax4.set_xlabel("Layer Transition (i → i+1)", fontsize=11)
ax4.set_title("Inter-Layer Similarity Heatmap: Question × Layer Transition",
              fontsize=12, fontweight="bold")
plt.setp(ax4.get_yticklabels(), fontsize=9)

fig.suptitle(
    f"Early Exit Analysis  |  Qwen2.5-0.5B  |  {n_layers} Layers",
    fontsize=13, fontweight="bold", y=1.01
)
fig.text(
    0.5, -0.01,
    "Inter-layer similarity reveals where representations stabilize — "
    "high similarity layers are candidates for early exit",
    ha="center", fontsize=10, color="#555555", style="italic"
)

plt.savefig("early_exit_analysis.png", dpi=150,
            bbox_inches="tight", facecolor=fig.get_facecolor())
print("图片已保存: early_exit_analysis.png")
print("\n实验完成！")
