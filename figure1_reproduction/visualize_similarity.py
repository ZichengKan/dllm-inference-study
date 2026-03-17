import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F

# ── 1. 读取数据 ──
print("读取激活值数据...")
all_activations = torch.load("activations.pt")

n_steps  = len(all_activations)          # 10
n_layers = len(all_activations[0])       # 32
seq_len  = all_activations[0][0]["V"].shape[1]  # 40
prompt_len = 20                          # prompt 占前20个位置

print(f"步数: {n_steps}, 层数: {n_layers}, 序列长度: {seq_len}")

# ── 2. 计算相邻步之间的 cosine similarity ──
def compute_similarity_matrix(all_activations, feat_name, step_a, step_b):
    """
    计算 step_a 和 step_b 之间，所有层所有位置的 cosine similarity
    返回 shape: (n_layers, seq_len)
    """
    sim_matrix = torch.zeros(n_layers, seq_len)

    for layer_idx in range(n_layers):
        # 取出两步的同一层同一特征
        # shape: (1, seq_len, d_model) → squeeze → (seq_len, d_model)
        tensor_a = all_activations[step_a][layer_idx][feat_name].squeeze(0)
        tensor_b = all_activations[step_b][layer_idx][feat_name].squeeze(0)

        # 转成 float32 再算（float16 精度不够）
        tensor_a = tensor_a.float()
        tensor_b = tensor_b.float()

        # 对每个 token 位置计算 cosine similarity
        # F.cosine_similarity 在 dim=-1 上计算，即对4096维向量算
        # 输入 shape: (seq_len, d_model)
        # 输出 shape: (seq_len,)
        sim = F.cosine_similarity(tensor_a, tensor_b, dim=-1)
        sim_matrix[layer_idx] = sim

    return sim_matrix  # (n_layers, seq_len)

# 计算相邻步 0→1 之间的 similarity（论文用的是相邻步）
print("计算 cosine similarity...")
sim_K      = compute_similarity_matrix(all_activations, "K",      0, 1)
sim_V      = compute_similarity_matrix(all_activations, "V",      0, 1)
sim_AttnOut= compute_similarity_matrix(all_activations, "AttnOut",0, 1)
sim_FFNOut = compute_similarity_matrix(all_activations, "FFNOut", 0, 1)

print(f"K similarity 矩阵 shape: {sim_K.shape}")
print(f"K similarity 均值: {sim_K.mean():.4f}")
print(f"V similarity 均值: {sim_V.mean():.4f}")

# ── 3. 画热力图 ──
print("\n开始画图...")

fig, axes = plt.subplots(1, 4, figsize=(24, 8))
titles = ["Key", "Value", "Attention Output", "FFN Output"]
matrices = [sim_K, sim_V, sim_AttnOut, sim_FFNOut]

for ax, matrix, title in zip(axes, matrices, titles):
    # 把 tensor 转成 numpy（seaborn 需要 numpy 格式）
    data = matrix.numpy()

    # 画热力图
    # vmin/vmax 固定颜色范围为 0~1
    # cmap="YlOrRd_r" 是颜色方案：similarity高=浅黄，similarity低=深红
    sns.heatmap(
        data,
        ax=ax,
        vmin=0.0, vmax=1.0,
        cmap="YlOrRd_r",
        xticklabels=False,   # x轴太密，不显示每个刻度
        yticklabels=8        # y轴每8层显示一个刻度
    )

    # 画一条竖线，分隔 prompt 和 response
    ax.axvline(x=prompt_len, color='blue', linewidth=2, linestyle='--')

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Token Position\n(left: prompt | right: response)", fontsize=10)
    ax.set_ylabel("Layer", fontsize=10)

    # 在图上标注 prompt 和 response 区域
    ax.text(prompt_len/2, -1.5, "Prompt",
            ha='center', fontsize=9, color='blue')
    ax.text(prompt_len + (seq_len-prompt_len)/2, -1.5, "Response",
            ha='center', fontsize=9, color='red')

plt.suptitle("Cosine Similarity of Activations Between Adjacent Denoising Steps\n"
             "(Step 0 → Step 1)",
             fontsize=14, y=1.02)
plt.tight_layout()

save_path = "figure1_reproduction.png"
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"图已保存到 {save_path}")

# ── 4. 打印数值统计，验证是否符合论文的结论 ──
print("\n=== 数值统计 ===")
for name, matrix in zip(titles, matrices):
    prompt_sim   = matrix[:, :prompt_len].mean().item()
    response_sim = matrix[:, prompt_len:].mean().item()
    print(f"{name:20s} | prompt均值: {prompt_sim:.4f} | response均值: {response_sim:.4f}")

print("\nDay 3 完成 ✓")