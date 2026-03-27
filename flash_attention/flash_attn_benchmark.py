import torch
import time
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np

# ── 设置 ──────────────────────────────────────────
device = "cuda"
dtype  = torch.float16
d_head = 64       # 每个 attention head 的维度
n_runs = 50       # 每个配置重复多少次取平均
seq_lens = [128, 256, 512, 1024, 2048]

# ── 标准 Attention（手动实现，走 HBM）────────────
def standard_attention(Q, K, V):
    scale = 1.0 / math.sqrt(Q.size(-1))
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (N, N) 写入 HBM
    P = torch.softmax(S, dim=-1)                       # 读写 HBM
    O = torch.matmul(P, V)                             # 读写 HBM
    return O

# ── Flash Attention（调用 PyTorch 内置，走 SRAM）──
def flash_attention(Q, K, V):
    # PyTorch 2.0+ 内置了 Flash Attention
    # scaled_dot_product_attention 会自动选择最优 kernel
    return torch.nn.functional.scaled_dot_product_attention(Q, K, V)

# ── 计时函数 ──────────────────────────────────────
def benchmark(fn, Q, K, V, n_runs):
    # 预热，让 GPU 进入稳定状态
    for _ in range(10):
        _ = fn(Q, K, V)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_runs):
        _ = fn(Q, K, V)
        torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) / n_runs * 1000  # 毫秒

# ── 主实验循环 ────────────────────────────────────
print("开始实验...")
print(f"{'序列长度':>8} {'标准Attn(ms)':>14} {'Flash Attn(ms)':>16} {'加速比':>8}")
print("-" * 52)

std_times   = []
flash_times = []
speedups    = []

for N in seq_lens:
    Q = torch.randn(1, 1, N, d_head, device=device, dtype=dtype)
    K = torch.randn(1, 1, N, d_head, device=device, dtype=dtype)
    V = torch.randn(1, 1, N, d_head, device=device, dtype=dtype)

    t_std   = benchmark(standard_attention, Q, K, V, n_runs)
    t_flash = benchmark(flash_attention,    Q, K, V, n_runs)
    speedup = t_std / t_flash

    std_times.append(t_std)
    flash_times.append(t_flash)
    speedups.append(speedup)

    print(f"{N:>8} {t_std:>14.3f} {t_flash:>16.3f} {speedup:>7.2f}x")

# ── 画图 ──────────────────────────────────────────
sns.set_style("whitegrid")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor("#F8F9FA")

# 颜色
color_std   = "#E63946"
color_flash = "#2D6A4F"
color_speed = "#264653"

# ── 左图：绝对时间对比 ─────────────────────────
x = np.arange(len(seq_lens))
width = 0.35

bars1 = ax1.bar(x - width/2, std_times,   width, label="Standard Attention",
                color=color_std,   alpha=0.85, edgecolor="white", linewidth=0.8)
bars2 = ax1.bar(x + width/2, flash_times, width, label="Flash Attention",
                color=color_flash, alpha=0.85, edgecolor="white", linewidth=0.8)

# 在 bar 顶部标注数值
for bar in bars1:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, h + 0.02,
             f"{h:.2f}", ha="center", va="bottom", fontsize=8, color=color_std)
for bar in bars2:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, h + 0.02,
             f"{h:.2f}", ha="center", va="bottom", fontsize=8, color=color_flash)

ax1.set_xlabel("Sequence Length (N)", fontsize=12)
ax1.set_ylabel("Latency (ms)", fontsize=12)
ax1.set_title("Attention Latency: Standard vs Flash", fontsize=13, fontweight="bold", pad=12)
ax1.set_xticks(x)
ax1.set_xticklabels(seq_lens)
ax1.legend(fontsize=10)
ax1.set_facecolor("#F8F9FA")

# ── 右图：加速比 ──────────────────────────────
ax2.plot(seq_lens, speedups, marker="o", linewidth=2.5,
         markersize=8, color=color_speed, label="Speedup")
ax2.fill_between(seq_lens, 1, speedups, alpha=0.15, color=color_speed)
ax2.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, label="Baseline (1x)")

# 标注每个点的加速比
for n, s in zip(seq_lens, speedups):
    ax2.annotate(f"{s:.2f}x",
                 xy=(n, s), xytext=(0, 12),
                 textcoords="offset points",
                 ha="center", fontsize=9,
                 color=color_speed, fontweight="bold")

ax2.set_xlabel("Sequence Length (N)", fontsize=12)
ax2.set_ylabel("Speedup (×)", fontsize=12)
ax2.set_title("Flash Attention Speedup over Standard", fontsize=13, fontweight="bold", pad=12)
ax2.set_ylim(0, max(speedups) * 1.3)
ax2.legend(fontsize=10)
ax2.set_facecolor("#F8F9FA")

# 标注核心结论
fig.text(0.5, 0.01,
    "Flash Attention avoids O(N²) HBM reads/writes by tiling Q,K,V in SRAM  "
    "— speedup grows with sequence length",
    ha="center", fontsize=10, color="#555555", style="italic")

plt.suptitle("Flash Attention Benchmark  |  RTX 4060 Laptop  |  FP16  |  d_head=64",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("flash_attention_benchmark.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("\n图片已保存：flash_attention_benchmark.png")
