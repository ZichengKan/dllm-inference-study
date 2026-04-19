import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os

PKL_PATH  = r"C:\Users\Vanillasky\Desktop\sim_log_7B_100samples.pkl"
JSON_PATH = r"C:\Users\Vanillasky\Desktop\results_7B_100samples.json"
OUT_DIR   = r"C:\Users\Vanillasky\Desktop\figures_7B"
os.makedirs(OUT_DIR, exist_ok=True)

print("加载数据...")
with open(PKL_PATH, 'rb') as f:
    data = pickle.load(f)
with open(JSON_PATH, 'r') as f:
    results = json.load(f)

sim_log     = data['sim_log']
flops_log   = data['flops_log']
h_vs_attn   = data['h_vs_attn_log']
h_vs_ffn    = data['h_vs_ffn_log']
ffn_sim_log = data['ffn_sim_log']

NUM_LAYERS = 28
print(f"sim_log 条数: {len(sim_log)}")

def build_step_layer_matrix(log, max_steps=40):
    data_dict = defaultdict(list)
    for entry in log:
        step  = entry['step']
        layer = entry['layer']
        if step > max_steps:
            continue
        sim_val = entry['token_sim'].float().mean().item()
        data_dict[(step, layer)].append(sim_val)
    if not data_dict:
        return np.full((NUM_LAYERS, max_steps), np.nan), np.full((NUM_LAYERS, max_steps), np.nan), max_steps
    actual_max = min(max(k[0] for k in data_dict.keys()), max_steps)
    matrix_mean = np.full((NUM_LAYERS, actual_max), np.nan)
    matrix_var  = np.full((NUM_LAYERS, actual_max), np.nan)
    for (step, layer), vals in data_dict.items():
        if 1 <= step <= actual_max and layer < NUM_LAYERS:
            matrix_mean[layer, step-1] = np.mean(vals)
            matrix_var[layer, step-1]  = np.var(vals)
    return matrix_mean, matrix_var, actual_max

def build_token_layer_matrix(log, max_tokens=64):
    data_dict = defaultdict(list)
    for entry in log:
        layer = entry['layer']
        token_sim = entry['token_sim'].float().squeeze(0)
        L = min(token_sim.shape[0], max_tokens)
        for pos in range(L):
            data_dict[(layer, pos)].append(token_sim[pos].item())
    matrix = np.full((NUM_LAYERS, max_tokens), np.nan)
    for (layer, pos), vals in data_dict.items():
        if layer < NUM_LAYERS and pos < max_tokens:
            matrix[layer, pos] = np.mean(vals)
    valid_cols = ~np.all(np.isnan(matrix), axis=0)
    return matrix[:, valid_cols]

def normalize_per_step(matrix):
    result = matrix.copy()
    for col in range(matrix.shape[1]):
        col_data = matrix[:, col]
        valid = col_data[~np.isnan(col_data)]
        if len(valid) == 0:
            continue
        vmin, vmax = valid.min(), valid.max()
        if vmax > vmin:
            result[:, col] = (col_data - vmin) / (vmax - vmin)
        else:
            result[:, col] = 0.5
    return result

def plot_heatmap(matrix, title, xlabel, ylabel, filename, cmap='viridis', vmin=None, vmax=None, figsize=(12,6)):
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax, origin='upper', interpolation='nearest')
    plt.colorbar(im, ax=ax)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"✅ 已保存：{filename}")

print("\n画图A：Layer skipping 动机图...")
mean_mat, var_mat, max_step = build_step_layer_matrix(sim_log)

seen = set()
single_log = []
for entry in sim_log:
    key = (entry['step'], entry['layer'])
    if key not in seen:
        seen.add(key)
        single_log.append(entry)
single_mean, _, _ = build_step_layer_matrix(single_log)

plot_heatmap(single_mean[:, :max_step], "Layer Similarity - One Sample\n(x: denoising step, y: layer)", "Denoising Step", "Layer", "A1_layer_sim_one_sample.png", vmin=0.5, vmax=1.0)
mean_norm = normalize_per_step(mean_mat[:, :max_step])
plot_heatmap(mean_norm, "Layer Similarity - Mean over Samples (normalized per step)", "Denoising Step", "Layer", "A2_layer_sim_mean.png", vmin=0, vmax=1)
var_norm = normalize_per_step(var_mat[:, :max_step])
plot_heatmap(var_norm, "Layer Similarity - Variance over Samples (normalized per step)", "Denoising Step", "Layer", "A3_layer_sim_var.png", vmin=0, vmax=1, cmap='plasma')

print("\n画图B：Token skipping 动机图...")
token_mat = build_token_layer_matrix(single_log)
plot_heatmap(token_mat, "Token Similarity - One Sample\n(x: token position, y: layer, avg over steps)", "Token Position", "Layer", "B1_token_sim_one_sample.png", vmin=0.5, vmax=1.0)
plot_heatmap(mean_norm, "Token Similarity - Mean over Samples (normalized per step)", "Denoising Step", "Layer", "B2_token_sim_mean.png", vmin=0, vmax=1)
plot_heatmap(var_norm, "Token Similarity - Variance over Samples (normalized per step)", "Denoising Step", "Layer", "B3_token_sim_var.png", vmin=0, vmax=1, cmap='plasma')

print("画图B4：散点图...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
if h_vs_attn:
    h_sims  = [x['h_sim']   for x in h_vs_attn]
    a_sims  = [x['attn_sim'] for x in h_vs_attn]
    idx = np.random.choice(len(h_sims), min(5000, len(h_sims)), replace=False)
    h_arr = np.array(h_sims)[idx]
    a_arr = np.array(a_sims)[idx]
    rho = np.corrcoef(h_arr, a_arr)[0,1]
    axes[0].scatter(h_arr, a_arr, alpha=0.1, s=5, color='steelblue')
    axes[0].plot([0,1],[0,1],'r--',linewidth=1)
    axes[0].set_title(f"H vs AttnOut\nρ = {rho:.3f}", fontsize=12)
    axes[0].set_xlabel("Similarity of H", fontsize=11)
    axes[0].set_ylabel("Similarity of ATTN OUT", fontsize=11)
    axes[0].set_xlim(0,1); axes[0].set_ylim(0,1)
if h_vs_ffn:
    h_sims2 = [x['h_sim']  for x in h_vs_ffn]
    f_sims  = [x['ffn_sim'] for x in h_vs_ffn]
    idx2 = np.random.choice(len(h_sims2), min(5000, len(h_sims2)), replace=False)
    h_arr2 = np.array(h_sims2)[idx2]
    f_arr  = np.array(f_sims)[idx2]
    rho2 = np.corrcoef(h_arr2, f_arr)[0,1]
    axes[1].scatter(h_arr2, f_arr, alpha=0.1, s=5, color='darkorange')
    axes[1].plot([0,1],[0,1],'r--',linewidth=1)
    axes[1].set_title(f"H vs FFN Out\nρ = {rho2:.3f}", fontsize=12)
    axes[1].set_xlabel("Similarity of H", fontsize=11)
    axes[1].set_ylabel("Similarity of FFN OUT", fontsize=11)
    axes[1].set_xlim(0,1); axes[1].set_ylim(0,1)
plt.suptitle("Similarity of Hidden State vs Output", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "B4_h_vs_output_scatter.png"), dpi=150)
plt.close()
print("✅ 已保存：B4_h_vs_output_scatter.png")

print("\n画图C：Column skipping 动机图...")
if ffn_sim_log:
    ffn_dict = defaultdict(list)
    for entry in ffn_sim_log:
        if entry['step'] >= 1:
            ffn_dict[(entry['step'], entry['layer'])].append(entry['sim'])
    actual_max = min(max(k[0] for k in ffn_dict.keys()), 40)
    ffn_matrix = np.full((NUM_LAYERS, actual_max), np.nan)
    for (step, layer), vals in ffn_dict.items():
        if step <= actual_max and layer < NUM_LAYERS:
            ffn_matrix[layer, step-1] = np.mean(vals)
    ffn_norm = normalize_per_step(ffn_matrix)
    plot_heatmap(ffn_norm, "FFN Temp Similarity (normalized per step)", "Denoising Step", "Layer", "C1_ffn_temp_sim.png", vmin=0, vmax=1)

if h_vs_attn:
    attn_dict = defaultdict(list)
    for entry in h_vs_attn:
        attn_dict[(entry['step'], entry['layer'])].append(entry['attn_sim'])
    actual_max = min(max(k[0] for k in attn_dict.keys()), 40)
    attn_matrix = np.full((NUM_LAYERS, actual_max), np.nan)
    for (step, layer), vals in attn_dict.items():
        if step <= actual_max and layer < NUM_LAYERS:
            attn_matrix[layer, step-1] = np.mean(vals)
    attn_norm = normalize_per_step(attn_matrix)
    plot_heatmap(attn_norm, "Attention Output Similarity (normalized per step)", "Denoising Step", "Layer", "C2_attn_sim.png", vmin=0, vmax=1)

print("\n画图D：Acc vs FLOPs reduction...")
colors = {
    'baseline':        ('black',     '*', 150, 'Baseline'),
    'token_threshold': ('steelblue', 'o',  80, 'Token Threshold'),
    'token_topk':      ('green',     's',  80, 'Token Top-k%'),
    'layer_avg':       ('orange',    '^',  80, 'Layer Avg'),
    'layer_max':       ('red',       'D',  80, 'Layer Max'),
}
fig, ax = plt.subplots(figsize=(10, 7))
plotted = set()
for r in results:
    name = r['name']
    acc  = r['accuracy'] * 100
    flops_r = r['flops_reduction'] * 100
    if name == 'baseline':            key = 'baseline'
    elif 'token_thr'  in name:        key = 'token_threshold'
    elif 'token_topk' in name:        key = 'token_topk'
    elif 'layer_avg'  in name:        key = 'layer_avg'
    else:                             key = 'layer_max'
    color, marker, size, label = colors[key]
    lbl = label if label not in plotted else ''
    ax.scatter(flops_r, acc, c=color, marker=marker, s=size, label=lbl, zorder=5, edgecolors='black', linewidths=0.5)
    plotted.add(label)
ax.set_xlabel("FLOPs Reduced Compared to Baseline (%)", fontsize=12)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title("Accuracy vs FLOPs Reduction\n(Fast-dLLM v2 7B, GSM8K 100 samples)", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(-5, 100)
ax.set_ylim(-5, 100)
ax.annotate('Better →', xy=(0.75, 0.85), xycoords='axes fraction', fontsize=11, color='gray')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "D_acc_vs_flops.png"), dpi=150)
plt.close()
print("✅ 已保存：D_acc_vs_flops.png")
print(f"\n所有图已保存到：{OUT_DIR}")