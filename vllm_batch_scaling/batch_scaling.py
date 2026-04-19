import time
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
from vllm import LLM, SamplingParams
import warnings
warnings.filterwarnings("ignore")

MODEL_NAME  = "Qwen/Qwen2.5-0.5B-Instruct"
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
MAX_TOKENS  = 50
PROMPT      = "Explain what machine learning is in one sentence."

def measure_batch(llm, sampling_params, batch_size, n_repeats=3):
    prompts = [PROMPT] * batch_size
    times   = []
    tokens  = []
    for _ in range(n_repeats):
        start   = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        elapsed = time.perf_counter() - start
        total_tok = sum(len(o.outputs[0].token_ids) for o in outputs)
        times.append(elapsed)
        tokens.append(total_tok)
    mean_time    = np.mean(times)
    total_tokens = np.mean(tokens)
    throughput   = total_tokens / mean_time
    latency      = mean_time / batch_size * 1000
    return throughput, latency, mean_time

if __name__ == '__main__':
    print("加载 vLLM...")
    llm = LLM(
        model=MODEL_NAME,
        dtype="float16",
        gpu_memory_utilization=0.8,
        max_model_len=512,
    )
    sampling_params = SamplingParams(temperature=0, max_tokens=MAX_TOKENS)
    print("加载完成\n")

    print(f"{'Batch':>6} {'Throughput(tok/s)':>18} {'Latency(ms/req)':>16} {'Time(s)':>8}")
    print("-" * 55)

    throughputs = []
    latencies   = []

    for bs in BATCH_SIZES:
        try:
            tp, lat, t = measure_batch(llm, sampling_params, bs)
            throughputs.append(tp)
            latencies.append(lat)
            print(f"{bs:>6} {tp:>18.1f} {lat:>16.1f} {t:>8.2f}")
        except Exception as e:
            print(f"{bs:>6} Error: {e}")
            throughputs.append(None)
            latencies.append(None)

    valid = [(bs, tp, lat) for bs, tp, lat in
             zip(BATCH_SIZES, throughputs, latencies) if tp is not None]
    valid_bs  = [x[0] for x in valid]
    valid_tp  = [x[1] for x in valid]
    valid_lat = [x[2] for x in valid]

    print("\n生成图表...")
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor("#F8F9FA")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    color_tp  = "#2D6A4F"
    color_lat = "#E63946"

    # 上图：吞吐量
    ax1.plot(valid_bs, valid_tp, marker="o", linewidth=2.5,
             markersize=9, color=color_tp, label="Throughput")
    ax1.fill_between(valid_bs, valid_tp, alpha=0.12, color=color_tp)
    for bs, tp in zip(valid_bs, valid_tp):
        ax1.annotate(f"{tp:.0f}",
                     xy=(bs, tp), xytext=(0, 12),
                     textcoords="offset points",
                     ha="center", fontsize=9,
                     color=color_tp, fontweight="bold")
    ax1.set_xlabel("Batch Size", fontsize=12)
    ax1.set_ylabel("Throughput (tokens/sec)", fontsize=12)
    ax1.set_title("vLLM Throughput vs Batch Size\n"
                  "(Larger batch → better GPU utilization → higher throughput)",
                  fontsize=12, fontweight="bold")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(valid_bs)
    ax1.set_xticklabels([str(b) for b in valid_bs])
    ax1.legend(fontsize=10)
    ax1.set_facecolor("#F8F9FA")

    # 左下：延迟
    ax2.plot(valid_bs, valid_lat, marker="s", linewidth=2.5,
             markersize=8, color=color_lat, label="Latency per request")
    ax2.fill_between(valid_bs, valid_lat, alpha=0.12, color=color_lat)
    for bs, lat in zip(valid_bs, valid_lat):
        ax2.annotate(f"{lat:.0f}ms",
                     xy=(bs, lat), xytext=(0, 10),
                     textcoords="offset points",
                     ha="center", fontsize=8, color=color_lat)
    ax2.set_xlabel("Batch Size", fontsize=11)
    ax2.set_ylabel("Latency per Request (ms)", fontsize=11)
    ax2.set_title("Latency vs Batch Size\n"
                  "(Larger batch → each request waits longer)",
                  fontsize=11, fontweight="bold")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(valid_bs)
    ax2.set_xticklabels([str(b) for b in valid_bs])
    ax2.legend(fontsize=9)
    ax2.set_facecolor("#F8F9FA")

    # 右下：归一化对比
    norm_tp  = np.array(valid_tp)  / max(valid_tp)
    norm_lat = np.array(valid_lat) / max(valid_lat)
    ax3.plot(valid_bs, norm_tp,  marker="o", linewidth=2,
             color=color_tp,  label="Throughput (normalized)")
    ax3.plot(valid_bs, norm_lat, marker="s", linewidth=2,
             color=color_lat, label="Latency (normalized)")
    ax3.set_xlabel("Batch Size", fontsize=11)
    ax3.set_ylabel("Normalized Value", fontsize=11)
    ax3.set_title("Throughput vs Latency Trade-off",
                  fontsize=11, fontweight="bold")
    ax3.set_xscale("log", base=2)
    ax3.set_xticks(valid_bs)
    ax3.set_xticklabels([str(b) for b in valid_bs])
    ax3.legend(fontsize=9)
    ax3.set_facecolor("#F8F9FA")

    fig.suptitle(
        f"vLLM Batch Scaling  |  Qwen2.5-0.5B  |  max_tokens={MAX_TOKENS}  |  RTX 4060 8GB",
        fontsize=12, fontweight="bold", y=1.01
    )
    fig.text(
        0.5, -0.01,
        "Throughput scales with batch size due to better GPU utilization — "
        "latency grows as each request waits for the full batch to complete",
        ha="center", fontsize=10, color="#555555", style="italic"
    )

    plt.savefig("batch_scaling.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    print("图片已保存: batch_scaling.png")
    print("\n实验完成！")
