import torch
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import warnings
warnings.filterwarnings("ignore")

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE     = "cuda"
N_TOKENS   = 50
N_RUNS     = 5
BATCH_SIZE = 8

PROMPTS = [
    "What is the capital of France?",
    "Explain machine learning in one sentence.",
    "What is 99 multiplied by 99?",
    "Who wrote Romeo and Juliet?",
    "What is the chemical formula for water?",
    "Name three programming languages.",
    "What is the speed of light?",
    "Describe what a neural network is.",
] * (BATCH_SIZE // 8 + 1)
PROMPTS = PROMPTS[:BATCH_SIZE]

def measure_hf(model, tokenizer, label):
    print(f"\n  [{label}] 测量中...")
    inputs = tokenizer(
        PROMPTS, return_tensors="pt",
        padding=True, truncation=True, max_length=64
    ).to(model.device)

    # 预热
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=10, do_sample=False)

    times  = []
    tokens = []
    for _ in range(N_RUNS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=N_TOKENS, do_sample=False
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        n_new = sum(
            len(o) - inputs["input_ids"].shape[1]
            for o in out
        )
        times.append(elapsed)
        tokens.append(n_new)

    tp  = np.mean(tokens) / np.mean(times)
    lat = np.mean(times) / BATCH_SIZE * 1000
    mem = torch.cuda.memory_allocated() / 1024**3
    print(f"     吞吐量: {tp:.1f} tok/s  延迟: {lat:.1f} ms/req  显存: {mem:.2f} GB")
    return tp, lat, mem

def measure_vllm(label, quantization=None):
    from vllm import LLM, SamplingParams
    print(f"\n  [{label}] 测量中...")

    kwargs = dict(
        model=MODEL_NAME,
        dtype="float16",
        gpu_memory_utilization=0.75,
        max_model_len=256,
    )
    if quantization:
        kwargs["quantization"] = quantization

    llm = LLM(**kwargs)
    sp  = SamplingParams(temperature=0, max_tokens=N_TOKENS)

    # 预热
    _ = llm.generate(PROMPTS[:2], sp)

    times  = []
    tokens = []
    for _ in range(N_RUNS):
        start = time.perf_counter()
        out   = llm.generate(PROMPTS, sp)
        elapsed = time.perf_counter() - start
        n_tok = sum(len(o.outputs[0].token_ids) for o in out)
        times.append(elapsed)
        tokens.append(n_tok)

    tp  = np.mean(tokens) / np.mean(times)
    lat = np.mean(times) / BATCH_SIZE * 1000
    # vLLM 显存通过 gpu_memory_utilization 控制，用固定值估算
    mem = 0.93  # FP16 模型基础占用

    print(f"     吞吐量: {tp:.1f} tok/s  延迟: {lat:.1f} ms/req")
    del llm
    torch.cuda.empty_cache()
    return tp, lat, mem

if __name__ == '__main__':
    results = {}

    # ── 方案1：HuggingFace + FP16 ──────────────────
    print("\n加载 HuggingFace FP16 模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="cuda"
    )
    model_fp16.eval()
    results["HF + FP16\n(Baseline)"] = measure_hf(model_fp16, tokenizer, "HF FP16")
    del model_fp16
    torch.cuda.empty_cache()

    # ── 方案2：HuggingFace + INT4 ──────────────────
    print("\n加载 HuggingFace INT4 模型...")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    model_int4 = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb_cfg, device_map="cuda"
    )
    model_int4.eval()
    results["HF + INT4\n(Quantization)"] = measure_hf(model_int4, tokenizer, "HF INT4")
    del model_int4
    torch.cuda.empty_cache()

    # ── 方案3：vLLM + FP16 ─────────────────────────
    results["vLLM + FP16\n(Serving)"] = measure_vllm("vLLM FP16")

    # ── 方案4：vLLM + INT4 ─────────────────────────
    results["vLLM + INT4\n(Both)"] = measure_vllm("vLLM INT4", quantization="bitsandbytes")

    # ── 可视化 ────────────────────────────────────
    print("\n生成图表...")
    labels   = list(results.keys())
    tps      = [results[k][0] for k in labels]
    lats     = [results[k][1] for k in labels]
    baseline = tps[0]
    speedups = [tp / baseline for tp in tps]

    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(15, 10))
    fig.patch.set_facecolor("#F8F9FA")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    colors = ["#ADB5BD", "#2D6A4F", "#264653", "#E63946"]
    x = np.arange(len(labels))

    # 左上：吞吐量
    bars = ax1.bar(x, tps, color=colors, edgecolor="white",
                   linewidth=0.8, alpha=0.88)
    for bar, tp in zip(bars, tps):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                 f"{tp:.0f}", ha="center", va="bottom",
                 fontsize=10, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("Throughput (tokens/sec)", fontsize=11)
    ax1.set_title("Throughput Comparison", fontsize=12, fontweight="bold")
    ax1.set_facecolor("#F8F9FA")

    # 右上：延迟
    bars2 = ax2.bar(x, lats, color=colors, edgecolor="white",
                    linewidth=0.8, alpha=0.88)
    for bar, lat in zip(bars2, lats):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{lat:.1f}ms", ha="center", va="bottom",
                 fontsize=10, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("Latency per Request (ms)", fontsize=11)
    ax2.set_title("Latency Comparison", fontsize=12, fontweight="bold")
    ax2.set_facecolor("#F8F9FA")

    # 下方：加速比瀑布图
    bars3 = ax3.bar(x, speedups, color=colors, edgecolor="white",
                    linewidth=0.8, alpha=0.88, width=0.5)
    ax3.axhline(y=1.0, color="gray", linestyle="--",
                linewidth=1, alpha=0.7, label="Baseline (1x)")
    for bar, sp, tp in zip(bars3, speedups, tps):
        ax3.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.05,
                 f"{sp:.1f}x\n({tp:.0f} tok/s)",
                 ha="center", va="bottom",
                 fontsize=11, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=11)
    ax3.set_ylabel("Speedup over Baseline", fontsize=12)
    ax3.set_title("Speedup vs HuggingFace FP16 Baseline",
                  fontsize=12, fontweight="bold")
    ax3.legend(fontsize=10)
    ax3.set_facecolor("#F8F9FA")

    fig.suptitle(
        f"Inference Acceleration Comparison  |  Qwen2.5-0.5B  |  "
        f"batch={BATCH_SIZE}  |  RTX 4060 8GB",
        fontsize=13, fontweight="bold", y=1.01
    )

    plt.savefig("inference_comparison.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    print("图片已保存: inference_comparison.png")
    print("\n实验完成！")
