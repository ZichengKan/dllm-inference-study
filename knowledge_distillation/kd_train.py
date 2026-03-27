import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── 配置 ──────────────────────────────────────────
TEACHER_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
STUDENT_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
N_SAMPLES    = 300
N_STEPS      = 100
BATCH_SIZE   = 4
LR           = 1e-5
TEMPERATURE  = 4.0      # soft label 的温度，越高分布越平滑
ALPHA        = 0.5      # hard loss 和 soft loss 的混合比例
MAX_LEN      = 128
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

print(f"使用设备: {DEVICE}")

# ── 加载 tokenizer ─────────────────────────────────
print("加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(STUDENT_NAME)
tokenizer.pad_token = tokenizer.eos_token

# ── 准备数据 ───────────────────────────────────────
print("加载数据集...")
dataset = load_dataset("tatsu-lab/alpaca", split=f"train[:{N_SAMPLES}]")

def format_sample(example):
    if example["input"]:
        text = (f"### Instruction:\n{example['instruction']}\n\n"
                f"### Input:\n{example['input']}\n\n"
                f"### Response:\n{example['output']}")
    else:
        text = (f"### Instruction:\n{example['instruction']}\n\n"
                f"### Response:\n{example['output']}")
    return {"text": text}

dataset = dataset.map(format_sample)

def tokenize(example):
    result = tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length"
    )
    result["labels"] = result["input_ids"].copy()
    return result

tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)
tokenized.set_format("torch")
dataloader = DataLoader(tokenized, batch_size=BATCH_SIZE, shuffle=True)

# ── 加载 Teacher（只推理，不训练）─────────────────
print("加载 Teacher 模型（Qwen2.5-1.5B）...")
teacher = AutoModelForCausalLM.from_pretrained(
    TEACHER_NAME,
    torch_dtype=torch.float16,
    device_map="cuda"
)
teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False
print("Teacher 加载完成")

# ── 训练函数 ───────────────────────────────────────
def train_one_run(mode, n_steps=N_STEPS):
    """
    mode: 'hard'  → 只用 cross entropy loss
          'soft'  → 只用 KL divergence（蒸馏 loss）
          'mixed' → alpha * hard + (1-alpha) * soft
    """
    print(f"\n训练模式: {mode}")
    student = AutoModelForCausalLM.from_pretrained(
        STUDENT_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )
    student.train()
    optimizer = AdamW(student.parameters(), lr=LR)
    losses = []

    data_iter = iter(dataloader)
    for step in range(n_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)

        # Student forward
        student_out = student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        hard_loss = student_out.loss
        if torch.isnan(hard_loss):
            print(f" Step {step}:loss is nan, stopping")
            break

        if mode in ("soft", "mixed"):
            # Teacher forward（不算梯度）
            with torch.no_grad():
                teacher_out = teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

            # 用温度 T 软化 logits
            student_logits = student_out.logits.float() / TEMPERATURE
            teacher_logits = teacher_out.logits.float() / TEMPERATURE

            # KL divergence loss（soft label loss）
            soft_loss = F.kl_div(
                F.log_softmax(student_logits, dim=-1),
                F.softmax(teacher_logits, dim=-1),
                reduction="batchmean"
            ) * (TEMPERATURE ** 2)   # 温度补偿

            if mode == "soft":
                loss = soft_loss
            else:  # mixed
                loss = ALPHA * hard_loss + (1 - ALPHA) * soft_loss
        else:
            loss = hard_loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())

        if step % 20 == 0:
            print(f"  Step {step:3d} | loss = {loss.item():.4f}")

    # 测试回答
    student.eval()
    test_prompt = "### Instruction:\nWhat is the capital of Japan?\n\n### Response:\n"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = student.generate(**inputs, max_new_tokens=40, do_sample=False)
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)

    del student
    torch.cuda.empty_cache()
    return losses, answer

# ── 运行三种模式 ───────────────────────────────────
results = {}
answers = {}

for mode in ["hard", "soft", "mixed"]:
    losses, answer = train_one_run(mode)
    results[mode] = losses
    answers[mode] = answer
    print(f"  [{mode}] 回答: {answer[:80]}")

# ── 可视化 ────────────────────────────────────────
print("\n生成图表...")
sns.set_style("whitegrid")
fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor("#F8F9FA")
gs  = gridspec.GridSpec(2, 1, figure=fig, hspace=0.45)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

colors = {"hard": "#E63946", "soft": "#2D6A4F", "mixed": "#264653"}
labels = {
    "hard":  "Hard Label (SFT only)",
    "soft":  f"Soft Label (KD only, T={TEMPERATURE})",
    "mixed": f"Mixed (α={ALPHA} hard + {1-ALPHA} soft)"
}

# 上图：loss 曲线
steps = list(range(N_STEPS))
for mode in ["hard", "soft", "mixed"]:
    raw = results[mode]
    # 平滑
    smoothed = np.convolve(raw, np.ones(5)/5, mode='same')
    ax1.plot(steps, smoothed, linewidth=2.2,
             color=colors[mode], label=labels[mode])
    ax1.fill_between(steps, smoothed, alpha=0.08, color=colors[mode])

ax1.set_xlabel("Training Step", fontsize=11)
ax1.set_ylabel("Loss", fontsize=11)
ax1.set_title("Knowledge Distillation: Training Loss Comparison",
              fontsize=12, fontweight="bold")
ax1.legend(fontsize=10)
ax1.set_facecolor("#F8F9FA")

# 下图：回答对比（文字表格）
ax2.axis("off")
rows = []
for mode in ["hard", "soft", "mixed"]:
    ans = answers[mode]
    if len(ans) > 70:
        ans = ans[:70] + "..."
    rows.append([labels[mode], ans])

table = ax2.table(
    cellText=rows,
    colLabels=["Training Mode", 'Answer to "What is the capital of Japan?"'],
    cellLoc="left",
    loc="center",
    colWidths=[0.32, 0.68]
)
table.auto_set_font_size(False)
table.set_fontsize(9.5)
table.scale(1, 2.8)

# 表头样式
for j in range(2):
    table[0, j].set_facecolor("#264653")
    table[0, j].set_text_props(color="white", fontweight="bold")

# 行样式
row_colors = ["#F0F7F4", "#FFF5F5", "#F0F4F7"]
for i, color in enumerate(row_colors):
    for j in range(2):
        table[i+1, j].set_facecolor(color)

ax2.set_title("Output Quality Comparison", fontsize=12,
              fontweight="bold", pad=20)

fig.suptitle(
    f"Knowledge Distillation  |  Teacher: Qwen2.5-1.5B  →  Student: Qwen2.5-0.5B  |  "
    f"{N_SAMPLES} samples, {N_STEPS} steps",
    fontsize=11, fontweight="bold", y=1.01
)

plt.savefig("kd_benchmark.png", dpi=150,
            bbox_inches="tight", facecolor=fig.get_facecolor())
print("图片已保存: kd_benchmark.png")
print("\n实验完成！")
