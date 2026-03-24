import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

# ── 1. 加载 tokenizer 和模型（用 QLoRA：INT4 量化 + LoRA）──
print("加载模型...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Qwen 没有 pad token，用 eos 代替

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="cuda"
)

# ── 2. 打印原始模型参数量 ──
total_params = sum(p.numel() for p in model.parameters())
print(f"原始模型总参数量: {total_params:,}")

# ── 3. 挂上 LoRA ──
# target_modules：只对 q_proj 和 v_proj 挂 LoRA
# r=8：低秩矩阵的秩，越大表达能力越强但参数越多
# lora_alpha=16：缩放系数，通常设为 2*r
# lora_dropout=0.05：防止过拟合
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none"
)

model = get_peft_model(model, lora_config)

# ── 4. 打印 LoRA 参数量，对比原始 ──
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"LoRA 可训练参数量: {trainable:,}")
print(f"总参数量:          {total:,}")
print(f"可训练比例:        {100 * trainable / total:.4f}%")

# ── 5. 准备数据：用 Alpaca 数据集前 200 条 ──
print("\n加载数据集...")
dataset = load_dataset("tatsu-lab/alpaca", split="train[:200]")

def format_sample(example):
    # 把每条数据格式化成 instruction → output 的对话格式
    if example["input"]:
        text = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
    else:
        text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    return {"text": text}

dataset = dataset.map(format_sample)

def tokenize(example):
    result = tokenizer(
        example["text"],
        truncation=True,
        max_length=256,
        padding="max_length"
    )
    result["labels"] = result["input_ids"].copy()
    return result

tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)
tokenized.set_format("torch")

# ── 6. 训练前：记录模型对一道题的回答 ──
test_prompt = "### Instruction:\nWhat is the capital of Japan?\n\n### Response:\n"
inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")

print("\n训练前的回答：")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=30, do_sample=False)
new_tokens = out[0][inputs["input_ids"].shape[1]:]
print(tokenizer.decode(new_tokens, skip_special_tokens=True))

# ── 7. 手动训练循环（100步）──
from torch.utils.data import DataLoader
from torch.optim import AdamW

dataloader = DataLoader(tokenized, batch_size=4, shuffle=True)
optimizer  = AdamW(model.parameters(), lr=2e-4)

model.train()
print("\n开始训练（100步）...")

losses = []
for step, batch in enumerate(dataloader):
    if step >= 100:
        break

    input_ids = batch["input_ids"].to("cuda")
    labels    = batch["labels"].to("cuda")
    attention_mask = batch["attention_mask"].to("cuda")

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    losses.append(loss.item())

    if step % 20 == 0:
        print(f"  Step {step:3d} | loss = {loss.item():.4f}")

print(f"\n训练完成，初始 loss: {losses[0]:.4f}，最终 loss: {losses[-1]:.4f}")

# ── 8. 训练后：记录同一道题的回答，对比变化 ──
model.eval()
print("\n训练后的回答：")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=30, do_sample=False)
new_tokens = out[0][inputs["input_ids"].shape[1]:]
print(tokenizer.decode(new_tokens, skip_special_tokens=True))
