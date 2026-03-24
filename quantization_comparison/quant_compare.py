import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

prompts = [
    "What is the capital of France?",
    "Explain what a transformer is in one sentence.",
    "What is 123 multiplied by 456?",
]

def measure(model, tokenizer, label):
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")

    # 显存占用
    mem = torch.cuda.memory_allocated() / 1024**3
    print(f"显存占用: {mem:.2f} GB")

    # 生成速度
    inputs = tokenizer(prompts[0], return_tensors="pt").to(model.device)
    n_tokens = 50

    start = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=n_tokens,
            do_sample=False
        )
    elapsed = time.time() - start
    speed = n_tokens / elapsed
    print(f"生成速度: {speed:.1f} tokens/sec")

    # 回答质量（3道题）
    print("回答质量：")
    for p in prompts:
        inp = tokenizer(p, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=30, do_sample=False)
        # 只解码新生成的部分
        new_tokens = out[0][inp['input_ids'].shape[1]:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
        print(f"  Q: {p}")
        print(f"  A: {answer}\n")

    torch.cuda.empty_cache()

# ── FP16 ──
print("加载 FP16 模型...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_fp16 = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="cuda"
)
model_fp16.eval()
measure(model_fp16, tokenizer, "FP16")
del model_fp16
torch.cuda.empty_cache()

# ── INT8 ──
print("\n加载 INT8 模型...")
bnb_int8 = BitsAndBytesConfig(load_in_8bit=True)
model_int8 = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_int8,
    device_map="cuda"
)
model_int8.eval()
measure(model_int8, tokenizer, "INT8")
del model_int8
torch.cuda.empty_cache()

# ── INT4 ──
print("\n加载 INT4 模型...")
bnb_int4 = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
model_int4 = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_int4,
    device_map="cuda"
)
model_int4.eval()
measure(model_int4, tokenizer, "INT4")
