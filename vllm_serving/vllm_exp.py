import time
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

prompts = [
    "What is the capital of France?",
    "Explain what machine learning is in one sentence.",
    "What is 99 multiplied by 99?",
    "Name three programming languages.",
    "What is the speed of light?",
]

def measure_hf(model, tokenizer, prompts, label):
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    start = time.time()
    total_tokens = 0
    for p in prompts:
        inputs = tokenizer(p, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        total_tokens += len(new_tokens)
    elapsed = time.time() - start
    print(f"总耗时:    {elapsed:.2f} 秒")
    print(f"总token数: {total_tokens}")
    print(f"吞吐量:    {total_tokens / elapsed:.1f} tokens/sec")

def measure_vllm(llm, prompts, label):
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    sampling_params = SamplingParams(temperature=0, max_tokens=50)
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - start
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"总耗时:    {elapsed:.2f} 秒")
    print(f"总token数: {total_tokens}")
    print(f"吞吐量:    {total_tokens / elapsed:.1f} tokens/sec")
    print("\n回答示例：")
    for output in outputs:
        print(f"  Q: {output.prompt[:50]}")
        print(f"  A: {output.outputs[0].text[:80]}\n")

if __name__ == '__main__':
    # ── HuggingFace 普通推理 ──
    print("加载 HuggingFace 模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_hf = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    model_hf.eval()
    measure_hf(model_hf, tokenizer, prompts, "HuggingFace（逐条推理）")
    del model_hf
    torch.cuda.empty_cache()

    # ── vLLM 推理 ──
    print("\n\n加载 vLLM 模型...")
    llm = LLM(
        model=model_name,
        dtype="float16",
        gpu_memory_utilization=0.7,
        max_model_len=512
    )
    measure_vllm(llm, prompts, "vLLM（PagedAttention + Continuous Batching）")
