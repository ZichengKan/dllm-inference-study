"""
run_experiments.py
批量跑18种设置，收集准确率和FLOPs数据。
本地用1.5B模型验证，服务器换成7B模型。

运行方式：
    python run_experiments.py --model 1.5B --samples 5   # 本地验证（5道题）
    python run_experiments.py --model 7B --samples 100  # 服务器正式实验
"""

import argparse
import os
import json
import pickle
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='1.5B', choices=['1.5B', '7B'])
parser.add_argument('--samples', type=int, default=5)
parser.add_argument('--output_dir', type=str, default='results')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

MODEL_PATH = {
    '1.5B': 'Efficient-Large-Model/Fast_dLLM_v2_1.5B',
    '7B':   'Efficient-Large-Model/Fast_dLLM_v2_7B',
}[args.model]

# 18种实验设置
SETTINGS = [
    # baseline
    {'name': 'baseline',           'skip_mode': 'none',              'threshold': None, 'topk': None},
    # token-level threshold
    {'name': 'token_thr_0.995',    'skip_mode': 'token_threshold',   'threshold': 0.995, 'topk': None},
    {'name': 'token_thr_0.99',     'skip_mode': 'token_threshold',   'threshold': 0.99,  'topk': None},
    {'name': 'token_thr_0.98',     'skip_mode': 'token_threshold',   'threshold': 0.98,  'topk': None},
    {'name': 'token_thr_0.97',     'skip_mode': 'token_threshold',   'threshold': 0.97,  'topk': None},
    {'name': 'token_thr_0.96',     'skip_mode': 'token_threshold',   'threshold': 0.96,  'topk': None},
    # token-level top-k%
    {'name': 'token_topk_25',      'skip_mode': 'token_topk',        'threshold': None, 'topk': 0.25},
    {'name': 'token_topk_50',      'skip_mode': 'token_topk',        'threshold': None, 'topk': 0.50},
    # layer-level avg
    {'name': 'layer_avg_0.999',    'skip_mode': 'layer_avg',         'threshold': 0.999, 'topk': None},
    {'name': 'layer_avg_0.995',    'skip_mode': 'layer_avg',         'threshold': 0.995, 'topk': None},
    {'name': 'layer_avg_0.99',     'skip_mode': 'layer_avg',         'threshold': 0.99,  'topk': None},
    {'name': 'layer_avg_0.98',     'skip_mode': 'layer_avg',         'threshold': 0.98,  'topk': None},
    {'name': 'layer_avg_0.97',     'skip_mode': 'layer_avg',         'threshold': 0.97,  'topk': None},
    # layer-level max
    {'name': 'layer_max_0.999',    'skip_mode': 'layer_max',         'threshold': 0.999, 'topk': None},
    {'name': 'layer_max_0.995',    'skip_mode': 'layer_max',         'threshold': 0.995, 'topk': None},
    {'name': 'layer_max_0.99',     'skip_mode': 'layer_max',         'threshold': 0.99,  'topk': None},
    {'name': 'layer_max_0.98',     'skip_mode': 'layer_max',         'threshold': 0.98,  'topk': None},
    {'name': 'layer_max_0.97',     'skip_mode': 'layer_max',         'threshold': 0.97,  'topk': None},
]


def load_gsm8k(n_samples):
    """加载 GSM8K 数据集的前 n_samples 道题"""
    from datasets import load_dataset
    ds = load_dataset('openai/gsm8k', 'main', split='test')
    return [ds[i] for i in range(n_samples)]


def extract_answer(text):
    """从模型输出里提取 boxed 里的答案"""
    import re
    matches = re.findall(r'\\boxed\{([^}]+)\}', text)
    if matches:
        ans = matches[-1].strip().replace(',', '')
        try:
            return float(ans)
        except:
            return ans
    # 备用：找最后一个数字
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        try:
            return float(numbers[-1])
        except:
            return None
    return None


def extract_gt_answer(solution_text):
    """从 GSM8K 的标准答案里提取数字"""
    import re
    # GSM8K 答案格式：'#### 18'
    match = re.search(r'####\s*(-?\d+)', solution_text)
    if match:
        try:
            return float(match.group(1).replace(',', ''))
        except:
            return None
    return None


def run_one_setting(setting, samples, model, tokenizer, device):
    """跑一种设置，返回准确率、FLOPs reduction、平均步数"""
    import types
    import skip_patch
    import generation_functions

    skip_mode = setting['skip_mode']
    threshold = setting['threshold'] if setting['threshold'] is not None else 0.995
    topk = setting['topk'] if setting['topk'] is not None else 0.25

    # 初始化 skip_cache（同时清空上一轮的数据）
    skip_cache = skip_patch.init_skip_cache(
        model,
        skip_mode=skip_mode,
        threshold=threshold,
        topk=topk,
        record_attn=False,
        record_ffn=False,
    )

    correct = 0
    total_steps = 0

    for i, sample in enumerate(samples):
        question = sample['question']
        gt = extract_gt_answer(sample['answer'])

        question_with_prompt = question.replace(
            "Answer:", "Please reason step by step, and put your final answer within \\boxed{}."
        )
        if "Please reason" not in question_with_prompt:
            question_with_prompt += " Please reason step by step, and put your final answer within \\boxed{}."

        messages = [{"role": "user", "content": question_with_prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = tokenizer(formatted, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        seq_len = torch.tensor([input_ids.shape[1]], device=device)

        with torch.no_grad():
            generated_ids = model.mdm_sample(
                input_ids, tokenizer=tokenizer,
                block_size=32, small_block_size=8,
                max_new_tokens=512, mask_id=151665,
                min_len=input_ids.shape[1],
                seq_len=seq_len, use_block_cache=False,
                threshold=0.9, top_p=0.95, temperature=0.0,
            )

        answer_text = tokenizer.decode(
            generated_ids[0][input_ids.shape[1]:], skip_special_tokens=True
        )
        pred = extract_answer(answer_text)

        if pred is not None and gt is not None and abs(pred - gt) < 1e-3:
            correct += 1

        # 统计这道题的 denoising steps（从 flops_log 里统计）
        current_cache = model.model.layers[0]._skip_cache_ref[0]
        steps_this_sample = current_cache['flops_log'][-1]['step'] if current_cache['flops_log'] else 0
        total_steps += steps_this_sample

        print(f"  [{i+1}/{len(samples)}] gt={gt}, pred={pred}, {'✓' if pred==gt else '✗'}")

    # 计算 FLOPs reduction
    current_cache = model.model.layers[0]._skip_cache_ref[0]
    total_tokens = sum(x['total_tokens'] for x in current_cache['flops_log'])
    skipped_tokens = sum(x['skipped_tokens'] for x in current_cache['flops_log'])
    flops_reduction = skipped_tokens / total_tokens if total_tokens > 0 else 0.0

    accuracy = correct / len(samples)
    avg_steps = total_steps / len(samples)

    return {
        'name': setting['name'],
        'accuracy': accuracy,
        'flops_reduction': flops_reduction,
        'avg_steps': avg_steps,
        'correct': correct,
        'total': len(samples),
    }


def main():
    import types
    import skip_patch
    import generation_functions
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备：{device}，模型：{MODEL_PATH}，样本数：{args.samples}")

    print("加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(device).eval()

    # apply patch（只需一次）
    skip_patch.apply(model)

    model.mdm_sample = types.MethodType(
        generation_functions.Fast_dLLM_QwenForCausalLM.batch_sample, model
    )

    print("加载 GSM8K 数据集...")
    samples = load_gsm8k(args.samples)
    print(f"加载了 {len(samples)} 道题\n")

    all_results = []

    for i, setting in enumerate(SETTINGS):
        print(f"[{i+1}/{len(SETTINGS)}] 运行设置：{setting['name']}")
        result = run_one_setting(setting, samples, model, tokenizer, device)
        all_results.append(result)
        print(f"  准确率: {result['accuracy']*100:.1f}%  "
              f"FLOPs reduction: {result['flops_reduction']*100:.1f}%  "
              f"平均步数: {result['avg_steps']:.1f}\n")

    # 保存结果
    output_path = os.path.join(args.output_dir, f'results_{args.model}_{args.samples}samples.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到：{output_path}")

    # 打印汇总表
    print("\n" + "="*70)
    print(f"{'设置':<25} {'准确率':>8} {'FLOPs减少':>12} {'平均步数':>10}")
    print("-"*70)
    for r in all_results:
        print(f"{r['name']:<25} {r['accuracy']*100:>7.1f}%  "
              f"{r['flops_reduction']*100:>10.1f}%  {r['avg_steps']:>9.1f}")
    print("="*70)

    # 同时保存完整的 sim_log（供画图用）
    current_cache = model.model.layers[0]._skip_cache_ref[0]
    log_path = os.path.join(args.output_dir, f'sim_log_{args.model}_{args.samples}samples.pkl')
    with open(log_path, 'wb') as f:
        pickle.dump({
            'sim_log': current_cache.get('sim_log', []),
            'flops_log': current_cache.get('flops_log', []),
            'h_vs_attn_log': current_cache.get('h_vs_attn_log', []),
            'h_vs_ffn_log': current_cache.get('h_vs_ffn_log', []),
            'ffn_sim_log': current_cache.get('ffn_sim_log', []),
        }, f)
    print(f"日志数据已保存到：{log_path}")


if __name__ == '__main__':
    main()