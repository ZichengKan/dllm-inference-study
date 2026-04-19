"""
collect_attn.py
收集 attention weights 分布数据，用于画图 C3。
只跑 baseline（不跳过任何计算），统计 attention weights 落在各区间的次数。

运行方式：
    python collect_attn.py
结果保存到：attn_hist.pkl，包含 counts 和 bins 两个字段。
"""

import torch, types, sys, pickle
import numpy as np
sys.path.insert(0, '.')
from transformers import AutoTokenizer, AutoModelForCausalLM
import skip_patch, generation_functions
from datasets import load_dataset

MODEL_PATH = 'Efficient-Large-Model/Fast_dLLM_v2_7B'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_SAMPLES = 25
OUT_PATH = 'attn_hist.pkl'

# 40个对数刻度区间：0.0001 ~ 1.0
bins = []
for start, end in [(0.0001, 0.001), (0.001, 0.01), (0.01, 0.1), (0.1, 1.0)]:
    bins.extend(np.logspace(np.log10(start), np.log10(end), 11)[:-1].tolist())
bins.append(1.0)
bins = np.array(bins)
counts = np.zeros(40, dtype=np.int64)

print('加载模型...')
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16
).to(DEVICE).eval()
skip_patch.apply(model)
skip_cache = skip_patch.init_skip_cache(model, skip_mode='none', record_attn=True)
model.mdm_sample = types.MethodType(
    generation_functions.Fast_dLLM_QwenForCausalLM.batch_sample, model
)
print('模型加载完成')

ds = load_dataset('openai/gsm8k', 'main', split='test')
for i in range(N_SAMPLES):
    sample = ds[i]
    question = sample['question'] + ' Please reason step by step, and put your final answer within \\boxed{}.'
    messages = [{'role': 'user', 'content': question}]
    formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(formatted, return_tensors='pt')
    input_ids = inputs['input_ids'].to(DEVICE)
    seq_len = torch.tensor([input_ids.shape[1]], device=DEVICE)

    with torch.no_grad():
        model.mdm_sample(
            input_ids, tokenizer=tokenizer,
            block_size=32, small_block_size=8,
            max_new_tokens=512, mask_id=151665,
            min_len=input_ids.shape[1], seq_len=seq_len,
            use_block_cache=False, threshold=0.9,
            top_p=0.95, temperature=0.0,
        )

    # 每道题处理完立刻统计，清空 log 节省内存
    current = model.model.layers[0]._skip_cache_ref[0]
    log = current.get('attn_weight_log', [])
    for aw in log:
        vals = aw.flatten().numpy()
        mask = (vals >= 0.0001) & (vals <= 1.0)
        c, _ = np.histogram(vals[mask], bins=bins)
        counts[:len(c)] += c
    current['attn_weight_log'] = []

    if (i + 1) % 5 == 0:
        print(f'[{i+1}/{N_SAMPLES}] 完成，总统计值：{counts.sum()}')

with open(OUT_PATH, 'wb') as f:
    pickle.dump({'counts': counts, 'bins': bins}, f)
print(f'已保存到：{OUT_PATH}，总统计值：{counts.sum()}')