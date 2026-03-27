import torch
import os
from transformers import AutoTokenizer, AutoModel

model_name = "GSAI-ML/LLaDA-8B-Instruct"
MASK_TOKEN_ID = 126336

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model.eval()

# 确认模型有多少层
n_layers = len(model.model.transformer.blocks)
print(f"模型共 {n_layers} 层")

# ── 准备输入 ──
prompt = "What is the capital of France?"
messages = [{"role": "user", "content": prompt}]
formatted = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=False
)
input_ids = tokenizer(formatted, return_tensors="pt")["input_ids"]
first_device = next(model.parameters()).device
input_ids = input_ids.to(first_device)
prompt_len = input_ids.shape[1]

response_length = 20
steps = 10

response = torch.full(
    (1, response_length), MASK_TOKEN_ID,
    dtype=torch.long, device=first_device
)
x = torch.cat([input_ids, response], dim=1)
print(f"输入序列 shape: {x.shape}")

# ── Hook 注册函数 ──
def register_all_hooks(model, storage, step):
    """
    给所有层的 K、V、AttnOut、FFNOut 注册 hook
    storage[step][layer_idx][feature_name] = tensor
    """
    handles = []
    storage[step] = {}

    for i, block in enumerate(model.model.transformer.blocks):
        storage[step][i] = {}

        def make_hook(s, layer_idx, feat_name):
            def hook_fn(module, input, output):
                storage[s][layer_idx][feat_name] = output.detach().cpu()
            return hook_fn

        handles.append(block.k_proj.register_forward_hook(
            make_hook(step, i, "K")))
        handles.append(block.v_proj.register_forward_hook(
            make_hook(step, i, "V")))
        handles.append(block.attn_out.register_forward_hook(
            make_hook(step, i, "AttnOut")))
        handles.append(block.ff_out.register_forward_hook(
            make_hook(step, i, "FFNOut")))

    return handles

# ── 推理循环，每步都收集激活值 ──
# all_activations[step][layer_idx][feat_name] = tensor (1, seq_len, d_model)
all_activations = {}

timesteps = torch.linspace(1.0, 1.0 / steps, steps)

print("\n开始推理并收集激活值...")
for i, t in enumerate(timesteps):
    s = t - 1.0 / steps

    # 注册这一步的所有 hook
    handles = register_all_hooks(model, all_activations, i)

    # forward
    with torch.no_grad():
        logits = model(x).logits

    # 移除所有 hook
    for h in handles:
        h.remove()

    # 正常的推理逻辑（remask）
    x0_pred  = logits.argmax(dim=-1)
    probs     = torch.softmax(logits.float(), dim=-1)
    confidence = probs.max(dim=-1).values

    x_resp    = x[:, prompt_len:].clone()
    pred_resp = x0_pred[:, prompt_len:]
    conf_resp = confidence[:, prompt_len:]
    is_masked = (x_resp == MASK_TOKEN_ID)
    n_masked  = is_masked.sum().item()

    x_resp[is_masked] = pred_resp[is_masked]

    if s > 0 and n_masked > 0:
        remask_ratio = s / t
        n_remask = max(1, int(n_masked * remask_ratio))
        conf_for_remask = conf_resp.clone()
        conf_for_remask[~is_masked] = float('inf')
        _, remask_idx = conf_for_remask[0].topk(n_remask, largest=False)
        x_resp[0, remask_idx] = MASK_TOKEN_ID

    x[:, prompt_len:] = x_resp

    n_still_masked = (x[:, prompt_len:] == MASK_TOKEN_ID).sum().item()
    print(f"Step {i+1:2d} | 收集了 {n_layers} 层 × 4种特征 | 剩余MASK: {n_still_masked}")

# ── 验证收集到的数据 ──
print(f"\n=== 数据验证 ===")
print(f"收集了 {len(all_activations)} 个步骤的数据")
print(f"每步有 {len(all_activations[0])} 层的数据")
print(f"每层有 {list(all_activations[0][0].keys())} 四种特征")
print(f"示例 - Step0, Layer0, V的shape: {all_activations[0][0]['V'].shape}")
print(f"示例 - Step1, Layer0, V的shape: {all_activations[1][0]['V'].shape}")

# ── 保存到磁盘 ──
save_path = "activations.pt"
torch.save(all_activations, save_path)
print(f"\n数据已保存到 {save_path}")
print(f"文件大小: {os.path.getsize(save_path) / 1024**2:.1f} MB")
print("\nDay 2 完成 ✓")