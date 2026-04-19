"""
skip_patch.py
核心文件：给 Fast_dLLM_QwenDecoderLayer 加入 skipping 能力。
"""

import torch
import torch.nn.functional as F


def _new_decoder_layer_forward(
    self,
    hidden_states,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    use_cache=False,
    cache_position=None,
    position_embeddings=None,
    update_past_key_values=False,
    use_block_cache=False,
    block_past_key_values=None,
    replace_position=None,
    **kwargs,
):
    layer_idx = self.self_attn.layer_idx
    skip_cache = getattr(self, '_skip_cache_ref', [None])[0]

    # ── ① 第一个 LayerNorm ──────────────────────────────────────────
    residual = hidden_states
    normed = self.input_layernorm(hidden_states)   # [B, L, D]

    # ── skipping 判断 ────────────────────────────────────────────────
    skip_mask = None        # token-level：哪些 token 跳过
    skip_mlp_only = False   # layer-level：只跳过 MLP（Attention 必须正常跑以更新 KV cache）

    if skip_cache is not None:
        step = skip_cache.get('step', 0)
        skip_mode = skip_cache.get('skip_mode', 'none')
        threshold = skip_cache.get('threshold', 0.995)
        topk = skip_cache.get('topk', 0.25)
        prev_normed_dict = skip_cache.get('prev_normed', {})

        if step > 0 and layer_idx in prev_normed_dict:
            prev_normed = prev_normed_dict[layer_idx]

            # 形状不一致时不做 skipping
            if normed.shape == prev_normed.shape:
                token_sim = F.cosine_similarity(normed, prev_normed, dim=-1)  # [B, L]

                # 记录 sim_log（图A、B用）
                if 'sim_log' in skip_cache:
                    skip_cache['sim_log'].append({
                        'step': step,
                        'layer': layer_idx,
                        'token_sim': token_sim.detach().cpu(),
                    })

                # Layer-level：整层相似度高时，跳过 MLP（Attention 仍然运行）
                if skip_mode == 'layer_avg':
                    if token_sim.mean().item() > threshold:
                        skip_mlp_only = True
                elif skip_mode == 'layer_max':
                    if token_sim.max().item() > threshold:
                        skip_mlp_only = True

                # Token-level
                elif skip_mode == 'token_threshold':
                    skip_mask = (token_sim > threshold)
                elif skip_mode == 'token_topk':
                    B, L = token_sim.shape
                    num_compute = max(1, int(L * topk))
                    _, compute_indices = token_sim.topk(num_compute, dim=-1, largest=False)
                    skip_mask = torch.ones(B, L, dtype=torch.bool, device=normed.device)
                    skip_mask.scatter_(-1, compute_indices, False)

        # FLOPs 统计日志
        if 'flops_log' in skip_cache:
            B, L, D = normed.shape
            if skip_mlp_only:
                skipped = B * L   # MLP 全跳
            elif skip_mask is not None:
                skipped = int(skip_mask.sum().item())
            else:
                skipped = 0
            skip_cache['flops_log'].append({
                'step': skip_cache.get('step', 0),
                'layer': layer_idx,
                'total_tokens': B * L,
                'skipped_tokens': skipped,
            })

    # ── ② Attention（始终运行，保证 KV cache 正确更新）──────────────
    attn_out = self.self_attn(
        hidden_states=normed,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        update_past_key_values=update_past_key_values,
        use_block_cache=use_block_cache,
        block_past_key_values=block_past_key_values,
        replace_position=replace_position,
        **kwargs,
    )

    # 记录 h_vs_attn（图B4用）
    if skip_cache is not None and 'h_vs_attn_log' in skip_cache:
        step = skip_cache.get('step', 0)
        if step > 0 and layer_idx in skip_cache.get('prev_attn_out', {}):
            prev_attn = skip_cache['prev_attn_out'][layer_idx]
            if attn_out.shape == prev_attn.shape and skip_cache.get('sim_log'):
                last = skip_cache['sim_log'][-1]
                if last['layer'] == layer_idx and last['step'] == step:
                    h_sim = last['token_sim'].mean().item()
                    a_sim = F.cosine_similarity(
                        attn_out.reshape(attn_out.shape[0], -1),
                        prev_attn.reshape(prev_attn.shape[0], -1), dim=-1
                    ).mean().item()
                    skip_cache['h_vs_attn_log'].append({
                        'h_sim': h_sim, 'attn_sim': a_sim,
                        'layer': layer_idx, 'step': step,
                    })

    # Token-level：替换跳过 token 的 attn_out
    if skip_mask is not None and skip_cache is not None:
        prev_attn = skip_cache.get('prev_attn_out', {}).get(layer_idx)
        if prev_attn is not None and attn_out.shape == prev_attn.shape:
            attn_out = torch.where(skip_mask.unsqueeze(-1), prev_attn, attn_out)

    # ── ③ 残差 ──────────────────────────────────────────────────────
    hidden_states = residual + attn_out

    # ── ④ 第二个 LayerNorm ──────────────────────────────────────────
    residual = hidden_states
    normed2 = self.post_attention_layernorm(hidden_states)

    # ── ⑤ MLP ───────────────────────────────────────────────────────
    temp = self.mlp.act_fn(self.mlp.gate_proj(normed2)) * self.mlp.up_proj(normed2)
    mlp_out = self.mlp.down_proj(temp)

    # 记录 FFN temp similarity（图C3用）
    if skip_cache is not None and 'ffn_sim_log' in skip_cache:
        step = skip_cache.get('step', 0)
        if step > 0 and layer_idx in skip_cache.get('prev_temp', {}):
            prev_temp = skip_cache['prev_temp'][layer_idx]
            if temp.shape == prev_temp.shape:
                B, L, D_ff = temp.shape
                sim = F.cosine_similarity(
                    temp.reshape(B, -1), prev_temp.reshape(B, -1), dim=-1
                ).mean().item()
                skip_cache['ffn_sim_log'].append({
                    'step': step, 'layer': layer_idx, 'sim': sim,
                })

    # 记录 h_vs_ffn（图B4用）
    if skip_cache is not None and 'h_vs_ffn_log' in skip_cache:
        step = skip_cache.get('step', 0)
        if step > 0 and layer_idx in skip_cache.get('prev_mlp_out', {}):
            prev_mlp = skip_cache['prev_mlp_out'][layer_idx]
            if mlp_out.shape == prev_mlp.shape and skip_cache.get('sim_log'):
                last = skip_cache['sim_log'][-1]
                if last['layer'] == layer_idx and last['step'] == step:
                    h_sim = last['token_sim'].mean().item()
                    f_sim = F.cosine_similarity(
                        mlp_out.reshape(mlp_out.shape[0], -1),
                        prev_mlp.reshape(prev_mlp.shape[0], -1), dim=-1
                    ).mean().item()
                    skip_cache['h_vs_ffn_log'].append({
                        'h_sim': h_sim, 'ffn_sim': f_sim,
                        'layer': layer_idx, 'step': step,
                    })

    # Layer-level：跳过 MLP，直接复用上一步的 mlp_out
    if skip_mlp_only and skip_cache is not None:
        prev_mlp = skip_cache.get('prev_mlp_out', {}).get(layer_idx)
        if prev_mlp is not None and mlp_out.shape == prev_mlp.shape:
            mlp_out = prev_mlp

    # Token-level：替换跳过 token 的 mlp_out
    elif skip_mask is not None and skip_cache is not None:
        prev_mlp = skip_cache.get('prev_mlp_out', {}).get(layer_idx)
        if prev_mlp is not None and mlp_out.shape == prev_mlp.shape:
            mlp_out = torch.where(skip_mask.unsqueeze(-1), prev_mlp, mlp_out)

    # ── ⑥ 残差 ──────────────────────────────────────────────────────
    hidden_states = residual + mlp_out

    # ── 保存本步缓存 ─────────────────────────────────────────────────
    if skip_cache is not None:
        skip_cache.setdefault('prev_normed', {})[layer_idx] = normed.detach()
        skip_cache.setdefault('prev_attn_out', {})[layer_idx] = attn_out.detach()
        skip_cache.setdefault('prev_mlp_out', {})[layer_idx] = mlp_out.detach()
        skip_cache.setdefault('prev_layer_out', {})[layer_idx] = hidden_states.detach()
        skip_cache.setdefault('prev_temp', {})[layer_idx] = temp.detach()

    return hidden_states


def _compute_attention_stats(self, query_states, key_states, value_states):
    """题目指定的获取 attn_weight 的方法（图C1、C2用）"""
    def repeat_kv(hidden_states, n_rep):
        B, H, L, D = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(B, H, n_rep, L, D)
        return hidden_states.reshape(B, H * n_rep, L, D)

    if query_states.size(1) != key_states.size(1):
        n_groups = query_states.size(1) // key_states.size(1)
        key_states = repeat_kv(key_states, n_groups)
        value_states = repeat_kv(value_states, n_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
    attn_weights = torch.exp(attn_weights.float() * self.scaling)
    return attn_weights


def apply(model):
    layer_class = type(model.model.layers[0])
    layer_class.forward = _new_decoder_layer_forward
    attn_class = type(model.model.layers[0].self_attn)
    attn_class._compute_attention_stats = _compute_attention_stats
    print(f"✅ skip_patch 已应用到 {layer_class.__name__}")


def init_skip_cache(model, skip_mode='none', threshold=0.995, topk=0.25,
                    record_attn=False, record_ffn=False):
    old_ref = getattr(model.model.layers[0], '_skip_cache_ref', [None])
    old_cache = old_ref[0] if old_ref[0] is not None else {}

    skip_cache = {
        'step': 0,
        'skip_mode': skip_mode,
        'threshold': threshold,
        'topk': topk,
        'record_attn': record_attn,
        'record_ffn': record_ffn,
        'prev_normed': {},
        'prev_attn_out': {},
        'prev_mlp_out': {},
        'prev_layer_out': {},
        'prev_temp': {},
        'sim_log':       old_cache.get('sim_log', []),
        'flops_log':     old_cache.get('flops_log', []),
        'h_vs_attn_log': old_cache.get('h_vs_attn_log', []),
        'h_vs_ffn_log':  old_cache.get('h_vs_ffn_log', []),
        'ffn_sim_log':   old_cache.get('ffn_sim_log', []),
    }

    if record_attn:
        skip_cache['attn_sim_log']    = old_cache.get('attn_sim_log', [])
        skip_cache['attn_weight_log'] = old_cache.get('attn_weight_log', [])

    skip_cache_ref = [skip_cache]
    for layer in model.model.layers:
        layer._skip_cache_ref = skip_cache_ref
        layer.self_attn._skip_cache_ref = skip_cache_ref

    return skip_cache
