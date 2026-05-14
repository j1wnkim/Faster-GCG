import torch
from transformers import PreTrainedConfig
from transformers.cache_utils import (
    Cache,
    CacheLayerMixin,
    DynamicCache,
    LinearAttentionCacheLayerMixin,
    StaticCache,
)


@torch.no_grad()
def copy_prefix_cache(prefix_cache: Cache, config: PreTrainedConfig, batch_size=1):
    if isinstance(prefix_cache, StaticCache):
        prefix_cache_batch = StaticCache(
            config=config, max_cache_len=prefix_cache.get_seq_length()
        )
    elif isinstance(prefix_cache, DynamicCache):
        prefix_cache_batch = DynamicCache(config=config)
    else:
        raise ValueError(f"Prefix cache type {str(type(prefix_cache))} unknown")
    for layer_idx, layer in enumerate(prefix_cache.layers):
        if isinstance(layer, LinearAttentionCacheLayerMixin):
            c_exp = layer.conv_states.expand(batch_size, -1, -1, -1).contiguous()
            r_exp = layer.recurrent_states.expand(batch_size, -1, -1, -1).contiguous()
            prefix_cache_batch.update_conv_state(c_exp, layer_idx=layer_idx)
            prefix_cache_batch.update_recurrent_state(r_exp, layer_idx=layer_idx)
        else:
            # isinstance(layer, CacheLayerMixin)
            # Expand prefix KV to batch size
            k_exp = layer.keys.expand(batch_size, -1, -1, -1).contiguous()
            v_exp = layer.values.expand(batch_size, -1, -1, -1).contiguous()
            prefix_cache_batch.update(
                key_states=k_exp,
                value_states=v_exp,
                layer_idx=layer_idx,
            )
    return prefix_cache_batch
