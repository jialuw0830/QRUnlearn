import numpy as np
import torch


# def _add_noise_on_rows(weight: torch.Tensor, row_indices: torch.Tensor) -> None:
#     if row_indices is None or row_indices.numel() == 0:
#         return
#     for r in row_indices.to(weight.device).long().unique().tolist():
#         row = weight[r]
#         delta = (row.amax() - row.amin()) / 255.0
#         weight[r] = row + (torch.rand_like(row) - 0.5) * delta 



@torch.no_grad()
def _add_uniform_qnoise_on_rows(
    weight: torch.Tensor,
    row_indices: torch.Tensor,
    bits: int = 4,
    clip: str = "minmax",
    c: float = 2.5,
    noise_scale: float = 1.0,
    group_size: int = 128,
) -> None:
    if row_indices is None or row_indices.numel() == 0:
        return
    idx = row_indices.to(weight.device).long().unique()
    rows = weight.index_select(0, idx)
    reduce_dims = tuple(range(1, rows.ndim))
    qmax = 2 ** (bits - 1) - 1

    if rows.ndim < 2 or group_size <= 1:
        if clip == "std":
            alpha = c * rows.float().std(dim=reduce_dims, keepdim=True)
        else:
            alpha = rows.float().abs().amax(dim=reduce_dims, keepdim=True)
        scale = (alpha / qmax).clamp_min(torch.finfo(torch.float32).eps)
        noise = (torch.rand_like(rows) - 0.5) * scale * noise_scale
        rows.add_(noise.to(rows.dtype))
    else:
        num_cols = rows.shape[1]
        noise = torch.zeros_like(rows)
        for start in range(0, num_cols, group_size):
            end = min(start + group_size, num_cols)
            segment = rows[:, start:end]
            if clip == "std":
                alpha = c * segment.float().std(dim=reduce_dims, keepdim=True)
            else:
                alpha = segment.float().abs().amax(dim=reduce_dims, keepdim=True)
            scale = (alpha / qmax).clamp_min(torch.finfo(torch.float32).eps)
            chunk = (torch.rand_like(segment) - 0.5) * scale * noise_scale
            noise[:, start:end] = chunk.to(rows.dtype)
        rows.add_(noise)
    weight.index_copy_(0, idx, rows)

def add_noise_to_model(model: torch.nn.Module,
                       npz_path: str,
                       noise_scale: float = 1.0,
                       group_size: int = 128) -> torch.nn.Module:
    """
    Add row-wise noise to modules based on outlier row indices from NPZ.
    Set group_size=1 for pure row-wise (no grouping), or >1 (e.g., 128) for grouped columns.
    """
    data = np.load(npz_path)

    def get_module_names():
        return ["q", "k", "v", "o", "gate", "up", "down"]

    def get_modules(layer):
        return [
            layer.self_attn.q_proj,
            layer.self_attn.k_proj,
            layer.self_attn.v_proj,
            layer.self_attn.o_proj,
            layer.mlp.gate_proj,
            layer.mlp.up_proj,
            layer.mlp.down_proj,
        ]

    noisy = model.bfloat16()
    layers = noisy.model.layers
    names = get_module_names()

    layer_ids = set()
    for k in data.files:
        if k.startswith("layer_"):
            parts = k.split("_")
            if len(parts) >= 3 and parts[1].isdigit():
                layer_ids.add(int(parts[1]))
    nlayers = max(layer_ids) + 1 if len(layer_ids) > 0 else 0

    for l in range(min(nlayers, len(layers))):
        layer = layers[l]
        for name, module in zip(names, get_modules(layer)):
            key_rows = f"layer_{l}_{name}_rows"
            if key_rows in data.files:
                rows_np = data[key_rows]
                if rows_np.size > 0:
                    rows = torch.as_tensor(rows_np, dtype=torch.long, device=module.weight.device)
                    _add_uniform_qnoise_on_rows(module.weight.data,
                                                rows,
                                                noise_scale=noise_scale,
                                                group_size=group_size)
    return noisy


# End of file with minimal API

