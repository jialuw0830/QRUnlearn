"""
Find forget-specific outliers: weights with high Fisher on forget data and low Fisher on retain data.

score = F_forget / (F_retain + eps); per module, keep the top `sensitivity_percent`% and save
their (row, col) indices to `{output_folder}/forget_specific_outliers_sens{X}.npz`.
"""

import os
import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import transformers

from .dataset import DefaultDataset


def modules(layer):
    """Linear modules of a LLaMA layer, keyed by short name."""
    attn, mlp = layer.self_attn, layer.mlp
    return {
        "q": attn.q_proj, "k": attn.k_proj, "v": attn.v_proj, "o": attn.o_proj,
        "gate": mlp.gate_proj, "up": mlp.up_proj, "down": mlp.down_proj,
    }


def fisher(model, dataloader, device="cuda"):
    """Fisher information (sum of squared gradients) for every linear weight: {layer: {name: tensor}}."""
    model = model.to(device).bfloat16()
    model.train()
    layers = model.model.layers

    for param in model.model.parameters():
        param.requires_grad = True

    grads = {l: {n: torch.zeros_like(m.weight.data) for n, m in modules(layer).items()}
             for l, layer in enumerate(layers)}

    hooks = []
    for l, layer in enumerate(layers):
        for n, m in modules(layer).items():
            hooks.append(m.weight.register_hook(lambda g, acc=grads[l][n]: acc.add_(g.pow(2))))

    for data in tqdm(dataloader, desc="fisher"):
        data = data.to(device)
        model.zero_grad()
        model(input_ids=data, labels=data).loss.backward()

    for h in hooks:
        h.remove()

    return {l: {n: g.float().cpu() for n, g in d.items()} for l, d in grads.items()}


def find(model, forget_loader, retain_loader, output_folder, sensitivity_percent, eps=1e-8):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fisher_forget = fisher(model, forget_loader, device)
    fisher_retain = fisher(model, retain_loader, device)

    outliers = {}
    for l in fisher_forget:
        for n in fisher_forget[l]:
            score = (fisher_forget[l][n] / (fisher_retain[l][n] + eps)).numpy()
            k = int(score.size * sensitivity_percent / 100)
            if k == 0:
                continue
            thres = np.partition(score.flatten(), -k)[-k]
            rows, cols = np.where(score > thres)
            outliers[f"layer_{l}_{n}_rows"] = rows.astype(np.int32)
            outliers[f"layer_{l}_{n}_cols"] = cols.astype(np.int32)

    os.makedirs(output_folder, exist_ok=True)
    sens = str(sensitivity_percent).replace(".", "_")
    output_file = os.path.join(output_folder, f"forget_specific_outliers_sens{sens}.npz")
    np.savez_compressed(output_file, **outliers)
    print(f"Saved to {output_file}")


def loader(txt, tokenizer, max_length, batch_size, n=100):
    dataset = DefaultDataset(txt, tokenizer, max_len=max_length, add_bos_token=True)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    return DataLoader(Subset(dataset, indices[:n]), batch_size=batch_size, shuffle=False, collate_fn=torch.stack)


def main():
    parser = argparse.ArgumentParser(description="Find forget-specific outliers")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Defaults to model_path")
    parser.add_argument("--forget_txt", type=str, required=True)
    parser.add_argument("--retain_txt", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--sensitivity_percent", type=float, default=0.05, help="Top-k percent per module")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eps", type=float, default=1e-8)
    args = parser.parse_args()

    model = transformers.AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_path or args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    forget_loader = loader(args.forget_txt, tokenizer, args.max_length, args.batch_size)
    retain_loader = loader(args.retain_txt, tokenizer, args.max_length, args.batch_size)
    find(model, forget_loader, retain_loader, args.output_folder, args.sensitivity_percent, args.eps)


if __name__ == "__main__":
    main()
