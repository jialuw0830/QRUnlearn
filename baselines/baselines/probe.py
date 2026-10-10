"""
Probe per-layer forget/retain gradient log-ratio.

For each batch, score[l] = log(||g_forget[l]|| + eps) - log(||g_retain[l]|| + eps),
averaged over batches. Model is never updated.
"""

import argparse
import json
import math
import os

import torch
from tqdm import tqdm

from .utils import load_model_and_tokenizer
from .dataset import ForgetRetainDataset


def norms(model, layers, batch):
    """Per-layer gradient L2 norm of the CE loss on `batch`."""
    model.zero_grad(set_to_none=True)
    model(**{k: v.to(model.device) for k, v in batch.items()}).loss.backward()
    return [
        torch.stack([p.grad.float().norm() for p in layer.parameters()]).norm().item()
        for layer in layers
    ]


def probe(model, loader, eps=1e-8):
    model.requires_grad_(False)
    layers = model.model.layers
    layers.requires_grad_(True)

    scores = torch.zeros(len(layers), dtype=torch.float64)
    for forget, retain in tqdm(loader, desc="probe"):
        f, r = norms(model, layers, forget), norms(model, layers, retain)
        scores += torch.tensor([math.log(a + eps) - math.log(b + eps) for a, b in zip(f, r)])
    return (scores / len(loader)).tolist()


def main():
    parser = argparse.ArgumentParser(description="Probe per-layer forget/retain gradient log-ratio")
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--tokenizer_dir', type=str, default=None, help="Defaults to model_dir")
    parser.add_argument('--forget_data_file', type=str, required=True)
    parser.add_argument('--retain_data_file', type=str, required=True)
    parser.add_argument('--max_len', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model_dir, tokenizer_dir=args.tokenizer_dir or args.model_dir)
    model.train()

    dataset = ForgetRetainDataset(args.forget_data_file, tokenizer,
                                  retain_file_path=args.retain_data_file, max_len=args.max_len)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                         collate_fn=dataset.get_collate_fn())

    scores = probe(model, loader, args.eps)
    for i, s in enumerate(scores):
        print(f"Layer {i}: {s:.6f}")

    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump({'layer_grad_norms': {str(i): s for i, s in enumerate(scores)}}, f, indent=2)
    print(f"Saved to {args.output_file}")


if __name__ == '__main__':
    main()
