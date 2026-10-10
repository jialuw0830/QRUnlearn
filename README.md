# QRUnlearn: Quantization-Robust Unlearning through the Lens of Retain–Forget Loss Landscapes Interaction [COLM 2026]

LLM unlearning that still holds after post-training quantization. An unlearned model often looks clean in full precision, but 4-bit quantization brings the forgotten knowledge back. We trace this to sharp forget-loss basins and add two components on top of standard unlearning objectives (GA / NPO with GDR / KLR):

- **Sensitivity-guided noisy regularization.** Score each weight by the Fisher ratio `ρ = F_forget / (F_retain + ε)` and inject quantization-scale noise into the rows that hold the top-τ weights, steering unlearning toward a flatter forget basin.
- **Forget-critical optimization.** Score each layer by the gradient-norm ratio `S = ‖∇L_forget‖ / (‖∇L_retain‖ + ε)` and update only the top-K layers, freezing the rest to preserve utility.

## Installation

```bash
conda env create -f environment.yml
conda activate py3102
```

## Data and models

Download the MUSE data into `data/`:

```bash
python load_data.py
```

Scripts read models from `$MODEL_ROOT` (default `/playpen-shared/huaizhi/jialu/models`), which should contain:

| Directory | Source |
|---|---|
| `MUSE-News_target` | [muse-bench/MUSE-News_target](https://huggingface.co/muse-bench/MUSE-News_target) |
| `MUSE-Books_target` | [muse-bench/MUSE-Books_target](https://huggingface.co/muse-bench/MUSE-Books_target) |
| `Llama-2-7b-hf` | [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) (tokenizer) |

```bash
export MODEL_ROOT=/path/to/models   # optional
```

## Pipeline

Every script takes the corpus as its first argument (`books` or `news`) and can be run from any directory. Choose GPUs with `CUDA_VISIBLE_DEVICES`.

```bash
# 1. Forget-critical layers: per-layer gradient-norm score S (Eq. 8)
bash baselines/scripts/probe.sh books
#    -> outputs/probe/books.json

# 2. Sensitive weights: per-weight Fisher ratio ρ (Eq. 6), top-τ per module
bash baselines/scripts/find_outliers.sh books
#    -> outputs/outliers/books/forget_specific_outliers_sens{0_1,0_5}.npz

# 3. Unlearn with noisy regularization + forget-critical layers, then evaluate in 4-bit
CUDA_VISIBLE_DEVICES=0 bash baselines/scripts/unlearn_books.sh
#    -> outputs/ckpt/books/<run>/ and <run>_4bit_metrics.csv

# News: run the same three steps with `news`, then
bash baselines/scripts/unlearn_news.sh
bash eval.sh news
```

### Outputs

```
outputs/
├── probe/{books,news}.json                              # layer scores
├── outliers/{books,news}/forget_specific_outliers_sens*.npz  # (row, col) of sensitive weights
└── ckpt/{books,news}/<run>/  +  <run>_4bit_metrics.csv  # unlearned models and metrics
```

## Configuration

Experiments are configured at the top of `baselines/scripts/unlearn_{books,news}.sh`.

| Variable | Meaning |
|---|---|
| `EXPERIMENT_CONFIGS` | `"algo:loss_scale:epochs"` (news also accepts `:sensitivity`). `loss_scale` is the retain weight λ_ret |
| `SENSITIVITY_PERCENTS` | selection ratio τ in % (paper: 0.1) |
| `GRADNORM_TOPK` | number of forget-critical layers to update |
| `ADD_NOISE`, `NOISE_SCALE` | enable noisy regularization and scale its magnitude |
| `LR_CANDIDATES` | learning rates |

Supported `--algo` values: `ga`, `npo`, `ga_gdr`, `ga_klr`, `npo_gdr`, `npo_klr`. The SURE baseline is available as `*_sure` (e.g. `ga_gdr_sure`) with `--threshold`.

Paper hyperparameters (η = 1e-5, τ = 0.1%, top-10 forget-critical layers for all runs):

| Method | News λ_ret | News epochs | Books λ_ret | Books epochs |
|---|---|---|---|---|
| GA_KLR | 5 | 10 | 5 | 1 |
| NPO_KLR | 3 | 15 | 3 | 5 |
| GA_GDR | 1 | 10 | 5 | 1 |
| NPO_GDR | 5 | 10 | 3 | 5 |

## Evaluation

`eval.sh <corpus>` evaluates every model under `outputs/ckpt/<corpus>` and skips models that already have a metrics file. Set `quantize_4bit` / `quantize_8bit` at the top of the script (both 0 = full precision). Metrics: VerbMem on D_f (`verbmem_f`), KnowMem on D_f (`knowmem_f`), KnowMem on D_r (`knowmem_r`), and PrivLeak (`privleak`).

To evaluate a single model directly:

```bash
python eval.py \
  --model_dirs outputs/ckpt/books/<run> --names <run> \
  --corpus books --quantize_4bit 1 \
  --out_file out.csv
```

## Repository layout

```
baselines/
├── unlearn.py              # unlearning entry point
├── baselines/
│   ├── iterative.py        # unlearning losses, layer freezing, noise injection
│   ├── probe.py            # forget-critical layer scores
│   ├── outlier.py          # Fisher-ratio sensitive weights
│   ├── noise.py            # quantization-scale noise on selected rows
│   └── dataset.py, utils.py, config.py
└── scripts/                # probe.sh, find_outliers.sh, unlearn_{books,news}.sh
eval.py, eval.sh            # MUSE evaluation (FP / 8-bit / 4-bit)
metrics/                    # VerbMem, KnowMem, PrivLeak
LLama_factory/              # utility evaluation helpers
```

## Citation

```bibtex
@inproceedings{wang2026qrunlearn,
  title     = {Quantization-Robust Unlearning through the Lens of Retain-Forget Loss Landscapes Interaction},
  author    = {Wang, Jialu and Deng, Jianing and Luo, Shuqing and Li, Yuanzhe and Wang, Dongwei and Hu, Jingtong and Yang, Huanrui and Wang, Song and Chen, Tianlong},
  booktitle = {Conference on Language Modeling (COLM)},
  year      = {2026}
}
```

## Acknowledgement

This code builds on [Catastrophic Failure of LLM Unlearning via Quantization](https://openreview.net/forum?id=lHSeDYamnz) (SURE) and [MUSE](https://github.com/jaechan-repo/muse_bench). Utility evaluation is based on [RWKU](https://github.com/jinzhuoran/RWKU).
