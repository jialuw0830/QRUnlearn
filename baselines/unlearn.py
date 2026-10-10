import sys
import pathlib
BASELINE_PATH = pathlib.Path(__file__).parent.resolve()
sys.path.append(BASELINE_PATH)

from baselines import it_unlearn

import argparse


def main():
    args = get_args()

    it_unlearn(
        args.model_dir, args.data_file, args.out_dir,
        retain_data_file=args.retain_data_file,
        loss_type=args.algo,
        per_device_batch_size=args.per_device_batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        max_len=args.max_len,
        tokenizer_dir=args.tokenizer_dir,
        resume_from_checkpoint=args.resume_from_checkpoint,
        alpha=args.alpha,
        loss_scale=args.loss_scale,
        threshold=args.threshold,
        # pipeline args
        add_noise=args.add_noise,
        outlier_indices_path=args.outlier_indices_path,
        noise_mode=args.noise_mode,
        noise_scale=args.noise_scale,
        # Layer-wise finetune args
        gradnorm_json=args.gradnorm_json,
        gradnorm_topk=args.gradnorm_topk,
        gradnorm_percent=args.gradnorm_percent,
        gradnorm_mode=args.gradnorm_mode,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    return;


def get_args():
    parser = argparse.ArgumentParser(description="Unlearning baselines")
    parser.add_argument('--algo', type=str)
    parser.add_argument(
        '--model_dir', type=str,
        help="Path to the target model's hf directory."
    )
    parser.add_argument(
        '--tokenizer_dir', type=str, default=None,
        help="Path to the tokenizer's hf directory. Defaults to the target model's directory."
    )
    parser.add_argument(
        '--data_file', type=str,
        help="Path to the forget set file."
    )
    parser.add_argument(
        '--out_dir', type=str,
        help="Path to the output model's hf directory. Creates the directory if it doesn't already exist."
    )
    parser.add_argument(
        '--max_len', type=int, default=4096,
        help="max length of input ids fed to the model"
    )
    parser.add_argument(
        '--resume_from_checkpoint', action='store_true',
    )

    # Gradient ascent & Gradient difference
    parser.add_argument('--per_device_batch_size', type=int, default=2)
    parser.add_argument(
        '--retain_data_file', type=str, default=None,
        help="Path to the retain set file. Required if algo is gradient difference (gd)."
    )
    parser.add_argument(
        '--lr', type=float, default=1e-5,
        help="Learning rate if algo is either gradient ascent (ga), gradient difference (gd)."
    )
    parser.add_argument(
        '--epochs', type=int, default=5,
        help="Number of epochs of training if algo is either gradient ascent (ga), gradient difference (gd)."
    )
    parser.add_argument(
        '--alpha', type=float, default=1.0,
        help="weight for utility constraint"
    )
    parser.add_argument(
        '--loss_scale', type=float, default=2.0,
        help="Loss scale factor for gdr/klr (multiplied with alpha). Default: 2.0"
    )
    parser.add_argument(
        '--threshold', type=int, default=95,
        help="threshold for SURE to filter salient modules"
    )
    
    # noise 
    parser.add_argument('--add_noise', action='store_true', help="Inject noise before training")
    parser.add_argument('--outlier_indices_path', type=str, default=None, help="Directory containing layer_{i}.pkl outlier indices (required for 'outlier' mode)")
    parser.add_argument('--noise_mode', type=str, default='row',
                        choices=['row', 'group'],
                        help="Noise mode: 'row' (row-wise, no grouping) or 'group' (grouped by columns). Default: 'row'")
    parser.add_argument('--noise_scale', type=float, default=1.0, 
                        help="Scale factor for noise magnitude. Default: 1.0")
    
    # Layer-wise finetune options
    parser.add_argument('--gradnorm_json', type=str, default=None, help="Path to gradnorm probe result JSON (from probe.py)")
    parser.add_argument('--gradnorm_topk', type=int, default=None, help="Top-k layers with largest grad norm to finetune")
    parser.add_argument('--gradnorm_percent', type=float, default=None, help="Percent of layers (0-100) with largest grad norm to finetune")
    parser.add_argument('--gradnorm_mode', type=str, default='high', choices=['high', 'low'], help="'high' for largest grad norm (default), 'low' for smallest")

    # LoRA finetune options
    parser.add_argument('--use_lora', action='store_true', help="Enable LoRA finetuning.")
    parser.add_argument('--lora_r', type=int, default=8, help="LoRA rank.")
    parser.add_argument('--lora_alpha', type=int, default=None, help="LoRA alpha (default: 2 * lora_r).")
    parser.add_argument('--lora_dropout', type=float, default=0.0, help="LoRA dropout.")

    args = parser.parse_args()

    if args.algo == 'gd':
        # gradient difference. Retain set is required
        assert args.retain_data_file is not None, "Gradient difference selected. Retain set required."

    if args.add_noise:
        if args.noise_mode in {'row', 'group'}:
            assert args.outlier_indices_path is not None, "outlier_indices_path is required when add_noise=True and noise_mode is 'row' or 'group'"

    return args


if __name__ == '__main__':
    main()
