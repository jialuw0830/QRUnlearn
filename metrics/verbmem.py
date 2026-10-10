from .logger import RougeEvalLogger

from tqdm.contrib import tzip
from typing import List


def eval(
    model, tokenizer,
    prompts: List[str], gts: List[str],
    max_new_tokens : int = 128
):
    logger = RougeEvalLogger()
    for prompt, gt in tzip(prompts, gts):
        # Encode the `prompt` into `input_ids`
        input_ids = tokenizer(
            prompt,
            return_tensors='pt',
            add_special_tokens=True,
            padding=True,
            pad_to_multiple_of=8
        ).input_ids
        assert len(input_ids) == 1

        gt_ids = tokenizer(gt, return_tensors='pt', add_special_tokens=True).input_ids[:, :max_new_tokens]

        # 修复bitsandbytes INT8量化对齐问题
        import torch.nn.functional as F
        
        # 确保有 pad_token
        if getattr(tokenizer, "pad_token_id", None) is None:
            if getattr(tokenizer, "eos_token_id", None) is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                raise ValueError("Tokenizer has no pad_token_id or eos_token_id.")
        
        # 对齐输入序列长度到8的倍数
        input_ids_padded = input_ids.to(model.device)
        seq_len = input_ids_padded.size(1)
        pad = (-seq_len) % 8
        if pad:
            # 左侧补齐
            input_ids_padded = F.pad(input_ids_padded, (pad, 0), value=tokenizer.pad_token_id)
        
        # Use the `model` to generate the continuation of the `input_ids`.
        output_ids = model.generate(
            input_ids_padded,
            max_new_tokens=max_new_tokens,
            # do_sample=False,
            do_sample=True,
            temperature=0.9,
            pad_token_id=tokenizer.pad_token_id
            # pad_token_id=tokenizer.eos_token_id
            )
        
        # 调整输出长度计算，考虑padding
        output_ids = output_ids[:, len(input_ids_padded[0]):]
        output = tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True)[0]
        gt_short = tokenizer.batch_decode(
            gt_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True)[0]
        logger.log(prompt, gt_short, output)

    return logger.report()
