from .logger import RougeEvalLogger

from tqdm.contrib import tzip
from typing import List


def get_prefix_before_words_occur(string: str, words: List[str]) -> str:
    for word in words: string = string.split(word)[0]
    return string


def eval(
    model, tokenizer,
    questions: List[str], answers: List[str],
    icl_qs: List[str] = [], icl_as: List[str] = [],
    max_new_tokens : int = 32
):
    assert len(questions) == len(answers)
    assert len(icl_qs) == len(icl_as)

    logger = RougeEvalLogger()
    general_prompt: str = ""

    # Few-shot prompting
    for question, answer in zip(icl_qs, icl_as):
        general_prompt += f"Question: {question}\nAnswer: {answer}\n\n"

    for question, answer in tzip(questions, answers):
        prompt = general_prompt + f"Question: {question}\nAnswer: "

        # Encode the `prompt` into `input_ids`
        input_ids = tokenizer(
            prompt,
            return_tensors='pt',
            add_special_tokens=True,
            padding=True,
            pad_to_multiple_of=8).input_ids

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
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id)
        
        # 调整输出长度计算，考虑padding
        output_ids = output_ids[:, len(input_ids_padded[0]):]

        output = tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True)[0]

        output = get_prefix_before_words_occur(output, ["\n\n", "\nQuestion", "Question:"])
        logger.log(prompt, answer, output, question=question)

    return logger.report()
