import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
tensor_parallel_size = 1
# os.environ["CUDA_VISIBLE_DEVICES"]='0'
# tensor_parallel_size = 1
import numpy as np

from math_evaluation.parser import extract_answer

from math_evaluation.grader import math_equal

from read_data import read_gsm8k, read_deepscaler, read_MATH

import random

from tqdm import tqdm

import sys
import json


MAX_INT = sys.maxsize

MAX_INT

from transformers import AutoTokenizer

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

tokenizer = AutoTokenizer.from_pretrained(model_name)

from vllm import LLM, SamplingParams


# llm = LLM(model=model_name, tensor_parallel_size=1, max_num_seqs=512, max_model_len=8192, gpu_memory_utilization=0.95)

train=True

data_name = "math"
if data_name == "gsm8k":
    num_sequences = 16
    gsm8k_input, gsm8k_output = read_gsm8k(train=train)
elif data_name == 'deepscaler':
    num_sequences = 8
    gsm8k_input, gsm8k_output = read_deepscaler()
elif data_name == 'math':
    num_sequences = 16
    gsm8k_input, gsm8k_output = read_MATH(train=train)

# gsm8k_input, gsm8k_output = gsm8k_input[:10], gsm8k_output[:10]

input_prompts = [
            tokenizer.apply_chat_template(
                [
        # {"role": "system", "content": ""},
        {"role": "user", "content": f"Please reason step by step, and put your final answer within \\boxed{{}}. {prompt}"},
    ],
                tokenize=False,
                add_generation_prompt=True,
            )
        for prompt in gsm8k_input]

len(input_prompts)


    
gold_anses = gsm8k_output

def generating():
    all_samples = []
    llm = LLM(model=model_name, tensor_parallel_size=tensor_parallel_size, dtype="bfloat16")
    for seed_id in range(num_sequences):
        sampling_params = SamplingParams(max_tokens=8192, temperature=0.7, stop=["<|im_end|>", "<｜end▁of▁sentence｜>"], stop_token_ids=[151645, 151643],
                                        seed=seed_id)
        responses = llm.generate(input_prompts, sampling_params=sampling_params, use_tqdm=True)
        pre_texts = [output.outputs[0].text for output in responses]
        all_samples.append(pre_texts)

        save_items(all_samples)


def save_items(all_samples):
    preds = np.array(all_samples).T.tolist()


    items = [{'prompt': input_prompt, 'gold_ans': gold_ans, 'preds': sub_preds} for input_prompt, gold_ans, sub_preds in zip(input_prompts, gold_anses, preds)]


    if not os.path.exists(f"synthesized_data/{data_name}"):
        os.makedirs(f"synthesized_data/{data_name}")

    with open(f'synthesized_data/{data_name}/items.json', 'w') as f:
        json.dump(items, f, indent=4)

generating()



print("Done")
