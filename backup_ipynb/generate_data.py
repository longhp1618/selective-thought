import os
os.environ["CUDA_VISIBLE_DEVICES"]='4, 5, 6, 7'
tensor_parallel_size = 4
import numpy as np

from math_evaluation.parser import extract_answer

from math_evaluation.grader import math_equal

from read_data import read_gsm8k

import random

from tqdm import tqdm

import sys
MAX_INT = sys.maxsize

MAX_INT

from transformers import AutoTokenizer

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

tokenizer = AutoTokenizer.from_pretrained(model_name)

from vllm import LLM, SamplingParams

# llm = LLM(model=model_name, tensor_parallel_size=1, max_num_seqs=512, max_model_len=8192, gpu_memory_utilization=0.95)
llm = LLM(model=model_name, tensor_parallel_size=tensor_parallel_size, dtype="bfloat16")

train=True

gsm8k_input, gsm8k_output = read_gsm8k(train=train)

# gsm8k_input, gsm8k_output = gsm8k_input[:2], gsm8k_output[:2]

input_prompts = [
            tokenizer.apply_chat_template(
                [
        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{{}}."},
        {"role": "user", "content": prompt},
    ] ,
                tokenize=False,
                add_generation_prompt=True,
            )
        for prompt in gsm8k_input]

len(input_prompts)

num_sequences = 16
all_samples = []
for _ in range(num_sequences):
    sampling_params = SamplingParams(max_tokens=8192, temperature=0.7, stop=["<|im_end|>", "<｜end▁of▁sentence｜>"], stop_token_ids=[151645, 151643],
                                    seed=random.randint(0, 2**32-1))
    responses = llm.generate(input_prompts, sampling_params=sampling_params, use_tqdm=True)
    pre_texts = [output.outputs[0].text for output in responses]
    all_samples.append(pre_texts)

all_samples

preds = np.array(all_samples).T.tolist()

# preds = [[output.text for output in response.outputs] for response in responses]

pred_ans = [[extract_answer(sample, 'gsm8k', False) for sample in pred] for pred in preds]

gold_anses = [gold_seq[gold_seq.find("####"):].replace("####", "").strip() for gold_seq in gsm8k_output]

items = [{'prompt': input_prompt, 'gold_ans': gold_ans, 'positives': [], 'negatives': []} for input_prompt, gold_ans in zip(input_prompts, gold_anses)]

for idx in range(len(items)):
    for jdx in range(num_sequences):
        if math_equal(pred_ans[idx][jdx], items[idx]['gold_ans']):
            items[idx]['positives'].append(preds[idx][jdx])
        else:
            items[idx]['negatives'].append(preds[idx][jdx])

for item in items:
    if len(item['positives']) != 0:
        positives = item['positives']
        lengths = [len(tokenizer.encode(string)) for string in positives]
        sorted_positives = [s for score, s in sorted(zip(lengths, positives))]
        item['positives'] = sorted_positives
    if len(item['negatives']) != 0:
        negatives = item['negatives']
        lengths = [len(tokenizer.encode(string)) for string in negatives]
        sorted_negatives = [s for score, s in sorted(zip(lengths, negatives))]
        item['negatives'] = sorted_negatives

def make_dpo_data(item, last_epoch_item, t, drop_rate):
    prompt = item['prompt']
    positives = item['positives']
    negatives = item['negatives']

    if len(negatives) == 0:
        chosen = positives[0] # get the shortest correct solutions as the prefered samples
        reject = positives[-1] # get the longest correct solutions as unprefered samples if there is no incorrect solution

    if len(positives) == 0:
        if len(last_epoch_item['positives']) == 0:
            return None
        chosen = last_epoch_item['positives'][0] # get the shortest correct solutions in the previous synthesized data as the prefered samples
        reject = negatives[-1] # get the longest incorrect solutions as unprefered samples
    
    if len(positives) != 0 and len(negatives) != 0:
        chosen = positives[0] # get the shortest correct solutions as the prefered samples
        reject = negatives[-1] # # get the longest incorrect solutions as unprefered samples
    
    original_chosen = chosen
    lst = chosen.split("</think>")
    if len(lst) == 2:
        long_cot, answer = lst[0], lst[1]
        answer = "</think>" + answer
    else:
        long_cot = lst[0]
        answer = ""

    long_cot_encoded = t.encode(long_cot)
    drop_len = round(len(long_cot_encoded)*drop_rate)
    long_cot_drop = t.decode(long_cot_encoded[drop_len:], skip_special_tokens=True)
    chosen = long_cot_drop + answer
    if not chosen.endswith(t.eos_token):
        chosen += t.eos_token
    if not original_chosen.endswith(t.eos_token):
        original_chosen += t.eos_token
    if not reject.endswith(t.eos_token):
        reject += t.eos_token

    data_sample = [{'prompt': prompt, 'chosen': chosen, 'rejected': original_chosen},
                   {'prompt': prompt, 'chosen': chosen, 'rejected': reject}]
    return data_sample



import json
with open(f'synthesized_data/items_epoch0.json', 'w') as f:
    json.dump(items, f, indent=4)
items

data_dpo = []

for item in items:
    data_dpo.extend(make_dpo_data(item, item, tokenizer, 0.1))

print("Length of dpo data:", len(data_dpo))

data_dpo

import json
with open(f'synthesized_data/dpo_epoch0.jsonl', 'w', encoding='utf-8') as file:
    for item in data_dpo:
        file.write(json.dumps(item) + '\n')

with open(f'synthesized_data/dpo_epoch0.json', 'w') as f:
    json.dump(data_dpo, f, indent=4)



print("Done saving")
