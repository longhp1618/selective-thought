# import time
# time.sleep(3600*5)

import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'
import torch

from math_evaluation.parser import extract_answer

from read_data import read_gsm8k, read_deepscaler, read_MATH, read_ASDIV, read_SVAMP, read_math500

from transformers import AutoTokenizer


model_name = "saved_models/math/icot-20-1"
# model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

save_name = model_name
while save_name.find('/') != -1:
    save_name = save_name[save_name.find('/')+1:]

print(save_name)

tokenizer = AutoTokenizer.from_pretrained(model_name)

from vllm import LLM, SamplingParams

llm = LLM(model=model_name, tensor_parallel_size=1)


train=False

data_name = "math"

if data_name == "gsm8k":
    gsm8k_input, gsm8k_output = read_gsm8k(train=train)
elif data_name == 'deepscaler':
    gsm8k_input, gsm8k_output = read_deepscaler()
elif data_name == 'math':
    # gsm8k_input, gsm8k_output = read_MATH(train=train)
    gsm8k_input, gsm8k_output = read_math500()
elif data_name == 'svamp':
    gsm8k_input, gsm8k_output = read_SVAMP()
elif data_name == 'asdiv':
    gsm8k_input, gsm8k_output = read_ASDIV()


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

# input_prompts = [prompt + "</think>" for prompt in input_prompts]

num_samples = 4



output_dct = [{'question': question, 'label_text': label_text}
                for question, label_text in zip(gsm8k_input, gsm8k_output)]

if not os.path.exists(f"saved_outputs/{data_name}"):
        os.makedirs(f"saved_outputs/{data_name}")


for seed in range(num_samples):
    sampling_params = SamplingParams(max_tokens=32768, temperature=0.7, stop=["<|im_end|>", "<｜end▁of▁sentence｜>"], stop_token_ids=[151645, 151643],
                                    seed=seed)
    responses = llm.generate(input_prompts, sampling_params=sampling_params, use_tqdm=True)
    pred_texts = [response.outputs[0].text.strip() for response in responses]
    pred_token_ids = [response.outputs[0].token_ids for response in responses]
    for idx, pair in enumerate(zip(pred_texts, pred_token_ids)):
        pred_text, pred_tokens = pair
        output_dct[idx][f'pred_text_{seed}'] = pred_text
        output_dct[idx][f'pred_tokens_{seed}'] = pred_tokens
    
    if train==True:
        prefix='train'
    else:
        prefix='test'
    torch.save(output_dct, f"saved_outputs/{data_name}/{prefix}_{save_name}.pt")