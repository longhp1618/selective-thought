# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"]='4'
import torch

# %%
from math_evaluation.parser import extract_answer

# %%
from read_data import read_gsm8k

# %%
from transformers import AutoTokenizer

# %%
# model_name = "saved_models/sft-5-3"
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# %%
tokenizer = AutoTokenizer.from_pretrained(model_name)

# %%
from vllm import LLM, SamplingParams

llm = LLM(model=model_name, tensor_parallel_size=1)


# %%
train=False

# %%
gsm8k_input, gsm8k_output = read_gsm8k(train=train)

# %%
# gsm8k_input, gsm8k_output = gsm8k_input[:2], gsm8k_output[:2]

# %%
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

# input_prompts = [prompt + "</think>" for prompt in input_prompts]

# %%
num_samples = 4

# %%
output_dct = [{'question': question, 'label_text': label_text}
                for question, label_text in zip(gsm8k_input, gsm8k_output)]

# %%
len(output_dct)

# %%
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
    torch.save(output_dct, f"saved_outputs/{prefix}_{model_name[model_name.find('/')+1:]}.pt")

# %%



