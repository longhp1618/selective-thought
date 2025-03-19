# %%
import os
# os.environ["CUDA_VISIBLE_DEVICES"]='1'

# %%
from math_evaluation.parser import extract_answer

# %%
from read_data import read_gsm8k

# %%
from transformers import AutoTokenizer

# %%
# model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
model_name = "saved_models/dpo-1"

# %%
tokenizer = AutoTokenizer.from_pretrained(model_name)

# %%
from vllm import LLM, SamplingParams

llm = LLM(model=model_name, tensor_parallel_size=1)
sampling_params = SamplingParams(max_tokens=32768, temperature=0.7, stop=["<|im_end|>", "<｜end▁of▁sentence｜>"], stop_token_ids=[151645, 151643], n=1)


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

# %%
# input_prompts = input_prompts[:10]

# %%
# input_prompts = [i+"</think>" for i in input_prompts]

# %%
responses = llm.generate(input_prompts, sampling_params=sampling_params, use_tqdm=True)

# %%
assert len(responses) == len(gsm8k_input)

# %%
print(responses[0].outputs)

# %%
pred_texts = [response.outputs[0].text.strip() for response in responses]
pred_token_ids = [response.outputs[0].token_ids for response in responses]
# answers = [extract_answer(pred,  "gsm8k", use_last_number=False) for pred in predictions]
# labels = []

# %%
output_dct = [{'question': question, 'pred_text': pred_text, 'label_text': label_text, 'pred_tokens': pred_tokens}
              for question, pred_text, label_text, pred_tokens in zip(gsm8k_input, pred_texts, gsm8k_output, pred_token_ids)]

# %%
import torch

# %%
if train==True:
    prefix='train'
else:
    prefix='test'
torch.save(output_dct, f"saved_outputs/dpo_{prefix}_{model_name[model_name.find('/')+1:]}.pt")

# %%



