# import time
# time.sleep(3600*5)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch

from math_evaluation.parser import extract_answer

from read_data import read_gsm8k, read_deepscaler, read_MATH, read_ASDIV, read_SVAMP, read_math500

from transformers import AutoTokenizer
import swifter
import pandas as pd
from math_evaluation.parser import extract_answer
from math_evaluation.grader import math_equal
import argparse

# python 4seeds.py --data_name gsm8k --train_task gsm8k --device 0
parser = argparse.ArgumentParser(description="Training setting details")
parser.add_argument('--data_name', type=str, choices=["math", "gsm8k", "svamp", 'asdiv'], default='gsm8k', help='Training Dataset')
parser.add_argument('--train_task', type=str, choices=["math", "gsm8k"], default='gsm8k', help='Training Dataset')
parser.add_argument('--device', type=int, default=0, help='GPU id')
parser.add_argument('--compression_mode', type=int, default=0, help='GPU id')
args = parser.parse_args()

data_name = args.data_name
train_task = args.train_task
os.environ["CUDA_VISIBLE_DEVICES"]= str(args.device)

model_name = f"saved_models/{train_task}/control_icot-1-1"
# model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

save_name = model_name
while save_name.find('/') != -1:
    save_name = save_name[save_name.find('/')+1:]

print(save_name)

if not os.path.exists(f"txt/{data_name}"):
    os.makedirs(f"txt/{data_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)

from vllm import LLM, SamplingParams

llm = LLM(model=model_name, tensor_parallel_size=1)


train=False


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

# if model_name.find("selective") != -1:
input_prompts = [
        tokenizer.apply_chat_template(
            [
    {"role": "user", "content": f"{prompt} Please reason step by step, and put your final answer within \\boxed{{}}. Skip the first {args.compression_mode}% of thinking tokens.<｜Assistant｜>"},
],
            tokenize=False,
            add_generation_prompt=True,
        )
    for prompt in gsm8k_input]
# else:
# input_prompts = [
#         tokenizer.apply_chat_template(
#             [
#     {"role": "user", "content": f"{prompt} Please reason step by step, and put your final answer within \\boxed{{}}. Skip the first {int(args.compression_mode)*100}% of thinking tokens."},
# ],
#             tokenize=False,
#             add_generation_prompt=True,
#         )
    # for prompt in gsm8k_input]

# input_prompts = [prompt + "</think>" for prompt in input_prompts]

num_samples = 4



output_dct = [{'question': question, 'label_text': label_text}
                for question, label_text in zip(gsm8k_input, gsm8k_output)]

if not os.path.exists(f"saved_outputs/{data_name}"):
        os.makedirs(f"saved_outputs/{data_name}")


for seed in range(num_samples):
    sampling_params = SamplingParams(max_tokens=8192, temperature=0.7, stop=["<|im_end|>", "<｜end▁of▁sentence｜>"], stop_token_ids=[151645, 151643],
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
    torch.save(output_dct, f"saved_outputs/{data_name}/{prefix}_{save_name}_{args.compression_mode}.pt")

def compute_length(tokens):
    lens = [len(token) for token in tokens]
    return sum(lens)/len(lens)
def compute_acc(questions, pred_texts, label_texts):
    df = pd.DataFrame({"question": questions, 'pred_text': pred_texts, 'label_text': label_texts})
    df['pred'] = df['pred_text'].swifter.apply(lambda x: extract_answer(x, 'gsm8k', True))
    df['label'] = df['label_text'].swifter.apply(lambda x: extract_answer(x, 'gsm8k', True))
    preds = df['pred'].tolist()
    labels = df['label'].tolist()
    # track_lst = [math_equal(pred, label, timeout=True) for pred, label in zip(preds, labels)]
    df['pair'] = [(pred, label) for pred, label in zip(preds, labels)]
    df['check'] = df['pair'].swifter.apply(lambda x: math_equal(x[0], x[1], timeout=True))
    track_lst = df['check'].tolist()
    return sum(track_lst)/len(track_lst) 

def eval(data_name, compression_mode, save_name):
    path = f"saved_outputs/{data_name}/test_{save_name}_{compression_mode}.pt"
    outputs = torch.load(path)
    questions = [output['question'] for output in outputs]
    pred_texts_0 = [output['pred_text_0'] for output in outputs]
    pred_texts_1 = [output['pred_text_1'] for output in outputs]
    pred_texts_2 = [output['pred_text_2'] for output in outputs]
    pred_texts_3 = [output['pred_text_3'] for output in outputs]
    label_texts = [output['label_text'] for output in outputs]
    tokens_0 = [output['pred_tokens_0'] for output in outputs]
    tokens_1 = [output['pred_tokens_1'] for output in outputs]
    tokens_2 = [output['pred_tokens_2'] for output in outputs]
    tokens_3 = [output['pred_tokens_3'] for output in outputs]
    l0 = compute_length(tokens_0)
    l1 = compute_length(tokens_1)
    l2 = compute_length(tokens_2)
    l3 = compute_length(tokens_3)

    a0 = compute_acc(questions, pred_texts_0, label_texts)
    a1 = compute_acc(questions, pred_texts_1, label_texts)
    a2 = compute_acc(questions, pred_texts_2, label_texts)
    a3 = compute_acc(questions, pred_texts_3, label_texts)

    print("Average length", (l0+l1+l2+l3)/4)
    print("Average acc", (a0+a1+a2+a3)/4)
    save_txt_dir = f"txt/{data_name}/test_{save_name}_{args.compression_mode}.txt"
    file = open(save_txt_dir, "a+")
    print("Average length", (l0+l1+l2+l3)/4, file=file)
    print("Average acc", (a0+a1+a2+a3)/4, file=file)
    file.close()
eval(data_name, args.compression_mode, save_name)
