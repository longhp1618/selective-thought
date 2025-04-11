import argparse
import os

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataloader import create_dataloader
from read_data import read_gsm8k
import json

# python create_hidden_data.py --task math --device 0 --bs 4
def arguments():
    parser = argparse.ArgumentParser(description="Training setting details")
    parser.add_argument('--task', type=str, choices=["math", "gsm8k"], default='gsm8k', help='Training Dataset')
    # parser.add_argument('--model_gen', type=str, choices=["t5", "llama", "mistral"], default='mistral', help='Model family')
    # parser.add_argument('--size', type=str, choices=["small", "base", "large", '7B', '13B'], default='7B', help='Model size')
    parser.add_argument('--device', type=int, default=1, help='GPU id')
    parser.add_argument("--bs", type=int, default=8, help="batch size for training intepolator")

    args = parser.parse_args()
    args.model_gen = "r1"
    args.size = "1.5B"
    return args


def prepare(args):
    save_name = f"{args.model_gen}_{args.size}"
    task = args.task
    device = torch.device(f"cuda:{args.device}")
    bs = args.bs

    save_path = f"saved_hiddens/{task}/{save_name}/train"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map=f"cuda:{args.device}"
    )

    # if task == 'gsm8k':
    load_path = f"synthesized_data/{task}/icot.jsonl"
    with open(load_path, "r") as f:
        # items = json.load(f)
        items = [json.loads(line) for line in f]
    train_input = [item['question'] for item in items]
    train_output = [item['answer'] for item in items]
    # train_input = train_input[:5]
    # train_output = train_output[:5]

    MAX_LEN = 8192

    dataloader = create_dataloader(train_input, train_output, tokenizer, bs, MAX_LEN, shuffle=False)

    return dataloader, model, tokenizer, device, save_path


@torch.no_grad()
def get_hidden(batch, model, device=None):
    if device is None:
        device = model.device
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    ans_mask = batch['ans_mask'].to(device)
    rea_mask = batch['rea_mask'].to(device)
    
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    last_hidden = outputs.hidden_states[-1]

    return input_ids.detach(), rea_mask.detach(), ans_mask.detach(), last_hidden.detach(), batch['completion'], batch['instruction']

@torch.no_grad()
def process(labels, last_hidden, ans_mask, rea_mask, completion_texts):
    shifted_labels = torch.full_like(labels, -100)
    shifted_labels[:, :-1] = labels[:, 1:]
    rea_tokens = (shifted_labels+1)*rea_mask
    rea_tokens = [(i[i != 0]-1).int().cpu() for i in rea_tokens]
    ans_tokens = (shifted_labels+1)*ans_mask
    ans_tokens = [(i[i != 0]-1).int().cpu() for i in ans_tokens]

    rea_hiddens = last_hidden * rea_mask.unsqueeze(-1)
    rea_hiddens = [i[torch.any(i != 0, dim=1)].cpu() for i in rea_hiddens]
    ans_hiddens = last_hidden * ans_mask.unsqueeze(-1)
    ans_hiddens = [i[torch.any(i != 0, dim=1)].cpu() for i in ans_hiddens]

    # think_pos = completion_texts.find("</think>")
    # if think_pos != -1:
    short_cot_texts = [completion[completion.find("</think>"):] for completion in completion_texts]

    return rea_tokens, rea_hiddens, ans_tokens, ans_hiddens, short_cot_texts


@torch.no_grad()
def infer(dataloader, model, save_path, device=None):
    if device is None:
        device = model.device
    all_instruction_texts, all_short_cot_texts, all_rea_tokens, all_ans_tokens, all_rea_hiddens, all_ans_hiddens = [], [], [], [], [], []
    batch_iter = tqdm(dataloader, desc='Training', position=0, leave=True)
    for batch in batch_iter:
        labels, rea_mask, ans_mask, last_hidden, completion_texts, instruction_texts = get_hidden(batch, model, device)

        rea_tokens, rea_hiddens, ans_tokens, ans_hiddens, short_cot_texts = process(labels, last_hidden, ans_mask, rea_mask, completion_texts)

        all_instruction_texts.extend(instruction_texts)
        all_short_cot_texts.extend(short_cot_texts)
        all_rea_tokens.extend(rea_tokens)
        all_rea_hiddens.extend(rea_hiddens)
        all_ans_tokens.extend(ans_tokens)
        all_ans_hiddens.extend(ans_hiddens)

    data = (all_instruction_texts, all_short_cot_texts, all_rea_tokens, all_ans_tokens, all_rea_hiddens, all_ans_hiddens)
    torch.save(data, f"{save_path}/hidden_data.pt")

    return data


if __name__ == "__main__":
    args = arguments()
    dataloader, model, tokenizer, device, save_path = prepare(args)
    data = infer(dataloader, model, save_path, device)
