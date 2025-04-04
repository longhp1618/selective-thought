# SPDX-License-Identifier: Apache-2.0
# usage:
# VLLM_USE_V1=1 python examples/offline_inference/data_parallel.py
# we need to have a launcher to create multiple data parallel
# ranks. And each rank will create a vLLM instance to process its own prompts.
import os

from vllm import LLM, SamplingParams
from vllm.utils import get_open_port
from read_data import read_deepscaler, read_gsm8k, read_MATH
from transformers import AutoTokenizer
import numpy as np
import json
import math

GPUs_per_dp_rank = 1
num_proc = 1
DP_size = 4*num_proc

def save_items(all_samples, dp_rank, prompts, gold_anses):
    preds = np.array(all_samples).T.tolist()

    items = [{'prompt': input_prompt, 'gold_ans': gold_ans, 'preds': sub_preds} for input_prompt, gold_ans, sub_preds in zip(prompts, gold_anses, preds)]


    with open(f'synthesized_data/{data_name}/items_{dp_rank}.json', 'w') as f:
        json.dump(items, f, indent=4)

def main(dp_size, dp_rank, dp_master_ip, dp_master_port, GPUs_per_dp_rank, all_prompts, all_gold_anses):
    os.environ["VLLM_DP_RANK"] = str(dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)
    # set devices for each dp_rank
    # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
    #     str(i) for i in range(dp_rank * GPUs_per_dp_rank, (dp_rank + 1) *
    #                           GPUs_per_dp_rank))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(math.floor(dp_rank/num_proc))

    
    # with DP, each rank should process different prompts.
    # usually all the DP ranks process a full dataset,
    # and each rank processes a different part of the dataset.
    promts_per_rank = len(all_prompts) // dp_size
    start = dp_rank * promts_per_rank
    end = start + promts_per_rank
    prompts = all_prompts[start:end]
    gold_anses = all_gold_anses[start:end]

    if len(prompts) == 0:
        pass

    print(f"DP rank {dp_rank} needs to process {len(prompts)} prompts")

    # Create a sampling params object.
    # since we are doing data parallel, every rank can have different
    # sampling params. here we set different max_tokens for different
    # ranks for demonstration.
    # sampling_params = SamplingParams(temperature=0.8,
    #                                  top_p=0.95,
    #                                  max_tokens=16 * (dp_rank + 1))

    # Create an LLM.
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    llm = LLM(model=model_name,
              tensor_parallel_size=GPUs_per_dp_rank,  gpu_memory_utilization=0.96/num_proc)
    
    all_samples = []
    for seed_id in range(num_sequences):
        sampling_params = SamplingParams(max_tokens=8192, temperature=0.7, stop=["<|im_end|>", "<｜end▁of▁sentence｜>"], stop_token_ids=[151645, 151643],
                                        seed=seed_id)
        responses = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)
        pre_texts = [output.outputs[0].text for output in responses]
        all_samples.append(pre_texts)

        save_items(all_samples, dp_rank, prompts, gold_anses)



if __name__ == "__main__":

    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train=True

    data_name = "math"
    if data_name == "gsm8k":
        num_sequences = 16
        original_inputs, all_gold_anses = read_gsm8k(train=train)
    elif data_name == 'deepscaler':
        num_sequences = 8
        original_inputs, all_gold_anses = read_deepscaler()
    elif data_name == 'math':
        num_sequences = 16
        original_inputs, all_gold_anses = read_MATH(train=train)

    # original_inputs, all_gold_anses = original_inputs[:10], all_gold_anses[:10]

    if not os.path.exists(f"synthesized_data/{data_name}"):
        os.makedirs(f"synthesized_data/{data_name}")

    all_prompts = [
            tokenizer.apply_chat_template(
                [
        # {"role": "system", "content": ""},
        {"role": "user", "content": f"Please reason step by step, and put your final answer within \\boxed{{}}. {prompt}"},
    ],
                tokenize=False,
                add_generation_prompt=True,
            )
        for prompt in original_inputs]

    from multiprocessing import Process
    dp_master_ip = "127.0.0.1"
    dp_master_port = get_open_port()
    procs = []
    for i in range(DP_size):
        proc = Process(target=main,
                       args=(DP_size, i, dp_master_ip, dp_master_port,
                             GPUs_per_dp_rank, all_prompts, all_gold_anses))
        proc.start()
        procs.append(proc)
    exit_code = 0
    for proc in procs:
        proc.join()
        if proc.exitcode:
            exit_code = proc.exitcode

    exit(exit_code)