import json
import os
import re
import xml.etree.ElementTree as ET

import pandas as pd

from datasets import load_dataset
from math_evaluation.parser import extract_answer

def read_gsm8k(train=False):
    dataset = load_dataset("gsm8k", 'socratic')
    if train:
        gsm8k_input = dataset.data['train']['question'].to_pylist()
        gsm8k_output = dataset.data['train']['answer'].to_pylist()
    else:
        gsm8k_input = dataset.data['test']['question'].to_pylist()
        gsm8k_output = dataset.data['test']['answer'].to_pylist()
    gsm8k_output = [gold_seq[gold_seq.find("####"):].replace("####", "").strip() for gold_seq in gsm8k_output]
    return gsm8k_input, gsm8k_output

def read_deepscaler():
    dataset = load_dataset("agentica-org/DeepScaleR-Preview-Dataset", split='train')
    return dataset['problem'], dataset['answer']

def read_MATH(train=False):
    data = {"problem": [], 'level': [], 'type': [], 'solution': []}
    if train:
        name = 'train'
    else:
        name = 'test'
    categories = os.listdir(f"datasets/MATH/{name}/")
    for category in categories:
        if category == '.DS_Store':
            continue
        for idx in os.listdir(os.path.join(f"datasets/MATH/{name}", category)):
            path = os.path.join(f"datasets/MATH/{name}", category, idx)

            with open(path) as f:
                sample = json.load(f)
                for key in data.keys():
                    data[key].append(sample[key])
    df = pd.DataFrame(data)

    df['answer'] = df['solution'].apply(lambda x: extract_answer(x, 'math', False))

    test_input = df['problem'].tolist()
    test_output = df['answer'].tolist()



    return test_input, test_output