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

def read_math500():
    dataset = load_dataset("HuggingFaceH4/MATH-500")
    test_inputs = dataset.data['test']['problem'].to_pylist()
    test_outputs = dataset.data['test']['solution'].to_pylist()
    return test_inputs, test_outputs

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

def read_SVAMP():
    df = pd.read_json('datasets/SVAMP.json')
    df['problem'] = df['Body'] + " " + df['Question']
    df['cot_output'] = df['Answer'].apply(str)

    svamp_input = df['problem'].tolist()
    svamp_output = df['cot_output'].tolist()

    return svamp_input, svamp_output


def read_ASDIV():
    # Parse the XML file
    tree = ET.parse('datasets/ASDiv.xml')  # Replace 'file.xml' with the path to your XML file
    root = tree.getroot()

    # Extract data
    data = []
    for problem in root.find('ProblemSet'):
        problem_id = problem.attrib.get('ID')
        grade = problem.attrib.get('Grade')
        source = problem.attrib.get('Source')
        body = problem.find('Body').text
        question = problem.find('Question').text
        solution_type = problem.find('Solution-Type').text
        answer = problem.find('Answer').text
        formula = problem.find('Formula').text

        data.append(
            {
                'Problem ID': problem_id,
                'Grade': grade,
                'Source': source,
                'Body': body,
                'Question': question,
                'Solution Type': solution_type,
                'Answer': answer,
                'Formula': formula,
            }
        )

    # Convert to pandas DataFrame
    df = pd.DataFrame(data)
    df = df[df['Answer'].apply(lambda x: str(x).find(';') == -1)]  # remove problems asking multiple answers

    df['problem'] = df['Body'] + " " + df['Question']
    df['cot_output'] = df['Answer'].apply(lambda x: str(x).split()[0])

    asdiv_input = df['problem'].tolist()


    asdiv_output = df['cot_output'].tolist()

    return asdiv_input, asdiv_output