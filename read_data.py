import json
import os
import re
import xml.etree.ElementTree as ET

import pandas as pd

from datasets import load_dataset


def read_gsm8k(train=False):
    dataset = load_dataset("gsm8k", 'socratic')
    if train:
        gsm8k_input = dataset.data['train']['question'].to_pylist()
        gsm8k_output = dataset.data['train']['answer'].to_pylist()
    else:
        gsm8k_input = dataset.data['test']['question'].to_pylist()
        gsm8k_output = dataset.data['test']['answer'].to_pylist()

    return gsm8k_input, gsm8k_output
