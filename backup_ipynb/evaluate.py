# %%
import torch

# %%
from math_evaluation.parser import extract_answer
from math_evaluation.grader import math_equal

# %%


# %%
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
model_name = model_name[model_name.find('/')+1:]

# %%
outputs = torch.load(f"saved_outputs/no_think_test_{model_name}.pt")

# %%
questions = [output['question'] for output in outputs]
pred_texts = [output['pred_text'] for output in outputs]
label_texts = [output['label_text'] for output in outputs]

# %%
outputs[0].keys()

# %%
tokens = [output['pred_tokens'] for output in outputs]

# %%
lens = [len(token) for token in tokens]

# %%
max(lens)

# %%
import matplotlib.pyplot as plt

# %%
import pandas as pd

# %%
df = pd.DataFrame({"question": questions, 'pred_text': pred_texts, 'label_text': label_texts})

# %%
df['pred'] = df['pred_text'].apply(lambda x: extract_answer(x, 'gsm8k', False))

# %%
df['label'] = df['label_text'].apply(lambda x: x[x.find("####"):].replace("####", "").strip())

# %%
preds = df['pred'].tolist()
labels = df['label'].tolist()

# %%
track_lst = [math_equal(pred, label) for pred, label in zip(preds, labels)]

# %%
track_lst = []
for pred, label in zip(preds, labels):
    track_lst.append(math_equal(pred, label))

# %%
df

# %%
df['pred_text'][3]

# %%
sum(track_lst)/len(track_lst)

# %%



