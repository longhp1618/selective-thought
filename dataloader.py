import re
import torch
from torch.utils.data import DataLoader, Dataset

def create_ans_rea_mask(text, tokenizer):
    # Tokenize with offset mapping (without special tokens)
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    tokens = encoding['input_ids']
    offset_mapping = encoding['offset_mapping']

    # Initialize mask list with 0's
    rea_mask = [0] * len(tokens)
    ans_mask = [0] * len(tokens)

    # Mark reasoning tokens (mask value 1):
    # Find the position of the "</think>" marker in the text
    think_marker = "</think>"
    think_pos = text.find(think_marker)
    if think_pos != -1:
        for i, (start, end) in enumerate(offset_mapping):
            if end <= think_pos:
                rea_mask[i] = 1

    # Mark answer tokens (mask value 2) only for the final \boxed{...} block:
    # Use re.finditer to find all occurrences, then take the last one.
    boxed_matches = list(re.finditer(r'\\boxed\{(.*?)\}', text, re.DOTALL))
    if boxed_matches:
        final_match = boxed_matches[-1]
        ans_start, ans_end = final_match.span(1)
        for i, (start, end) in enumerate(offset_mapping):
            # Check if token is fully contained within the final answer region
            if start >= ans_start and end <= ans_end:
                ans_mask[i] = 1

    rea_mask = rea_mask[1:] + [0.]
    ans_mask = ans_mask[1:] + [0.]
    
    return tokens, rea_mask, ans_mask

def pad_batch(sequences, pad_value, batch_first=True, padding_side='right'):
    """
    Pad a list of variable-length 1D tensors (sequences) into a 2D tensor.
    This function:
      - If padding_side='right', just calls pad_sequence (which right-pads).
      - If padding_side='left', reverses each sequence, uses pad_sequence,
        then reverses back, effectively giving left-padding in the original orientation.
    """

    if padding_side == 'right':
        # Standard right padding
        return torch.nn.utils.rnn.pad_sequence(sequences, batch_first=batch_first, padding_value=pad_value)
    else:
        # Left padding side = 'left'
        # 1) Reverse each sequence
        reversed_seqs = [torch.flip(seq, dims=[0]) for seq in sequences]

        # 2) Right-pad the reversed sequences
        padded_reversed = torch.nn.utils.rnn.pad_sequence(reversed_seqs, batch_first=batch_first, padding_value=pad_value)

        # 3) Flip back along the correct dimension
        if batch_first:
            # We flip along dim=1 if batch_first
            return torch.flip(padded_reversed, dims=[1])
        else:
            # Otherwise, flip along dim=0
            return torch.flip(padded_reversed, dims=[0])


def collate_fn(batch, MAX_LEN, tokenizer):
    input_ids = [item["input_ids"][:MAX_LEN] for item in batch]
    attention_mask = [item["attention_mask"][:MAX_LEN] for item in batch]
    labels = [item["labels"][:MAX_LEN] for item in batch]

    # Pad sequences to the max_len within this batch
    input_ids = pad_batch(input_ids, batch_first=True, pad_value=tokenizer.pad_token_id, padding_side=tokenizer.padding_side)
    attention_mask = pad_batch(attention_mask, batch_first=True, pad_value=0, padding_side=tokenizer.padding_side)
    labels = pad_batch(labels, batch_first=True, pad_value=-100, padding_side=tokenizer.padding_side)

    if 'ans_mask' not in batch[0].keys():
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "input_texts": [item["input_texts"] for item in batch],
            "label_texts": [item["label_texts"] for item in batch],
        }
    else:
        ans_mask = [item["ans_mask"][:MAX_LEN] for item in batch]
        ans_mask = pad_batch(ans_mask, batch_first=True, pad_value=0, padding_side=tokenizer.padding_side)
        ans_mask = ans_mask[:, :MAX_LEN]
        rea_mask = [item["rea_mask"][:MAX_LEN] for item in batch]
        rea_mask = pad_batch(rea_mask, batch_first=True, pad_value=0, padding_side=tokenizer.padding_side)
        rea_mask = rea_mask[:, :MAX_LEN]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "ans_mask": ans_mask,
            "rea_mask": rea_mask,
            "instruction": [item["instruction"] for item in batch],
            "completion": [item["completion"] for item in batch],
        }

class DecoderDataset(Dataset):
    def __init__(self, instructions, completions, tokenizer, MAX_LEN):#, split_ans=False, test=False):
        self.instructions = instructions
        self.completions = completions
        self.tokenizer = tokenizer
        self.MAX_LEN = MAX_LEN
        # self.split_ans = split_ans
        # self.test = test
        # assert not (split_ans and test), "we don't need ans_mask, rea_mask for testing"

    def __len__(self):
        return len(self.instructions)

    def __getitem__(self, idx):
        instruction = self.instructions[idx]
        completion = self.completions[idx]

        instruction_tokens = self.tokenizer.apply_chat_template(
                [
        # {"role": "system", "content": ""},
        {"role": "user", "content": f"Please reason step by step, and put your final answer within \\boxed{{}}. {prompt}"},
    ],
                    tokenize=True,
                    add_generation_prompt=True,
                )

        completion_tokens, completion_rea_mask, completion_ans_mask = create_ans_rea_mask(completion, self.tokenizer)
        
        rea_mask = [0]*len(instruction_tokens) + completion_rea_mask
        ans_mask = [0]*len(instruction_tokens) + completion_ans_mask

        input_ids = instruction_tokens + completion_tokens
        # we don't optimize Cross-Entropy loss for instruction tokens
        labels = [-100]*len(instruction_tokens) + completion_tokens

        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(labels)
        ans_mask = torch.tensor(ans_mask)
        rea_mask = torch.tensor(rea_mask)

        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": labels,
            "ans_mask": ans_mask,
            "rea_mask": rea_mask,
            "instruction": instruction,
            "completion": completion,
        }
    
def create_dataloader(instructions, completions, tokenizer, bs, MAX_LEN, shuffle=True): #, split_ans=False, decoder=False, test=True):
    dataset = DecoderDataset(instructions, completions, tokenizer, MAX_LEN)
    dataloader = DataLoader(
        dataset,
        batch_size=bs,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_fn(batch, MAX_LEN, tokenizer),
    )

    return dataloader