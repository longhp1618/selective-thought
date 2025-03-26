from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import transformers
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from data_module import GSM8KForLLAMA
import torch
import time

drop_rate = 1

config_path = "configs/sft_rmall.yaml"
parser = transformers.HfArgumentParser(TrainingArguments)

cfg = OmegaConf.load(config_path)

trainer_args_dict = OmegaConf.to_container(cfg.trainer)
training_args = parser.parse_dict(trainer_args_dict)[0]

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

gsm8k_module = GSM8KForLLAMA(cfg.data, tokenizer, drop_rate)


trainer = Trainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args, 
        train_dataset=gsm8k_module.train_dataset,
        data_collator=gsm8k_module.data_collator,
    )


trainer.train()
trainer.save_model(training_args.output_dir)
