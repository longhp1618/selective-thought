# %%
import os
# os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
# os.environ["WANDB_DISABLED"] = "true"

# %%
# train_dpo.py
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
model = AutoModelForCausalLM.from_pretrained(model_name)
model_ref = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# train_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")


# %%
data_files = {
        'train': "synthesized_data/dpo_epoch0.jsonl"
    }
data = load_dataset("json", data_files=data_files)

# %%
import transformers
from trl import DPOConfig, DPOTrainer

# %%


# %%
parser = transformers.HfArgumentParser(DPOConfig)

# %%
config_path = "configs/draft.yaml"

# %%
from omegaconf import OmegaConf

config = OmegaConf.load(config_path)

# %%
training_args = parser.parse_dict(config)[0]

# %%
training_args

# %%
training_args.optim

# %%
data = data['train']

# %%
dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        train_dataset=data,
        tokenizer=tokenizer,
    )

# %%
dpo_trainer.train()

dpo_trainer.save_model(training_args.output_dir)



