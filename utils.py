import os
import random
import shutil

import numpy as np
import torch
from transformers import set_seed as hg_seed


def set_seed(seed):
    """for reproducibility
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)
    hg_seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_directories(task, method, model_gen, size, seed, train=True):
    # set up save name, and save directories for models with different task, sizess, seeds
    name = f"{method}_{model_gen}_{size}_{seed}"
    print(name, task)
    save_dir = f"./saved_models/{task}/{name}"
    save_csv_dir = f"./csv/{task}/{name}"
    save_txt_dir = f"./txt/{task}/{name}.txt"
    if train:
        if os.path.exists(save_dir):
            # rm = input(f"{save_dir} exists. Do you want to remove this directory? (y/n):")
            rm = "y"
            if rm == "y":
                # Remove the folder if it already exists
                shutil.rmtree(save_dir)
                shutil.rmtree(save_csv_dir)
            else:
                print("Okay. See you then")
                exit()
        os.makedirs(save_dir)
        os.makedirs(save_csv_dir)
        if not os.path.exists(f"./txt/{task}"):
            os.makedirs(f"./txt/{task}")

    return name, save_dir, save_csv_dir, save_txt_dir
