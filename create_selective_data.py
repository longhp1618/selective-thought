import argparse
import os

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch
from train_em import CustomLinear

# python create_selective_data.py --task gsm8k --model_gen llama --size 7B --device 7
def arguments():
    parser = argparse.ArgumentParser(description="Training setting details")
    parser.add_argument('--task', type=str, choices=["math", "gsm8k"], default='gsm8k', help='Training Dataset')
    # parser.add_argument('--model_gen', type=str, choices=["t5", "llama", "mistral"], default='mistral', help='Model family')
    # parser.add_argument('--size', type=str, choices=["small", "base", "large", '7B', '13B'], default='7B', help='Model size')
    parser.add_argument('--device', type=int, default=0, help='GPU id')

    args = parser.parse_args()
    args.model_gen = "r1"
    args.size = "1.5B"

    return args

def prepare(args):
    save_name = f"{args.model_gen}_{args.size}"
    task = args.task
    device = torch.device(f"cuda:{args.device}")
    save_path = f"saved_hiddens/{task}/{save_name}/train"
    load_path = f"saved_em_components/{task}/{save_name}"

    data = torch.load(f"{save_path}/hidden_data.pt")
    all_instruction_texts, all_short_cot_texts, all_rea_tokens, _, all_rea_hiddens, _ = data

    hidden_dim = all_rea_hiddens[0].shape[1]
    
    interpolator = CustomLinear(hidden_dim)
    interpolator.load_state_dict(torch.load(f"{load_path}/interpolator.pt", map_location='cpu'))
    interpolator = interpolator.to(device)
    interpolator.eval();
    means, variances, mixing_coeffs = torch.load(f"{load_path}/em.pt", map_location=device)

    return all_instruction_texts, all_short_cot_texts, all_rea_tokens, all_rea_hiddens, interpolator, means, variances, mixing_coeffs, device, save_path


@torch.no_grad()
def interpolation(interpolator, x, device):
    interpolator.eval()
    x = x.to(torch.float).to(device)
    y = interpolator(x)
    errors = (y - x).square().mean(dim=1)
    return errors.detach()

def e_step(data_point, means, variances, mixing_coeffs):
    def gaussian_pdf(x, mean, variance):
        """Gaussian probability density function."""
        return (1 / (torch.sqrt(2 * torch.pi * variance + 1e-8))) * torch.exp(-0.5 * ((x - mean) ** 2) / (variance + 1e-8))
    K = 2
    probabilities = []
    for k in range(K):
        probabilities.append(mixing_coeffs[k] * gaussian_pdf(data_point, means[k], variances[k]))
    probabilities = torch.stack(probabilities)
    probabilities /= probabilities.sum()
    return probabilities

def selective_data(interpolator, all_instruction_texts, all_short_cot_texts, all_rea_tokens, all_hiddens, device, save_path, means, variances, mixing_coeffs):
    errors = [interpolation(interpolator, hiddens, device) for hiddens in all_hiddens]
    masks = [((e_step(error, means, variances, mixing_coeffs).argmax(dim=0) + (error == 0).to(torch.int)) == 0).to(torch.int).cpu() for error in errors]
    errors = [error.detach().cpu() for error in errors]
    dct = {'instruction_texts': all_instruction_texts, 'short_cot_texts': all_short_cot_texts, 'rea_tokens': all_rea_tokens, 'seletive_masks': masks, "interpolation_error": errors}
    torch.save(dct, f"{save_path}/selective_data.pt")

    return dct

if __name__ == "__main__":
    args = arguments()
    print("Preparing")
    all_instruction_texts, all_short_cot_texts, all_rea_tokens, all_rea_hiddens, interpolator, means, variances, mixing_coeffs, device, save_path = prepare(args)
    print("Processing")
    data = selective_data(interpolator, all_instruction_texts, all_short_cot_texts, all_rea_tokens, all_rea_hiddens, device, save_path, means, variances, mixing_coeffs)