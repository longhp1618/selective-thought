import argparse
import os

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import set_seed


class CustomLinear(nn.Module):
    def __init__(self, input_dim, drop_rate=0.2):
        super(CustomLinear, self).__init__()
        self.input_dim = input_dim
        # Initialize the weight matrix
        self.weight = nn.Parameter(torch.empty(input_dim, input_dim))
        nn.init.xavier_uniform_(self.weight)
        self.dropout = nn.Dropout(drop_rate)
        # self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x):
        # Create a mask to zero out the diagonal
        mask = torch.ones_like(self.weight, device=x.device) - torch.eye(self.input_dim, device=x.device)
        masked_weight = self.weight * mask
        masked_weight = self.dropout(masked_weight)
        return x @ masked_weight.T  # + self.bias


# python train_em.py --task gsm8k --model_gen --mistral --size 7B --seed 0 --device 0 --num_epochs 20 --bs_interpolator 256 --lr 5e-4
def arguments():
    parser = argparse.ArgumentParser(description="Training setting details")
    parser.add_argument('--task', type=str, choices=["math", "gsm8k"], default='gsm8k', help='Training Dataset')
    # parser.add_argument('--model_gen', type=str, choices=["t5", "llama", "mistral"], default='mistral', help='Model family')
    # parser.add_argument('--size', type=str, choices=["small", "base", "large", '7B', '13B'], default='7B', help='Model size')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--device', type=int, default=0, help='GPU id')
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of Training Epochs")
    parser.add_argument("--bs_interpolator", type=int, default=256, help="batch size for training intepolator")
    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
    parser.add_argument("--drop_rate", type=float, default=0.2, help="Drop rate for Interpolator")

    args = parser.parse_args()
    args.model_gen = "r1"
    args.size = "1.5B"
    set_seed(args.seed)
    return args


def prepare(args):
    save_name = f"{args.model_gen}_{args.size}"
    task = args.task
    device = torch.device(f"cuda:{args.device}")
    num_epochs = args.num_epochs
    bs_interpolator = args.bs_interpolator
    lr = args.lr

    save_path = f"saved_em_components/{task}/{save_name}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # all_rea_hiddens = torch.load(f"saved_hiddens/{task}/{save_name}/train/all_rea_hiddens.pt", map_location='cpu').to(torch.float32)
    # all_ans_hiddens = torch.load(f"saved_hiddens/{task}/{save_name}/train/all_ans_hiddens.pt", map_location='cpu').to(torch.float32)
    _, _, _, _, all_rea_hiddens, all_ans_hiddens = torch.load(f"{save_path}/hidden_data.pt")
    all_rea_hiddens = torch.cat(all_rea_hiddens).to(torch.float32)
    all_ans_hiddens = torch.cat(all_ans_hiddens).to(torch.float32)

    ans_loader = DataLoader(all_ans_hiddens, batch_size=bs_interpolator, shuffle=True)
    rea_loader = DataLoader(all_rea_hiddens, batch_size=bs_interpolator * 100, shuffle=True)

    print("Answer hidden shape:", all_ans_hiddens.shape)
    print("Reasoning hidden shape:", all_rea_hiddens.shape)
    input_dim = all_ans_hiddens.shape[1]
    interpolator = CustomLinear(input_dim, args.drop_rate)

    interpolator = interpolator.to(device)
    optimizer = optim.Adam(interpolator.parameters(), lr=lr)

    return ans_loader, rea_loader, interpolator, optimizer, save_name, task, device, num_epochs, save_path


def train_interpolator(interpolator, optimizer, device, num_epochs, train_loader, save_path):
    print("Training Interpolator")
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        interpolator.train()
        loss_total = 0
        batch_iter = tqdm(train_loader)
        for step, x in enumerate(batch_iter):
            x = x.to(device)
            optimizer.zero_grad()  # Clear gradients
            y = interpolator(x)  # Forward pass
            loss = criterion(y, x)  # Compute loss

            loss.backward()  # Backpropagate gradients
            optimizer.step()  # Update parameters

            loss_total = (loss.item() + loss_total * step) / (step + 1)
            batch_iter.set_description(f"Interpolator Epoch {epoch} Loss {loss_total:.3f}")

    torch.save(interpolator.state_dict(), f"{save_path}/interpolator.pt")
    return interpolator, optimizer


@torch.no_grad()
def get_interpolation_error(interpolator, loader):
    interpolator.eval()
    criterion = nn.MSELoss()

    loss_total = 0
    errors = []

    batch_iter = tqdm(loader)
    for step, x in enumerate(batch_iter):
        x = x.to(device)
        y = interpolator(x)  # Forward pass
        loss = criterion(y, x)  # Compute loss

        loss_total = (loss.item() + loss_total * step) / (step + 1)
        batch_iter.set_description(f"Loss {loss_total:.3f}")

        errors += (y - x).square().mean(dim=1).tolist()

    return errors


def em_initialization(errors):
    # Compute quantiles
    q0, q25, q50, q75, q100 = np.percentile(errors, [0, 25, 50, 75, 100])

    # Filter values for each range
    range_0_to_50 = [x for x in errors if q0 <= x <= q50]
    # range_25_to_75 = [x for x in errors if q25 <= x <= q75]
    range_50_to_100 = [x for x in errors if q50 <= x <= q100]

    # Compute variance
    var_0_to_50 = np.var(range_0_to_50, ddof=1)  # ddof=1 for sample variance
    # var_25_to_75 = np.var(range_25_to_75, ddof=1)
    var_50_to_100 = np.var(range_50_to_100, ddof=1)

    print("q25, q50, q75", q25, q50, q75)
    print("Variance from 0th to 50th percentile:", var_0_to_50)
    # print("Variance from 25th to 75th percentile:", var_25_to_75)
    print("Variance from 50th to 100th percentile:", var_50_to_100)

    return q25, q75, var_0_to_50, var_50_to_100


def em(errors_rea, errors_ans, device, save_path):
    def gaussian_pdf(x, mean, variance):
        """Gaussian probability density function."""
        return (1 / (torch.sqrt(2 * torch.pi * variance + 1e-8))) * torch.exp(-0.5 * ((x - mean) ** 2) / (variance + 1e-8))

    print("Performing EM")

    n_rea = len(errors_rea)
    n_ans = len(errors_ans)
    errors = errors_rea + errors_ans
    q25, q75, var_0_to_50, var_50_to_100 = em_initialization(errors)
    errors= torch.tensor(errors, device=device, requires_grad=False)

    K = 2
    means = torch.tensor([q25, q75], device=device, requires_grad=False)
    variances = torch.tensor([var_0_to_50, var_50_to_100], device=device, requires_grad=False)
    mixing_coeffs = torch.ones(K, device=device, requires_grad=False) / K

    # EM Algorithm Parameters
    max_iterations = 10000
    threshold = 1e-8

    # Responsibility matrix (E-step)
    responsibilities = torch.zeros((len(errors), K), device=device, requires_grad=False)

    # EM Algorithm
    for iteration in tqdm(range(max_iterations)):
        # E-step: Compute responsibilities
        for k in range(K):
            responsibilities[:, k] = mixing_coeffs[k] * gaussian_pdf(errors, means[k], variances[k])
        responsibilities /= responsibilities.sum(dim=1, keepdim=True)

        # Hard assignment: Force errors_ans to cluster 0
        responsibilities[n_rea:, 0] = 1.0
        responsibilities[n_rea:, 1] = 0.0  # Cluster 1 responsibility is 0
        
        # M-step: Update parameters
        N_k = responsibilities.sum(dim=0)  # Effective number of points per cluster

        # Update means
        new_means = (responsibilities * errors[:, None]).sum(dim=0) / N_k

        # Update variances
        new_variances = (responsibilities * (errors[:, None] - new_means) ** 2).sum(dim=0) / N_k

        # Update mixing coefficients
        new_mixing_coeffs = N_k / len(errors)
        # Clamp cluster 0 mixing coefficient
        # if new_mixing_coeffs[0] > 0.4:
        #     excess = new_mixing_coeffs[0] - 0.4
        #     new_mixing_coeffs[0] = 0.4
        #     new_mixing_coeffs[1] += excess  # distribute the rest to cluster 1

        # Check for convergence
        if (
            torch.abs(new_means - means).max() < threshold
            and torch.abs(new_variances - variances).max() < threshold
            and torch.abs(new_mixing_coeffs - mixing_coeffs).max() < threshold
        ):
            print(f"Converged at iteration {iteration}.")
            break

        means, variances, mixing_coeffs = new_means, new_variances, new_mixing_coeffs

    # Print final parameters
    print("Final Means:", means.cpu().numpy())
    print("Final Variances:", variances.cpu().numpy())
    print("Final Mixing Coefficients:", mixing_coeffs.cpu().numpy())

    torch.save((means, variances, mixing_coeffs), f"{save_path}/em.pt")
    return means, variances, mixing_coeffs

if __name__ == "__main__":
    args = arguments()
    ans_loader, rea_loader, interpolator, optimizer, save_name, task, device, num_epochs, save_path = prepare(args)

    interpolator, optimizer = train_interpolator(interpolator, optimizer, device, num_epochs, ans_loader, save_path)

    print("Getting reasoning interpolation errors")
    errors_rea = get_interpolation_error(interpolator, rea_loader)
    print("Getting answering interpolation errors")
    errors_ans = get_interpolation_error(interpolator, ans_loader)

    means, variances, mixing_coeffs = em(errors_rea, errors_ans, device, save_path)
