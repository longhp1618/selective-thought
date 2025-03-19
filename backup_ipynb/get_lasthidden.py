# %%
from tqdm import tqdm
import torch

# %%
from transformers import AutoTokenizer, AutoModelForCausalLM
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# %%
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 
                    torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="cuda:7")

# %%
outputs = torch.load("saved_outputs/test_DeepSeek-R1-Distill-Qwen-1.5B.pt")

# %%
instructions = [output['question'] for output in outputs]
completions = [output['pred_text'] for output in outputs]

# %%
from dataloader import create_dataloader

# %%
bs = 1
MAX_LEN = 16384
dataloader = create_dataloader(instructions, completions, tokenizer, bs, MAX_LEN, shuffle=False)

# %%
model.eval();

# %%
device = model.device

# %%
all_rea_hiddens, all_ans_hiddens = None, None

# %%
@torch.no_grad()
def get_hidden(batch, model):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    ans_mask = batch['ans_mask'].to(device)
    rea_mask = batch['rea_mask'].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)
    last_hidden = outputs.hidden_states[-1]
    hidden_size = model.config.hidden_size
    rea_hidden = last_hidden * rea_mask.unsqueeze(-1)
    rea_hidden = rea_hidden.reshape(-1, hidden_size)
    non_zero_rows_mask_rea = torch.any(rea_hidden != 0, dim=1)
    rea_hidden = rea_hidden[non_zero_rows_mask_rea]

    ans_hidden = last_hidden * ans_mask.unsqueeze(-1)
    ans_hidden = ans_hidden.reshape(-1, hidden_size)
    non_zero_rows_mask_ans = torch.any(ans_hidden != 0, dim=1)
    ans_hidden = ans_hidden[non_zero_rows_mask_ans]
    return rea_hidden, ans_hidden

# %%
batch_iter = tqdm(dataloader, desc='Training', position=0, leave=True)

# %%
for batch in batch_iter:
    rea_hidden, ans_hidden = get_hidden(batch, model)
    if all_ans_hiddens is None:
        all_ans_hiddens = ans_hidden
    else:
        all_ans_hiddens = torch.cat([all_ans_hiddens, ans_hidden])

    if all_rea_hiddens is None:
        all_rea_hiddens = rea_hidden
    else:
        all_rea_hiddens = torch.cat([all_rea_hiddens, rea_hidden])

# %%
all_rea_hiddens.size()

# %%
from torch import nn
from torch.utils.data import DataLoader

# %%
model.config.attention_dropout

# %%
tokenizer.model_max_length

# %%
all_ans_hiddens = all_ans_hiddens.to(torch.float32)
all_rea_hiddens = all_rea_hiddens.to(torch.float32)

# %%
model_name = "DeepSeek-R1-Distill-Qwen-1.5B"

# %%
torch.save(all_rea_hiddens, f"saved_hiddens/all_rea_hiddens_{model_name}.pt")
torch.save(all_ans_hiddens, f"saved_hiddens/all_ans_hiddens_{model_name}.pt")

# %%
class CustomLinear(nn.Module):
    def __init__(self, input_dim):
        super(CustomLinear, self).__init__()
        self.input_dim = input_dim
        # Initialize the weight matrix
        self.weight = nn.Parameter(torch.empty(input_dim, input_dim))
        nn.init.xavier_uniform_(self.weight) 
        self.dropout = nn.Dropout(0.05)
        # self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x):
        # Create a mask to zero out the diagonal
        mask = torch.ones_like(self.weight, device=x.device) - torch.eye(self.input_dim, device=x.device)
        masked_weight = self.weight * mask
        masked_weight = self.dropout(masked_weight)
        return x @ masked_weight.T #+ self.bias


# %%
bs_interpolator = 256

# %%
train_loader = DataLoader(all_ans_hiddens, batch_size=bs_interpolator, shuffle=True)

# %%
rea_loader = DataLoader(all_rea_hiddens, batch_size=bs_interpolator*10, shuffle=True)

# %%
all_ans_hiddens.shape, all_rea_hiddens.shape

# %%
from torch import optim

# %%
input_dim = model.config.hidden_size
interpolator = CustomLinear(input_dim)
interpolator = interpolator#.to(dtype=torch.bfloat16)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=5e-4)
interpolator = interpolator.to(device)

# %%
num_epochs = 30
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
        
        loss_total = (loss.item() + loss_total*step)/(step+1)
        batch_iter.set_description(f"Loss {loss_total:.3f}")

# %%
interpolator.eval();
loss_total = 0
errors = []

batch_iter = tqdm(rea_loader)
for step, x in enumerate(batch_iter):
    x = x.to(device)
    optimizer.zero_grad()  # Clear gradients
    y = interpolator(x)  # Forward pass
    loss = criterion(y, x)  # Compute loss
    
    loss_total = (loss.item() + loss_total*step)/(step+1)
    batch_iter.set_description(f"Loss {loss_total:.3f}")

    errors += (y-x).square().mean(dim=1).tolist()

# %%
print(sum(errors)/len(errors))

# %%
import numpy as np

# %%
quantiles = np.percentile(errors, [25, 50, 75])

print("25th percentile:", quantiles[0])
print("50th percentile (median):", quantiles[1])
print("75th percentile:", quantiles[2])

# %%


# %%
# Compute quantiles
q0, q25, q50, q75, q100 = np.percentile(errors, [0, 25, 50, 75, 100])

# Filter values for each range
range_0_to_50 = [x for x in errors if q0 <= x <= q50]
range_25_to_75 = [x for x in errors if q25 <= x <= q75]
range_50_to_100 = [x for x in errors if q50 <= x <= q100]

# Compute variance
var_0_to_50 = np.var(range_0_to_50, ddof=1)  # ddof=1 for sample variance
var_25_to_75 = np.var(range_25_to_75, ddof=1)
var_50_to_100 = np.var(range_50_to_100, ddof=1)

# %%
# Print results
print("q25, q50, q75", q25, q50, q75)
print("Variance from 0th to 50th percentile:", var_0_to_50)
print("Variance from 25th to 75th percentile:", var_25_to_75)
print("Variance from 50th to 100th percentile:", var_50_to_100)

# %%
K = 3
errors = torch.tensor(errors, device=device, requires_grad=False)
means = torch.tensor([q25, q50, q75], device=device, requires_grad=False)
variances = torch.tensor([var_0_to_50, var_25_to_75, var_50_to_100], device=device, requires_grad=False)
mixing_coeffs = torch.ones(K, device=device, requires_grad=False) / K

# %%
# K = 2
# errors = torch.tensor(errors, device=device, requires_grad=False)
# means = torch.tensor([q25, q75], device=device, requires_grad=False)
# variances = torch.tensor([var_0_to_50, var_50_to_100], device=device, requires_grad=False)
# mixing_coeffs = torch.ones(K, device=device, requires_grad=False) / K

# %%
# EM Algorithm Parameters
max_iterations = 100000
threshold = 1e-8

# Responsibility matrix (E-step)
responsibilities = torch.zeros((len(errors), K), device=device, requires_grad=False)

# %%
def gaussian_pdf(x, mean, variance):
    """Gaussian probability density function."""
    return (1 / (torch.sqrt(2 * torch.pi * variance))) * torch.exp(-0.5 * ((x - mean) ** 2) / (variance)) + 1e-8

# EM Algorithm
for iteration in range(max_iterations):
    # E-step: Compute responsibilities
    for k in range(K):
        responsibilities[:, k] = mixing_coeffs[k] * gaussian_pdf(errors, means[k], variances[k])
    responsibilities /= responsibilities.sum(dim=1, keepdim=True)

    # M-step: Update parameters
    N_k = responsibilities.sum(dim=0)  # Effective number of points per cluster

    # Update means
    new_means = (responsibilities * errors[:, None]).sum(dim=0) / N_k

    # Update variances
    new_variances = (responsibilities * (errors[:, None] - new_means) ** 2).sum(dim=0) / N_k

    # Update mixing coefficients
    new_mixing_coeffs = N_k / len(errors)

    # Check for convergence
    if (torch.abs(new_means - means).max() < threshold and
        torch.abs(new_variances - variances).max() < threshold and
        torch.abs(new_mixing_coeffs - mixing_coeffs).max() < threshold):
        print(f"Converged at iteration {iteration}.")
        break

    means, variances, mixing_coeffs = new_means, new_variances, new_mixing_coeffs

# Print final parameters
print("Final Means:", means.cpu().numpy())
print("Final Variances:", variances.cpu().numpy())
print("Final Mixing Coefficients:", mixing_coeffs.cpu().numpy())

# %%
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math

for k in range(K):
    mu = means[k].detach().cpu().numpy()
    variance = variances[k].detach().cpu().numpy()
    sigma = math.sqrt(variance)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    if mu == means.min():
        plt.plot(x, stats.norm.pdf(x, mu, sigma), color='red')
    else:
        plt.plot(x, stats.norm.pdf(x, mu, sigma))
plt.show()

# %%
def e_step(data_point):
    probabilities = []
    for k in range(K):
        probabilities.append(mixing_coeffs[k] * gaussian_pdf(data_point, means[k], variances[k]))
    probabilities = torch.stack(probabilities)
    probabilities /= probabilities.sum()
    return probabilities

# %%
idx = 552

# %%
dataloader_1 = create_dataloader(instructions[idx:idx+1], completions[idx:idx+1], tokenizer, 1, MAX_LEN, shuffle=False)

# %%
for batch in dataloader_1:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    ans_mask = batch['ans_mask'].to(device)
    rea_mask = batch['rea_mask'].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)
    last_hidden = outputs.hidden_states[-1]
    break

# %%
last_hidden = last_hidden.squeeze(0)
labels = labels.squeeze(0)

# %%
last_hidden = last_hidden.to(torch.float32)
# labels = labels.to(torch.float32)

# %%
length = (labels==-100).sum()
last_hidden = last_hidden[length:]
labels = labels[length:]

# %%
for x, token in zip(last_hidden, labels):
    x = x.unsqueeze(0)
    x = x.to(device)
    y = interpolator(x)  # Forward pass
    # loss = criterion(y, x)  # Compute loss
    error = (y-x).square().mean().item()
    if e_step(error).argmax() == means.argmin():
        print(tokenizer.decode(token), error, e_step(error), "*")
    elif e_step(error).argmax() == means.argmax():
        print(tokenizer.decode(token), error, e_step(error), "###")
    else:
        print(tokenizer.decode(token), error, e_step(error))

# %%
from dataloader import create_ans_rea_mask

# %%
print(instructions[0] + "\n****" + completions[0])

# %%



