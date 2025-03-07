import torch
import numpy as np
import torch.nn as nn


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps, dtype=np.float64)
    alphas_cumprod0 = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod0 = alphas_cumprod0 / alphas_cumprod0[0]
    betas0 = 1 - (alphas_cumprod0[1:] / alphas_cumprod0[:-1])
    return torch.from_numpy(np.clip(betas0, 0, 0.999)).float()


T = 100  # number of diffusion steps

betas = cosine_beta_schedule(timesteps=T)
alphas = 1 - betas
alphas_prod = torch.cumprod(alphas, 0)
alphas_bar_sqrt = torch.sqrt(alphas_prod)
one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)
alphas_cumprod = torch.cumprod(alphas, 0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - sqrt_alphas_cumprod ** 2)


def extract(inputN, t, x):
    shape = x.shape
    out = torch.gather(inputN, 0, t.to(inputN.device))
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)


# Sampling function
def forward(x_0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_0)
    alphas_t = alphas_bar_sqrt[t]
    alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
    return alphas_t * x_0 + alphas_1_m_t * noise


def plot_diffusion(x_train):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 10, figsize=(28, 3))
    for i in range(10):
        q_i = forward(x_train, torch.tensor([i * 10]))
        axs[i].scatter(q_i[:, 0], q_i[:, 1], s=10)
        axs[i].set_axis_off();
        axs[i].set_title('timestep:' + str(i * 10))


def denoise(model, shape):
    x = torch.randn(shape)
    x_seq = [x]
    for i in range(T - 1, -1, -1):
        idx = torch.tensor([i]).long()
        eps_factor = (1 - alphas[idx]) / (one_minus_alphas_bar_sqrt[idx])
        eps_theta = model(x, idx * torch.ones(x.shape[0]).long())
        mean = torch.sqrt(1 / alphas[idx]) * (x - (eps_factor * eps_theta))
        z = torch.randn_like(x)
        sigma_t = torch.sqrt(betas[idx])
        x = mean + sigma_t * z
        x_seq.append(x)
    return x_seq


# class DenoisingNet(torch.nn.Module):
#     def __init__(self, h, d, complexity=3):
#         super().__init__()
#         self.complexity = complexity
#         self.layers = nn.ModuleList()
#
#         # First layer
#         self.layers.append(nn.Linear(h + 1, d))
#         self.layers.append(nn.LeakyReLU())
#
#         # Intermediate layers based on complexity
#         for _ in range(complexity - 1):
#             self.layers.append(nn.Linear(d + 1, d))
#             self.layers.append(nn.LeakyReLU())
#
#         # Final layer
#         self.layers.append(nn.Linear(d + 1, h))
#
#     def forward(self, x, t):
#         t = t / 100
#         x = torch.cat((x, t.reshape(-1, 1)), 1)
#
#         for i in range(0, len(self.layers) - 1, 2):
#             x = self.layers[i](x)  # Linear layer
#             x = self.layers[i + 1](x)  # Activation function
#             x = torch.cat((x, t.reshape(-1, 1)), 1)
#
#         return self.layers[-1](x)

class TimeEmbedding(nn.Module):
    """Embedding for time step t."""

    def __init__(self, emb_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, emb_dim),
            nn.SiLU(),  # Swish activation
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU()
        )

    def forward(self, t):
        t = t.float().view(-1, 1)  # Ensure (batch_size, 1) and convert to float
        return self.mlp(t)


class ResidualBlock(nn.Module):
    """Residual block with time embedding."""

    def __init__(self, d, emb_dim):
        super().__init__()
        self.fc1 = nn.Linear(d, d * 2)
        self.fc2 = nn.Linear(d * 2, d)
        self.emb_layer = nn.Linear(emb_dim, d * 2)  # To match fc1

        self.activation = nn.SiLU()  # Swish activation

    def forward(self, x, t_emb):
        residual = x  # Store input for residual connection
        h = self.fc1(x)
        h += self.emb_layer(t_emb)  # Add time embedding
        h = self.activation(h)
        h = self.fc2(h)
        return residual + h  # Residual connection


class DenoisingNet(nn.Module):
    """Residual Denoising Network with Time Embedding."""

    def __init__(self, h, d, complexity=3, emb_dim=16):
        super().__init__()
        self.complexity = complexity
        self.time_embedding = TimeEmbedding(emb_dim)

        # Initial projection
        self.input_layer = nn.Linear(h, d)

        # Residual blocks
        self.res_blocks = nn.ModuleList([ResidualBlock(d, emb_dim) for _ in range(complexity)])

        # Output layer
        self.output_layer = nn.Linear(d, h)

    def forward(self, x, t):
        t = t.float()  # Ensure t is a float tensor
        t_emb = self.time_embedding(t)  # Get time embedding

        x = self.input_layer(x)

        for block in self.res_blocks:
            x = block(x, t_emb)  # Residual connection

        return self.output_layer(x)
