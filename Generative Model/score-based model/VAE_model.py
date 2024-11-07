import torch
from torch import nn
from torch import Tensor
from typing import List

class VanillaVAE(nn.Module):

    def __init__(self, in_channels: int, latent_dim: int, hidden_dims: List = None,) -> None:
        super().__init__()

        self.latent_dim = latent_dim

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # build Encoder
        modules = []
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(
                           nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1),  # (h, w) -> (h/2, w/2)
                           nn.BatchNorm2d(h_dim),
                           nn.LeakyReLU()))
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        # define the distribution
        self.fc_mu = nn.Linear(hidden_dims[-1]*8, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*8, latent_dim)

        # build Decoder
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 8)

        hidden_dims.reverse()  # reverse the list
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Sequential(
                           nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1),
                           nn.BatchNorm2d(hidden_dims[i + 1]),
                           nn.LeakyReLU()))

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                           nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1),  # return to the original size
                           nn.BatchNorm2d(hidden_dims[-1]),
                           nn.LeakyReLU(),
                           nn.Conv2d(hidden_dims[-1], out_channels=1, kernel_size=3, padding=1), nn.Tanh())  # return to the original channels

    # forward function
    def encode(self, input: Tensor) -> List[Tensor]:  # input_size -> (batch_size, 1, 128, 64)
        result = self.encoder(input)  # size : (batch_size, num_channels, h, w) -> (batch_size, 512, 4, 2)
        result = torch.flatten(result, start_dim=1)  # (batch_size, 512*4*2)

        # Split the result into mu and var components of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, 512, 4, 2)

        result = self.decoder(result)  # (batch_size, 32, 64, 32)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu  # sample from the latent distribution

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    # sample from a normal distribution
    def sample(self, device):
        z = torch.randn(1, self.latent_dim)
        z = z.to(device)
        sample = self.decode(z)
        return sample
