import torch
from torch import nn
from torch import Tensor
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import List
import glob
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

print(torch.__version__)
print(torch.cuda.device_count())
device = torch.device('cuda')


# define a VAE model
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


# VAE loss
def loss_function(VAELossParams, kld_weight):
    recons, input, mu, log_var = VAELossParams

    recons_loss = F.mse_loss(recons, input)

    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0)

    loss = recons_loss + kld_weight * kld_loss
    return {"loss": loss,
            "Reconstruction_Loss": recons_loss.detach(),
            "KLD": -kld_loss.detach()}


# Sepia dataset
class CuttlefishDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.all_images = list(glob.iglob(root_dir + "/*.png"))  # obtain the path of all images

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        img_path = self.all_images[idx]
        image = Image.open(img_path).convert("L")

        if self.transform:
            image = self.transform(image)

        return image


root_dir = '/home/wbsong/Camouflage/Data/3_backgrounds/sepia213/pattern_dataset'
data_transforms = transforms.Compose([transforms.CenterCrop((512, 256)),
                                      transforms.Resize((128, 64)),
                                      transforms.ToTensor()])
training_dataset = CuttlefishDataset(root_dir, data_transforms)
training_dataloader = DataLoader(training_dataset, batch_size=32, shuffle=True)


# instantiate VAE model
in_channels = 1  # channels of input images
latent_dim = 256  # dimension of latent distribution of x
model = VanillaVAE(in_channels, latent_dim)
model.to(device)

# instantiate optimizer and scheduler
KLD_weight = 0.00025
learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)  # learning rate decay

# training
epochs = 150
training_loss = []
print('Training Start!')
for epoch in range(epochs):
    loss_epoch = 0
    for i, x in enumerate(training_dataloader):
        x = x.to(device)
        predictions = model(x)

        loss_batch = loss_function(predictions, KLD_weight)

        optimizer.zero_grad()
        loss_batch['loss'].backward()
        optimizer.step()

        loss_epoch += loss_batch['loss'].item()

    training_loss.append(loss_epoch / len(training_dataloader))
    scheduler.step()

training_loss = np.array(training_loss)
plt.figure(dpi=300)
plt.plot(range(epochs), training_loss)
plt.show()


# Validation
# reconstruction
idx = np.random.randint(178)
#idx = 0
all_images = list(glob.iglob(root_dir + "/*.png"))
img_path = all_images[idx]
image = Image.open(img_path).convert("L")
input = data_transforms(image).unsqueeze(0)
input = input.to(device)

model.eval()
with torch.no_grad():
    predictions = model(input)
    output = predictions[0]
    output = transforms.ToPILImage()(output.squeeze(0))
    input = transforms.ToPILImage()(input.squeeze(0))

    fig, axs = plt.subplots(1, 2, dpi=300)
    axs[0].imshow(input, cmap='gray')
    axs[1].imshow(output, cmap='gray')
    plt.show()

# sampling
model.eval()
with torch.no_grad():
    sample = model.sample(device)
    sample_img = transforms.ToPILImage()(sample.squeeze(0))

    plt.figure(dpi=300)
    plt.imshow(sample_img, cmap='gray')
    plt.show()

# save
torch.save(model.state_dict(), '/home/wbsong/Camouflage/Script/Generative model/VAE_weights.pth')