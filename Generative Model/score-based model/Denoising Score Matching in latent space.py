import torch
from torch import nn
import sys
sys.path.append("/home/wbsong/Camouflage/Script/Generative model")
from VAE_model import VanillaVAE
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Device
device = torch.device('cuda')

"""  Latent Space - VAE """
# instantiate VAE model
in_channels = 1  # channels of input images
latent_dim = 256  # dimension of latent distribution of x
VAE_model = VanillaVAE(in_channels, latent_dim)
VAE_model.load_state_dict(torch.load('/home/wbsong/Camouflage/Script/Generative model/VAE_weights.pth'))
VAE_model.eval()
VAE_model.to(device)


# decompose VAE for convinient
def Encoding(model, x):

    mu, logvar = model.encode(x)
    z = model.reparameterize(mu, logvar)

    return z


def Decoding(model, z):

    x_reco = model.decode(z)

    return x_reco


""" Dataset """
# image dataset
class pattern_dataset(Dataset):

    def __init__(self, root_dir, label, transform=None):
        self.root_dir = root_dir
        self.label = torch.tensor(label)
        self.transform = transform
        self.all_images = list(glob.iglob(root_dir + "/*.png"))  # obtain the path of all images

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        img_path = self.all_images[idx]
        image = Image.open(img_path).convert("L")

        if self.transform:
            image = self.transform(image)

        return image, self.label


disruptive_dir = "/home/wbsong/Camouflage/Data/3_backgrounds/sepia213/disruptive_pattern"
mottled_dir = "/home/wbsong/Camouflage/Data/3_backgrounds/sepia213/mottled_pattern"

data_transforms = transforms.Compose([transforms.CenterCrop((512, 256)),
                                      transforms.Resize((128, 64)),
                                      transforms.ToTensor()])

disruptive_dataset = pattern_dataset(disruptive_dir, 1.0, data_transforms)
mottled_dataset = pattern_dataset(mottled_dir, 0.0, data_transforms)
training_dataset = disruptive_dataset + mottled_dataset
training_dataloader = DataLoader(training_dataset, batch_size=32, shuffle=True)


""" Denosing Score Matching """
# score estimator --- MLP --- Denoising auto-encoder
class ScoreNet(nn.Module):

    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, 128), nn.Tanh(),
                                 nn.Linear(128, 64), nn.Tanh(),
                                 nn.Linear(64, 32), nn.Tanh(),
                                 nn.Linear(32, 64), nn.Tanh(),
                                 nn.Linear(64, 128), nn.Tanh(),
                                 nn.Linear(128, latent_dim))

    def forward(self, x):
        scores = self.net(x)
        return scores


score_model = ScoreNet(latent_dim)
score_model.to(device)

# training procedure
optimizer = torch.optim.Adam(score_model.parameters(), lr=1e-3)
epochs = 100
sigma = torch.tensor(1, device=device)
training_loss = []

for epoch in range(epochs):
    epoch_loss = 0
    for i, (batch_x, _) in enumerate(training_dataloader):

        # features in latent space
        batch_x = batch_x.to(device)
        batch_z = Encoding(VAE_model, batch_x)

        noise = torch.randn_like(batch_z)
        noised_z = batch_z + sigma * noise
        batch_loss = torch.mean(torch.sum((sigma * score_model(noised_z) + noise) ** 2, dim=1))

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        epoch_loss += batch_loss.item() / len(training_dataloader)

    training_loss.append(epoch_loss)
training_loss = np.array(training_loss)

plt.figure(dpi=300)
plt.plot(range(epochs), training_loss)
plt.show()

# validation --- denoise
x, _ = training_dataset[0]
x = input.unsqueeze(0).to(device)
z = Encoding(VAE_model, input)

noised_z = z + sigma * torch.randn_like(z)
with torch.no_grad():
    z = noised_z + score_model(noised_z) + sigma ** 2

noised_x_reco = Decoding(VAE_model, noised_z)
x_reco = Decoding(VAE_model, z)

fig, axs = plt.subplots(1, 2, dpi=300)
img = transforms.ToPILImage()(x_reco.squeeze(0).cpu())
noised_img = transforms.ToPILImage()(noised_x_reco.squeeze(0).cpu())
axs[0].imshow(img, cmap='gray')
axs[1].imshow(noised_img, cmap='gray')
plt.show()


""" Langevin MCMC"""
x_0, _ = mottled_dataset[0]
x_1, _ = disruptive_dataset[0]
# visualizaiton
img_0 = transforms.ToPILImage()(x_0)
img_1 = transforms.ToPILImage()(x_1)
fig, axs = plt.subplots(1, 2, dpi=300)
axs[0].imshow(img_0, cmap='gray')
axs[1].imshow(img_1, cmap='gray')
plt.show()

x_0 = x_0.unsqueeze(0).to(device)
z_0 = Encoding(VAE_model, x_0)
x_1 = x_1.unsqueeze(0).to(device)
z_1 = Encoding(VAE_model, x_1)

num_feedback = 10
step = 1 / 100
e1 = 0.5 # intrinsic gradient
e2 = 1  # input from classifier
e3 = 0.5 # noise
l = 1e-3  # exponential decay of input
threshold = 1e-3  # when to get the next feedback

z_t = z_0
trajectory = [z_0.detach()]
for i in range(num_feedback):

    z_t = z_t.requires_grad_(True)
    energy = - torch.sum((z_1 - z_t) ** 2)
    energy_grad = torch.autograd.grad(outputs=energy, inputs=z_t)[0]
    z_t = z_t.detach()

    j = 0
    while torch.norm(energy_grad) > threshold:
        energy_grad = energy_grad * np.exp(-j * l).item()

        with torch.no_grad():
            score = score_model(z_t)

        noise = torch.randn_like(z_t)
        z_t = z_t + e1 * score * step + e2 * energy_grad * step + e3 * noise * (step ** 0.5)
        trajectory.append(z_t)
        j += 1

fig, axs = plt.subplots(1, 11, dpi=300)
j = 0
for i in np.linspace(1, (len(trajectory)-1), 11, dtype=int):
    x_reco = Decoding(VAE_model, trajectory[i])
    x_reco = x_reco.squeeze(0)
    img = transforms.ToPILImage()(x_reco)
    axs[j].imshow(img, cmap='gray')
    axs[j].set_xticks([])
    axs[j].set_yticks([])
    j += 1
plt.show()

trajectory = torch.cat(trajectory, dim=0)
trajectory = trajectory.detach().cpu().numpy()

mean = np.mean(trajectory, axis=0, keepdims=True)
std = np.std(trajectory, axis=0, keepdims=True)
trajectory_std = (trajectory - mean) / std

pca = PCA(n_components=2, svd_solver='full').fit(trajectory_std)
trajectory_pca = pca.transform(trajectory_std)

plt.figure(dpi=300)
plt.plot(trajectory_pca[:, 0], trajectory_pca[:, 1])
plt.show()