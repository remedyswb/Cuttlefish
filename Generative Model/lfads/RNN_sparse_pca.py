import h5py
import torch
import random
import numpy as np
from torch import nn
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

print('Is GPU available?', torch.cuda.is_available())
print('Number of available GPUs:', torch.cuda.device_count())
device = torch.device('cuda:0')  # only use one gpu for current implementation


""" Dataset """
#input_path = "/share/home/wbsong/Data_Driven_Modeling/Data/MU_traces_spca.h5"
input_path = '/home/wbsong/Camouflage/Data/MU_traces_spca.h5'
#input_path = r"D:\learning\Neuroscience\Cephalopods\Camouflage\Data\MU_traces_spca.h5"
f = h5py.File(input_path, 'r')
print(list(f.keys()))
MUs_spca = np.array(f['sparse_pca_traces'])
len_chunk = np.array(f['length_chunks'])
MUs_spca_chunk = []
for i in range(1, 22):
    start = np.sum(len_chunk[:i])
    end = start + len_chunk[i]
    chunk_data = torch.from_numpy(MUs_spca[start:end, :]).to(torch.float32)
    chunk_data = chunk_data.to(device)
    MUs_spca_chunk.append(chunk_data[:, None, :])
# randomly divide into training(16) and validation dataset(5)
idx = list(range(21))
validation_idx = random.sample(idx, 5)
training_data = []
validation_data =[]
for i in idx:
    if i in validation_idx:
        validation_data.append(MUs_spca_chunk[i])
    else:
        training_data.append(MUs_spca_chunk[i])


""" Build up a model similar to LFADS """
class Lfads(nn.Module):

    def __init__(self, input_dim, encoder_dim, controller_dim, control_dim, decoder_dim, factor_dim):
        # input_size == output_size
        super().__init__()
        # Encoder RNN
        self.encoder_rnn = nn.GRU(input_size=input_dim, hidden_size=encoder_dim, num_layers=1, batch_first=False, bidirectional=True)
        self.controller_ho_fc = nn.Linear(in_features=2 * encoder_dim, out_features=controller_dim)
        self.decoder_ho_mean_fc = nn.Linear(in_features=2 * encoder_dim, out_features=decoder_dim)
        self.decoder_ho_logvar_fc = nn.Linear(in_features=2 * encoder_dim, out_features=decoder_dim)
        # Controller RNN
        self.controller_rnn = nn.GRUCell(input_size=2 * encoder_dim + factor_dim, hidden_size=controller_dim)
        self.control_mean_fc = nn.Linear(in_features=controller_dim, out_features=control_dim)
        self.control_logvar_fc = nn.Linear(in_features=controller_dim, out_features=control_dim)
        # Decoder RNN
        self.decoder_rnn = nn.GRUCell(input_size=control_dim, hidden_size=decoder_dim)
        self.factor_fc = nn.Linear(in_features=decoder_dim, out_features=factor_dim)
        self.output_fc = nn.Linear(in_features=factor_dim, out_features=input_dim)

    def reparameterization(self, mean, logvar):
        # sample latent variable from a gaussian distribution
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mean

    def kld_loss(self, mean, logvar):
        # compute KL-divergence between latent distribution and a standard gaussian distribution
        return torch.mean(-0.5 * torch.sum(1 + logvar - mean**2 - logvar.exp(), dim=1), dim=0)

    def forward(self, inputs):
        # Encode
        encoder_ht, encoder_ho = self.encoder_rnn(inputs)
        encoder_ho = torch.cat((encoder_ho[0], encoder_ho[1]), dim=1)
        # infer initial state for controller rnn
        controller_ho = self.controller_ho_fc(encoder_ho)  # is there a need for tanh?
        # sample initial state for decoder rnn
        decoder_ho_mean = self.decoder_ho_mean_fc(encoder_ho)
        decoder_ho_logvar = self.decoder_ho_logvar_fc(encoder_ho)
        vae_loss = self.kld_loss(decoder_ho_mean, decoder_ho_logvar)
        decoder_ho = self.reparameterization(decoder_ho_mean, decoder_ho_logvar)
        # infer the factors for t=0
        factors_o = self.factor_fc(decoder_ho)
        # Decode
        T = int(len(inputs))
        factors_t = factors_o
        factors = []
        controller_ht = controller_ho
        controls = []
        decoder_ht = decoder_ho
        outputs = []
        for t in range(T):
            # control module
            controller_ht = self.controller_rnn(torch.cat((factors_t, encoder_ht[t]), dim=1), controller_ht)
            control_mean = self.control_mean_fc(controller_ht)
            control_logvar = self.control_logvar_fc(controller_ht)
            vae_loss += self.kld_loss(control_mean, control_logvar) / T
            control = self.reparameterization(control_mean, control_logvar)
            controls.append(control)
            # decode module
            decoder_ht = self.decoder_rnn(control, decoder_ht)
            factors_t = self.factor_fc(decoder_ht)
            factors.append(factors_t)
            output = self.output_fc(factors_t)
            outputs.append(output)
        return torch.stack(controls, dim=0), torch.stack(factors, dim=0), torch.stack(outputs, dim=0), vae_loss

    def decoder_recurrence(self, decoder_ht_1):
        # intrinsic dynamics of decoder rnn
        batch_size = decoder_ht_1.shape[0]
        control = torch.zeros(batch_size, 1)
        decoder_ht = self.decoder_rnn(control, decoder_ht_1)
        return decoder_ht


# for instance
input_dim = 64
encoder_dim = 64
controller_dim = 64
control_dim = 3
decoder_dim = 128
factor_dim = 32
model = Lfads(input_dim, encoder_dim, controller_dim, control_dim, decoder_dim, factor_dim)
model.to(device)

""" Training & Validation """
epochs = 300
loss_fn = nn.MSELoss()
# optimization
learning_rate = 1e-2
kld_weight = 1e-3
named_params = list(model.named_parameters())
# indexes for recurrent weights: [15, 23]
rnn_hh_weights = []
others = []
i = 0
for name, params in named_params:
    if i == 15 or i == 23:
        print(name)
        rnn_hh_weights.append(params)
    else:
        others.append(params)
    i += 1
# add an l2 penalty to recurrent portions of controller and decoder
optimizer = torch.optim.Adam([{'params': rnn_hh_weights, 'weight_decay': 1e-3},
                              {'params': others, 'weight_decay': 0}], lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=5, min_lr=1e-5)

loss = []
test = []
for epoch in tqdm(range(epochs)):
    # training
    loss_epoch = 0
    for input in training_data:
        _, _, output, vae_loss = model(input)
        loss_epoch += (loss_fn(output, input) + vae_loss * kld_weight) / 16

    loss_epoch.backward()
    optimizer.step()
    optimizer.zero_grad()
    loss.append(loss_epoch.item())

    # validation
    with torch.no_grad():
        test_epoch = 0
        for input in validation_data:
            _, _, output, vae_loss = model(input)
            test_epoch += (loss_fn(output, input) + vae_loss * kld_weight) / 5
        test.append(test_epoch.item())

    scheduler.step(sum(test[epoch - 4:epoch + 1]) / 5)

    if (epoch+1) % 100 == 0:
        print(f' learning rate {epoch + 1}: ', optimizer.param_groups[0]['lr'])
        print(f'\n loss of epoch {epoch + 1}: ', loss_epoch.item())
        print(f' test loss of epoch {epoch + 1}: ', test_epoch.item())


fig, axes = plt.subplots(2, 1, figsize=[6, 8])
axes[0].set_title('training loss')
axes[0].plot(loss)
axes[1].set_title('test loss')
axes[1].plot(test)
fig.tight_layout()
plt.show()

# visualize performance of reconstructing training data
chunk_idx = np.random.randint(16)
input = training_data[chunk_idx]
with torch.no_grad():
    control, factor, output, _ = model(input)
input = input[:, 0, :].cpu().numpy()
control = control[:, 0, :].cpu().numpy()
factor = factor[:, 0, :].cpu().numpy()
output = output[:, 0, :].cpu().numpy()

# three major type of components: [0, 37, 51]
components_idx = [0, 50, 61, 5, 58, 37, 3, 2, 57, 51, 38]
fig = plt.figure(figsize=[9, 6])
ax1 = fig.add_subplot(121, projection='3d')
plt.title('trajectory of components')
ax1.plot(input[:, 0], input[:, 37], input[:, 51], label='real')
ax1.plot(output[:, 0], output[:, 37], output[:, 51], label='recon')
plt.legend()

ax2 = fig.add_subplot(122, projection='3d')
plt.title('trajectory of control')
ax2.plot(control[:, 0], control[:, 1], control[:, 2])
fig.tight_layout()
plt.show()

fig, axs = plt.subplots(3, 1, figsize=[8, 6])
for i in range(3):
    idx = [0, 37, 51][i]
    axs[i].plot(input[:, idx], label='real')
    axs[i].plot(output[:, idx], label='recon')
    axs[i].legend()
    axs[i].set_title(f'component_{idx}')
fig.tight_layout()
plt.show()

# Validation
chunk_idx = np.random.randint(5)
input = validation_data[chunk_idx]
with torch.no_grad():
    control, factor, output, _ = model(input)
input = input[:, 0, :].cpu().numpy()
control = control[:, 0, :].cpu().numpy()
factor = factor[:, 0, :].cpu().numpy()
output = output[:, 0, :].cpu().numpy()

fig = plt.figure(figsize=[9, 6])
ax1 = fig.add_subplot(121, projection='3d')
plt.title('trajectory of components')
ax1.plot(input[:, 0], input[:, 37], input[:, 51], label='real')
ax1.plot(output[:, 0], output[:, 37], output[:, 51], label='recon')
plt.legend()

ax2 = fig.add_subplot(122, projection='3d')
plt.title('trajectory of control')
ax2.plot(control[:, 0], control[:, 1], control[:, 2])
plt.show()

fig, axs = plt.subplots(3, 1, figsize=[8, 6])
for i in range(3):
    idx = [0, 37, 51][i]
    axs[i].plot(input[:, idx], label='real')
    axs[i].plot(output[:, idx], label='recon')
    axs[i].legend()
    axs[i].set_title(f'component_{idx}')
fig.tight_layout()
plt.show()

'''
""" Dynamical systems analysis """
# visualize low-dimensional decoder rnn activity for in training data trails
# considering pressure on memory, use the first five chunks corresponding to mottled to disruptive pattern transition
activity = []
for i in range(5):
    input = training_data[i]
    with torch.no_grad():
        _, rnn_activity, _ = model(input)  # (time_steps, batch_size=1, hidden_size)
        rnn_activity = rnn_activity[:, 0, :].numpy()  # (time_steps, hidden_size)
        activity.append(rnn_activity)
activity = np.concatenate(activity, axis=0)
# dimension reduction
pca = PCA(n_components=2)
pca.fit(activity)
print('variance explained: ', pca.explained_variance_ratio_.sum())
activity_pcs = pca.transform(activity)
plt.plot(activity_pcs[:, 0], activity_pcs[:, 1])
plt.show()
# visualize the trajectory  # wrong
plt.figure(dpi=200)
for i in range(1, 17):
    start = np.sum(len_chunk[:i])
    end = start + len_chunk[i]
    activity_pcs_chunk = activity_pcs[start: end, :]
    plt.plot(activity_pcs_chunk[:, 0], activity_pcs_chunk[:, 1], alpha=0.8)
plt.show()

# visualize vector field (intrinsic dynamics)
batch_size = 1024
decoder_ht_1 = torch.tensor(np.random.uniform(-1, 1, (batch_size, hidden_size)) * 6, dtype=torch.float32)
with torch.no_grad():
    decoder_ht = model.decoder_recurrence(decoder_ht_1)
deta_h = decoder_ht - decoder_ht_1
xy_pca = pca.transform(decoder_ht_1.numpy())
uv_pca = pca.transform(deta_h.numpy())
plt.figure(dpi=200)
plt.quiver(xy_pca[:, 0], xy_pca[:, 1], uv_pca[:, 0], uv_pca[:, 1], alpha=0.5)
plt.show()

# search for approximate fix points
# Freeze for parameters in the recurrent network
for param in model.parameters():
    param.requires_grad = False

# Here hidden activity is the variable to be optimized
batch_size = 128
hidden = torch.tensor(np.random.uniform(-1, 1, (batch_size, hidden_size)) * 5, requires_grad=True, dtype=torch.float32)

# Use Adam optimizer
optimizer = torch.optim.Adam([hidden], lr=0.001)  # why '[]'?
criterion = nn.MSELoss()

running_loss = 0
for i in range(15000):
    optimizer.zero_grad()  # zero the gradient buffers

    # Take the one-step recurrent function from the trained network
    new_h = model.decoder_recurrence(hidden)
    loss = criterion(new_h, hidden)
    loss.backward()
    optimizer.step()  # Does the update

    running_loss += loss.item()
    if i % 1000 == 999:
        running_loss /= 1000  # average every 100 steps
        print('Step {}, Loss {:0.4f}'.format(i + 1, running_loss))
        running_loss = 0  # reset

# visualize the location of fixed points
fixed_points = hidden.detach().numpy()
fixed_points_pcs = pca.transform(fixed_points)
plt.plot(fixed_points_pcs[:, 0], fixed_points_pcs[:, 1], '*')
plt.show()
'''