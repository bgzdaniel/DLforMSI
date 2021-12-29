# %%
from pyM2aia import M2aiaImageHelper
from ctypes import CDLL
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import SimpleITK as sitk
from pathlib import Path
import torch.nn.functional as F
import pandas as pd
import seaborn as sns

lib = CDLL('libM2aiaCoreIO.so')
image = "ew_section3_pos.imzML"
params = "m2PeakPicking.txt"

intensity_count = 0
I = None
pixel = None
with M2aiaImageHelper(lib, image, params) as helper:
    gen = helper.SpectrumIterator()
    pixel = next(gen)[2]
    intensity_count = len(pixel)
    I = helper.GetImage(1088.868, 0.249, np.float32)
    label1 = sitk.ReadImage("1088_868_label1.nrrd")
sitk.WriteImage(I, "1088_868.nrrd")
label1 = sitk.GetArrayFromImage(label1)
A = sitk.GetArrayFromImage(I)
x_size = A.shape[1]
y_size = A.shape[2]
print(A.shape)
print(I.GetSize())
print(intensity_count)


# %%
# Welford's online algorithm for calculating mean and variance

# For a new value newValue, compute the new count, new mean, the new M2.
# mean accumulates the mean of the entire dataset
# M2 aggregates the squared distance from the mean
# count aggregates the number of samples seen so far
def update(existingAggregate, newValue):
    (count, mean, M2) = existingAggregate
    count += 1
    delta = newValue - mean
    mean += delta / count
    delta2 = newValue - mean
    M2 += delta * delta2
    return (count, mean, M2)

# Retrieve the mean, variance and sample variance from an aggregate
def finalize(existingAggregate):
    (count, mean, M2) = existingAggregate
    if count < 2:
        return float("nan")
    else:
        (mean, variance, sampleVariance) = (mean, M2 / count, M2 / (count - 1))
        return (mean, variance, sampleVariance)

aggregate = [0, np.zeros((intensity_count)), np.zeros((intensity_count))]
with M2aiaImageHelper(lib, image, params) as helper:
    gen = helper.SpectrumIterator()
    for data in gen:
        i, xs, ys = data
        aggregate = update(aggregate, ys)
im_mean, im_variance, im_sampleVariance = finalize(aggregate)
im_stddev = np.sqrt(im_sampleVariance)
print(im_mean, im_variance, im_sampleVariance)

# %%
print(f"torch cuda version: {torch.version.cuda}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# %%
latent_size = 16
class ConvAE(nn.Module):
    def __init__(self):
        super(ConvAE, self).__init__()

        # pooling layer
        self.pool = nn.MaxPool1d(2)
        
        # encoding
        self.conv2 = nn.Conv1d(4, 16, 3, padding="same")
        self.conv1 = nn.Conv1d(1, 4, 3, padding="same")
        self.conv3 = nn.Conv1d(16, 64, 3, padding="same")
        self.conv4 = nn.Conv1d(64, 256, 3, padding="same")
        self.conv5 = nn.Conv1d(256, 1024, 3, padding="same")
        self.enc1 = nn.Linear(1024*1000, latent_size*2)
        
        # decoding
        self.dec1 = nn.Linear(latent_size, 1024*1000)
        self.deconv1 = nn.ConvTranspose1d(1024, 256, 2, stride=2)
        self.deconv2 = nn.ConvTranspose1d(256, 64, 2, stride=2)
        self.deconv3 = nn.ConvTranspose1d(64, 16, 2, stride=2)
        self.deconv4 = nn.ConvTranspose1d(16, 4, 2, stride=2)
        self.deconv5 = nn.ConvTranspose1d(4, 1, 2, stride=2)

    def encode(self, x):
        x = self.pool(torch.tanh(self.conv1(x)))
        x = self.pool(torch.tanh(self.conv2(x)))
        x = self.pool(torch.tanh(self.conv3(x)))
        x = self.pool(torch.tanh(self.conv4(x)))
        x = self.pool(torch.tanh(self.conv5(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.enc1(x)

        # sampling
        x = x.view(-1, 2, latent_size)
        mu = x[:, 0, :]
        logvar = x[:, 1, :]
        x = self.reparameterize(mu, logvar)
        return x, mu, logvar

    def decode(self, x):
        x = torch.tanh(self.dec1(x))
        x = x.view(-1, 1024, 1000)
        x = torch.tanh(self.deconv1(x))
        x = torch.tanh(self.deconv2(x))
        x = torch.tanh(self.deconv3(x))
        x = torch.tanh(self.deconv4(x))
        x = self.deconv5(x)
        return x

    def forward(self, x):
        # encoding
        x, mu, logvar = self.encode(x)

        # decoding
        x = self.decode(x)
        return x, mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        sample = mu + (eps * std)
        return sample

def train(batch):
    optimizer.zero_grad()
    reconstruction, mu, logvar = model(batch)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = loss_function(reconstruction, batch) # reconstruction loss
    loss = loss + kld
    loss.backward()
    optimizer.step()
    return loss

# %%
model = ConvAE()
if torch.cuda.device_count() > 1:
    print(torch.cuda.device_count(), "GPUs")
    model = nn.DataParallel(model)
model.to(device)
model_params = Path("/home/dbogacz/Development/pyM2aia/tests/model_params_conv_kld.pt")
print(f"model_params already exist? {model_params.is_file()}")
if model_params.is_file():
    model.load_state_dict(torch.load(model_params))
print(f"model on cuda? {next(model.parameters()).is_cuda}")
optimizer = optim.Adam(model.parameters())
loss_function = nn.MSELoss(reduction="sum")

# %%
iterations = 1000
total_loss = []
kld_loss = []
rec_loss = []
with M2aiaImageHelper(lib, image, params) as helper:

    # training
    batch_iter = helper.SpectrumRandomBatchIterator(32)
    for i in range(iterations):
        batch = next(batch_iter)
        batch -= im_mean # data zero centering
        batch /= (im_stddev + 1e-10) # data normalization
        batch = torch.unsqueeze(torch.from_numpy(batch).to(device), 1)
        batch = F.pad(batch, (0, 2307), "constant", 0)
        loss = train(batch)
        total_loss.append(loss.item())

        if i % (iterations//10) == 0:
            print(f"{i:6d}: loss: {loss.item():3.4f}")
    
    # save learned model parameters
    # torch.save(model.state_dict(), "/home/dbogacz/Development/pyM2aia/tests/model_params_conv_kld.pt")

# %%
array = np.array(total_loss)
array = np.convolve(array, np.full((10), 0.1), "valid")
plt.figure(1)
plt.plot(array)

# %%
data = []
with M2aiaImageHelper(lib, image, params) as helper:
    for i in range(2000):
        id = int(np.random.random_integers(0, helper.GetNumberOfSpectra()-1, 1))
        xs, ys = helper.GetSpectrum(id)
        x, y, z = helper.GetPixelPosition(id)
        label = str(label1[z, y, x])
        ys -= im_mean # data zero centering
        ys /= (im_stddev + 1e-10) # data normalization
        ys = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(ys).to(device), 0), 0)
        ys = F.pad(ys, (0, 2307), "constant", 0)
        with torch.no_grad():
            ys, mu, logvar = model.module.encode(ys)
            ys = torch.squeeze(ys, 0).cpu().numpy()
        ys = list(ys)
        ys.append(label)
        data.append(ys)
columns = ["dim" + str(i+1) for i in range(16)]
columns.append("label")
df = pd.DataFrame(data, columns = columns)
plot = sns.pairplot(df, hue="label")
fig = plot.fig
fig.savefig("/home/dbogacz/Development/pyM2aia/tests/worm_conv_kld_scatmat.png")