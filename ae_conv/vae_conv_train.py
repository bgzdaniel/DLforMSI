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

# model definition
class ConvAE(nn.Module):
    def __init__(self, latent_size):
        super(ConvAE, self).__init__()

        # pooling layer
        self.pool = nn.MaxPool1d(2)
        
        # encoding
        self.conv2 = nn.Conv1d(4, 16, 3, padding="same")
        self.conv1 = nn.Conv1d(1, 4, 3, padding="same")
        self.conv3 = nn.Conv1d(16, 64, 3, padding="same")
        self.conv4 = nn.Conv1d(64, 256, 3, padding="same")
        self.conv5 = nn.Conv1d(256, 1024, 3, padding="same")
        self.enc1 = nn.Linear(1024*1000, latent_size)
        
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
        return x

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
        x = self.encode(x)

        # decoding
        x = self.decode(x)
        return x

def train(batch):
    optimizer.zero_grad()
    reconstruction = model(batch)
    loss = loss_function(reconstruction, batch) # reconstruction loss
    loss.backward()
    optimizer.step()
    return loss

# model initialization
def init_model(latent_size):
    print(f"torch cuda version: {torch.version.cuda}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Used device: {device}")
    model = ConvAE(latent_size)
    if torch.cuda.device_count() > 1:
        print(torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)
    model.to(device)
    model_params = Path("/home/dbogacz/Development/pyM2aia/tests/model_params_conv.pt")
    print(f"model_params already exist? {model_params.is_file()}")
    if model_params.is_file():
        print(f"loading model_params...")
        model.load_state_dict(torch.load(model_params))
    print(f"model on cuda? {next(model.parameters()).is_cuda}")
    optimizer = optim.Adam(model.parameters())
    loss_function = nn.MSELoss()
    return model, optimizer, loss_function, device

lib = CDLL('libM2aiaCoreIO.so')
image = "ew_section3_pos.imzML"
params = "m2PeakPicking.txt"

if __name__ == '__main__':
    with M2aiaImageHelper(lib, image, params) as helper:
        gen = helper.SpectrumIterator()
        pixel = next(gen)[2]
        intensity_count = len(pixel)
        I = helper.GetImage(1088.868, 0.249, np.float32)
        A = sitk.GetArrayFromImage(I)
        x_size = A.shape[1]
        y_size = A.shape[2]
        print(A.shape)
        print(I.GetSize())
        print(intensity_count)

        aggregate = [0, np.zeros((intensity_count)), np.zeros((intensity_count))]
        gen = helper.SpectrumIterator()
        for data in gen:
            i, xs, ys = data
            aggregate = update(aggregate, ys)
        im_mean, im_variance, im_sampleVariance = finalize(aggregate)
        im_stddev = np.sqrt(im_sampleVariance)

        # initialize model
        latent_size = 16
        model, optimizer, loss_function, device = init_model(latent_size)
        
        # training
        iterations = 500
        total_loss = []
        batch_iter = helper.SpectrumRandomBatchIterator(64)
        for i in range(iterations):
            batch = next(batch_iter)
            # batch -= im_mean # data zero centering
            # batch /= (im_stddev + 1e-10) # data normalization
            batch = torch.unsqueeze(torch.from_numpy(batch).to(device), 1)
            batch = F.pad(batch, (0, 2307), "constant", 0)
            loss = train(batch)
            total_loss.append(loss.item())

            if i % (iterations//10) == 0:
                print(f"{i:6d}: loss: {loss.item():3.8f}")
        
        # save learned model parameters
        torch.save(model.state_dict(), "/home/dbogacz/Development/pyM2aia/tests/model_params_conv.pt")

        # saving image
        print("creating .nrrd file...")
        imgData = np.zeros((1, x_size, y_size, latent_size))
        gen = helper.SpectrumIterator()
        for data in gen:
            id, xs, ys = data
            x, y, z = helper.GetPixelPosition(id)
            # ys -= im_mean # data zero centering
            # ys /= (im_stddev + 1e-10) # data normalization
            ys = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(ys).to(device), 0), 0)
            ys = F.pad(ys, (0, 2307), "constant", 0)
            with torch.no_grad():
                latent_vector = model.module.encode(ys)
            imgData[z, y, x, :] = latent_vector.cpu().numpy()
        im = sitk.GetImageFromArray(imgData)
        sitk.WriteImage(im, "/home/dbogacz/Development/pyM2aia/tests/worm_conv_section3.nrrd")

        array = np.array(total_loss)
        array = np.convolve(array, np.full((10), 0.1), "valid")
        plt.figure(1)
        plt.plot(array)
        plt.savefig('vae_conv_training_loss.png')
    