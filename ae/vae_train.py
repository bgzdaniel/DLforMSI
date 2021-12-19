from pyM2aia import M2aiaImageHelper
from ctypes import CDLL
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import SimpleITK as sitk
from pathlib import Path

print(f"torch version: {torch.__version__}")

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
class Vae(nn.Module):
    def __init__(self, intensity_count, latent_size):
        super(Vae, self).__init__()

        # encoder
        self.enc1 = nn.Linear(intensity_count, intensity_count//8)
        self.enc2 = nn.Linear(intensity_count//8, intensity_count//64)
        self.enc3 = nn.Linear(intensity_count//64, latent_size)

        # decoder
        self.dec1 = nn.Linear(latent_size, intensity_count//64)
        self.dec2 = nn.Linear(intensity_count//64, intensity_count//8)
        self.dec3 = nn.Linear(intensity_count//8, intensity_count)

    def encode(self, x):
        x = torch.tanh(self.enc1(x))
        x = torch.tanh(self.enc2(x))
        x = self.enc3(x)
        return x

    def decode(self, x):
        x = torch.tanh(self.dec1(x))
        x = torch.tanh(self.dec2(x))
        x = self.dec3(x)
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
    loss = loss_function(reconstruction, batch)
    loss.backward()
    optimizer.step()
    return loss

# model initialization
def initialize_model(intensity_count, latent_size):
    print(f"torch cuda version: {torch.version.cuda}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Used device: {device}")
    model = Vae(intensity_count, latent_size)
    if torch.cuda.device_count() > 1:
        print(torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)
    model.to(device)
    model_params = Path("/home/dbogacz/Development/pyM2aia/tests/model_params.pt")
    print(f"model_params already exist? {model_params.is_file()}")
    if model_params.is_file():
        print(f"loading model_params...")
        model.load_state_dict(torch.load(model_params))
    print(f"model on cuda? {next(model.parameters()).is_cuda}")
    optimizer = optim.Adam(model.parameters())
    loss_function = nn.MSELoss()
    return model, optimizer, loss_function, device


lib = CDLL('libM2aiaCoreIO.so')
image = "../training_data/ew_section3_pos.imzML"
params = "../m2PeakPicking.txt"

if __name__ == '__main__':
    with M2aiaImageHelper(lib, image, params) as helper:

        # get intensity count and size of the image
        gen = helper.SpectrumIterator()
        pixel = next(gen)[2]
        intensity_count = len(pixel)
        I = helper.GetImage(1000, 0.54, np.float32)
        A = sitk.GetArrayFromImage(I)
        x_size = A.shape[1]
        y_size = A.shape[2]
        print(A.shape)
        print(intensity_count)

        # initialize model
        latent_size = 16
        model, optimizer, loss_function, device = initialize_model(intensity_count, latent_size)

        # calculate mean and variance
        aggregate = [0, np.zeros((intensity_count)), np.zeros((intensity_count))]
        gen = helper.SpectrumIterator()
        for data in gen:
            i, xs, ys = data
            aggregate = update(aggregate, ys)
        im_mean, im_variance, im_sampleVariance = finalize(aggregate)
        im_stddev = np.sqrt(im_sampleVariance)

        # training
        iterations = 1000
        total_loss = []
        kld_loss = []
        rec_loss = []
        batch_iter = helper.SpectrumRandomBatchIterator(512)
        for i in range(iterations):
            batch = next(batch_iter)
            batch -= im_mean # data zero centering
            batch /= (im_stddev + 1e-10) # data normalization
            batch = torch.from_numpy(batch).to(device)
            loss = train(batch)
            
            # visualization
            total_loss.append(loss.item())

            if i % (iterations//10) == 0:
                print(f"{i:6d}: loss: {loss.item():3.4f}")
        
        # save learned model parameters
        torch.save(model.state_dict(), "/home/dbogacz/Development/pyM2aia/tests/model_params.pt")

        # save encoded image to .nrrd file
        print(f"saving .nrrd image...")
        imgData = np.zeros((1, x_size, y_size, latent_size))
        gen = helper.SpectrumIterator()
        for data in gen:
            id, xs, ys = data
            x, y, z = helper.GetPixelPosition(id)
            ys -= im_mean # data zero centering
            ys /= (im_stddev + 1e-10) # data normalization
            ys = torch.unsqueeze(torch.from_numpy(ys).to(device), 0)
            with torch.no_grad():
                latent_vector = model.module.encode(ys)
            imgData[z, y, x, :] = latent_vector.cpu().numpy()
        im = sitk.GetImageFromArray(imgData)
        sitk.WriteImage(im, "/home/dbogacz/Development/pyM2aia/tests/worm_section3.nrrd")

        # plot loss
        array = np.array(total_loss)
        array = np.convolve(array, np.full((10), 0.1), "valid")
        plt.figure(1)
        plt.plot(array)
        plt.savefig('vae_training_loss.png')