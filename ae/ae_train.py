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

latent_size = 5

# model definition
class Vae(nn.Module):
    def __init__(self, intensity_count):
        super(Vae, self).__init__()

        self.activation = nn.ReLU()

        self.enc1_bn = nn.BatchNorm1d(512)
        self.enc1 = nn.Linear(intensity_count, 512)
        self.enc2_bn = nn.BatchNorm1d(5)
        self.enc2 = nn.Linear(512, 5)

        self.dec1_bn = nn.BatchNorm1d(512)
        self.dec1 = nn.Linear(5, 512)
        self.dec2_bn = nn.BatchNorm1d(intensity_count)
        self.dec2 = nn.Linear(512, intensity_count)

        # self.activation = nn.Tanh()

        # self.enc1 = nn.Linear(intensity_count, intensity_count//16)
        # self.enc2 = nn.Linear(intensity_count//16, intensity_count//256)
        # self.enc3 = nn.Linear(intensity_count//256, latent_size)

        # self.dec1 = nn.Linear(latent_size, intensity_count//256)
        # self.dec2 = nn.Linear(intensity_count//256, intensity_count//16)
        # self.dec3 = nn.Linear(intensity_count//16, intensity_count)

    def encode(self, x):
        x = self.activation(self.enc1_bn(self.enc1(x)))
        x = self.activation(self.enc2_bn(self.enc2(x)))

        # x = self.activation(self.enc1(x))
        # x = self.activation(self.enc2(x))
        # x = self.enc3(x)

        return x

    def decode(self, x):
        x = self.activation(self.dec1_bn(self.dec1(x)))
        x = self.activation(self.dec2_bn(self.dec2(x)))

        # x = self.activation(self.dec1(x))
        # x = self.activation(self.dec2(x))
        # x = self.dec3(x)

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
def init_model(intensity_count):
    print(f"torch cuda version: {torch.version.cuda}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Used device: {device}")
    model = Vae(intensity_count)
    if torch.cuda.device_count() > 1:
        print(torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)
    model.to(device)
    model_params = Path("model_params.pt")
    print(f"model_params already exist? {model_params.is_file()}")
    if model_params.is_file():
        print(f"loading model_params...")
        model.load_state_dict(torch.load(model_params))
    print(f"model on cuda? {next(model.parameters()).is_cuda}")
    optimizer = optim.Adam(model.parameters())
    loss_function = nn.MSELoss()
    return model, optimizer, loss_function, device


lib = CDLL('libM2aiaCoreIO.so')
image = "/home/dbogacz/Development/pyM2aia/tests/training_data/ew_section2_pos.imzML"
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
        print(f"worm shape: {A.shape}")
        print(intensity_count)

        # initialize model
        model, optimizer, loss_function, device = init_model(intensity_count)

        # calculate mean and variance
        aggregate = [0, np.zeros((intensity_count)), np.zeros((intensity_count))]
        gen = helper.SpectrumIterator()
        for data in gen:
            i, xs, ys = data
            aggregate = update(aggregate, ys)
        im_mean, im_variance, im_sampleVariance = finalize(aggregate)
        im_stddev = np.sqrt(im_sampleVariance)

        # training
        iterations = 4000
        total_loss = []
        batch_iter = helper.SpectrumRandomBatchIterator(32)
        for i in range(iterations):
            batch = next(batch_iter)
            # batch -= im_mean # data zero centering
            batch /= (im_stddev + 1e-10) # data normalization
            batch = torch.from_numpy(batch).to(device)
            loss = train(batch)
            
            # visualization
            total_loss.append(loss.item())

            if i % (iterations//10) == 0:
                print(f"{i:6d}: loss: {loss.item():3.4f}")
        
        # save learned model parameters
        torch.save(model.state_dict(), "model_params.pt")

        pixel_count = x_size * y_size
        data = np.zeros((pixel_count, intensity_count))
        mz_array = np.zeros(intensity_count)
        xpos = np.zeros(pixel_count)
        ypos = np.zeros(pixel_count)
        gen = helper.SpectrumIterator()
        xs = None
        for pixel in gen:
            id, xs, ys = pixel
            y, x, z = helper.GetPixelPosition(id)
            data[id, :] = ys
            xpos[id] = x
            ypos[id] = y
        mz_array = xs
        data = data.astype(np.float32)
        mz_array = mz_array.astype(np.float32)
        xpos = xpos.astype(int)
        ypos = ypos.astype(int)
        # data_mean = np.mean(data, 0)
        # data -= data_mean
        data_std = np.std(data, 0)
        data /= (data_std + 1e-10)

        # save encoded image to .nrrd file
        print(f"saving .nrrd image...")
        imgData = np.zeros((1, x_size, y_size, latent_size))
        print("creating encoded image...")
        batch = torch.from_numpy(data).to(device)
        encoded = None
        with torch.no_grad():
            encoded = model.module.encode(batch).cpu().numpy()
        imgData[0, xpos, ypos, :] = encoded
        im = sitk.GetImageFromArray(imgData)
        sitk.WriteImage(im, "worm.nrrd")

        imgData = np.squeeze(imgData, 0)
        img_list = []
        for i in range(latent_size):
            img_list.append(imgData[:, :, i])
        if(latent_size == 5):
            plt.figure(1, figsize=(20, 4))
            for i in range(latent_size):
                plt.subplot(1, latent_size, i+1)
                plt.imshow(img_list[i], interpolation="none")
                plt.title("dim" + str(i+1))
                plt.axis("off")
                plt.colorbar()
            plt.savefig("worm_encoding.png")
        else:
            plt.figure(1, figsize=(17, 17))
            for i in range(latent_size):
                plt.subplot(4, 4, i+1)
                plt.imshow(img_list[i], interpolation="none")
                plt.title("dim" + str(i+1))
                plt.axis("off")
                plt.colorbar()
            plt.savefig("worm_encoding.png")

        # plot loss
        array = np.array(total_loss)
        array = np.convolve(array, np.full((10), 0.1), "valid")
        plt.figure(2)
        plt.plot(array)
        plt.savefig("worm_loss.png")