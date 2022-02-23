import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import SimpleITK as sitk
from pathlib import Path
import time

from load_data import *

print(f"torch version: {torch.__version__}")

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
        # x = self.activation(self.enc3(x))

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

if __name__ == '__main__':
    start = time.time()

    data, mz_array, xpos, ypos = load_data()
    print(f"data shape: {data.shape}")
    intensity_count = data.shape[1]
    pixel_count = data.shape[0]

    model, optimizer, loss_function, device = init_model(intensity_count)
    total_loss = []
    iterations = 10000
    batch_size = 32
    for i in range(iterations):
        idx = np.random.randint(pixel_count, size=(batch_size))
        batch = torch.from_numpy(data[idx, :]).to(device)
        loss = train(batch)
        
        # visualization
        total_loss.append(loss.item())

        if i % (iterations//10) == 0:
            print(f"{i:6d}: loss: {loss.item():3.8f}")

    print("saving model params...")
    torch.save(model.state_dict(), "model_params.pt")

    plt.figure(1)
    plt.plot(np.convolve(total_loss[1000:], np.full((10), 0.1), mode="valid"))
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.savefig("worm_loss.png")

    x_size = (np.max(xpos) + 1).astype(int)
    y_size = (np.max(ypos) + 1).astype(int)
    print(f"x_size: {x_size}, y_size: {y_size}")
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
        plt.figure(figsize=(9, 15))
        for i in range(latent_size):
            plt.subplot(3, 2, i+1)
            plt.imshow(img_list[i], interpolation="none")
            plt.title("dim" + str(i+1))
            plt.axis("off")
            plt.colorbar()
        plt.savefig("worm_encoding.png")
    else:
        plt.figure(figsize=(17, 17))
        for i in range(latent_size):
            plt.subplot(4, 4, i+1)
            plt.imshow(img_list[i], interpolation="none")
            plt.title("dim" + str(i+1))
            plt.axis("off")
            plt.colorbar()
        plt.savefig("worm_encoding.png")

    end = time.time()
    print(f"execution time: {(end - start)/60} minutes")