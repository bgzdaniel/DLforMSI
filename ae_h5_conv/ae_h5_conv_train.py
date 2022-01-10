import numpy as np
import h5py
import SimpleITK as sitk
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from pathlib import Path
import torch.nn.functional as F

latent_size = 16

class ConvAE(nn.Module):
    def __init__(self, intensity_count):
        super(ConvAE, self).__init__()

        self.intensity_count = intensity_count

        # pooling layer
        self.pool = nn.MaxPool1d(2)
        
        # encoding
        self.conv1 = nn.Conv1d(1, 2, 3, padding="same")
        self.conv2 = nn.Conv1d(2, 4, 3, padding="same")
        self.conv3 = nn.Conv1d(4, 8, 3, padding="same")
        self.conv4 = nn.Conv1d(8, 16, 3, padding="same")
        self.conv5 = nn.Conv1d(16, 32, 3, padding="same")
        self.enc1 = nn.Linear(32*(intensity_count//(2**5)), latent_size)
        
        # decoding
        self.dec1 = nn.Linear(latent_size, 32*(intensity_count//(2**5)))
        self.deconv1 = nn.ConvTranspose1d(32, 16, 2, stride=2)
        self.deconv2 = nn.ConvTranspose1d(16, 8, 2, stride=2)
        self.deconv3 = nn.ConvTranspose1d(8, 4, 2, stride=2)
        self.deconv4 = nn.ConvTranspose1d(4, 2, 2, stride=2)
        self.deconv5 = nn.ConvTranspose1d(2, 1, 2, stride=2)

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
        x = x.view(-1, 32, (self.intensity_count//(2**5)))
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
def init_model(intensity_count):
    print(f"torch cuda version: {torch.version.cuda}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Used device: {device}")
    model = ConvAE(intensity_count)
    print(torch.cuda.device_count(), "GPUs")
    if torch.cuda.device_count() > 1:
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

def load_data():
    print("loading spectrum data...")
    f =  h5py.File("../training_data/msiPL_Dataset/Prostate/P_1900.h5", "r")
    print(f.keys())
    data = np.transpose(np.array(f["Data"][:])).astype(np.float32)
    data_mean = np.mean(data, 0)
    data_std = np.std(data, 0)
    data -= data_mean
    data /= (data_std + 1e-10)
    mz_array = np.array(f["mzArray"][:]).astype(np.float32)
    xpos = np.array(f["xLocation"][:]).astype(np.float32)
    ypos = np.array(f["yLocation"][:]).astype(np.float32)
    return data, mz_array, xpos.astype(int), ypos.astype(int)

if __name__ == '__main__':
    data, mz_array, xpos, ypos = load_data()
    print(f"data shape: {data.shape}")
    intensity_count = data.shape[1]
    padded_count = 64000
    pixel_count = data.shape[0]

    model, optimizer, loss_function, device = init_model(padded_count)
    total_loss = []
    iterations = 1000
    batch_size = 32
    for i in range(iterations):
        idx = np.random.randint(pixel_count, size=(batch_size))
        batch = torch.from_numpy(data[idx, :]).to(device)
        batch = F.pad(batch, (0, (padded_count-intensity_count)), "constant", 0)
        batch = torch.unsqueeze(batch, 1)
        loss = train(batch)
        
        # visualization
        total_loss.append(loss.item())

        if i % (iterations//10) == 0:
            print(f"{i:6d}: loss: {loss.item():3.6f}")

    print("saving model params...")
    torch.save(model.state_dict(), "model_params.pt")

    plt.figure(1)
    plt.plot(np.convolve(total_loss, np.full((10), 0.1), mode="valid"))
    plt.savefig("prostate_loss.png")

    x_size = (np.max(xpos) + 1).astype(int)
    y_size = (np.max(ypos) + 1).astype(int)
    print(f"x_size: {x_size}, y_size: {y_size}")
    imgData = np.zeros((1, x_size, y_size, latent_size))

    print("creating encoded image...")
    encoded = torch.zeros(pixel_count, latent_size)
    for i in range(0, pixel_count):
        batch = torch.from_numpy(data[i, :]).to(device)
        batch = F.pad(batch, (0, (padded_count-intensity_count)), "constant", 0)
        batch = torch.unsqueeze(torch.unsqueeze(batch, 0), 0)
        with torch.no_grad():
            encoded = model.module.encode(batch).cpu().numpy()
        imgData[0, xpos[i], ypos[i], :] = encoded
    
    im = sitk.GetImageFromArray(imgData)
    sitk.WriteImage(im, "prostate.nrrd")

    imgData = np.squeeze(imgData, 0)
    img_list = []
    for i in range(latent_size):
        img_list.append(imgData[:, :, i])
    plt.figure(figsize=(10, 10))
    for i in range(latent_size):
        plt.subplot(4, 4, i+1)
        plt.imshow(img_list[i], interpolation="none")
        plt.title("dim" + str(i+1))
        plt.axis("off")
        plt.colorbar()
    plt.savefig("prostate_encoding.png")