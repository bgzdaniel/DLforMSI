from ae_h5_conv_train import *

data, mz_array, xpos, ypos = load_data()
pixel_count = data.shape[0]
intensity_count = data.shape[1]
padded_count = 64000
x_size = (np.max(xpos) + 1).astype(int)
y_size = (np.max(ypos) + 1).astype(int)

def get_plot_image(mz, bound):
    lower = mz - bound
    higher = mz + bound
    idx = np.argwhere((mz_array >= lower) & (mz_array <= higher))
    im = np.squeeze(np.sum(data[:, idx], 1), 1)
    imgData = np.zeros((x_size, y_size))
    imgData[xpos, ypos] = im
    return imgData

im1 = get_plot_image(738.4548, 0.1)
im2 = get_plot_image(600.0614, 0.15)

model, optimizer, loss_function, device = init_model(padded_count)
out_data = torch.zeros(pixel_count, padded_count)
for i in range(0, pixel_count):
        batch = torch.from_numpy(data[i, :]).to(device)
        batch = F.pad(batch, (0, (padded_count-intensity_count)), "constant", 0)
        batch = torch.unsqueeze(torch.unsqueeze(batch, 0), 0)
        with torch.no_grad():
            out_data[i, :] = torch.squeeze(torch.squeeze(model(batch), 0), 0)
data = out_data.cpu().numpy()

im3 = get_plot_image(738.4548, 0.1)
im4 = get_plot_image(600.0614, 0.15)

plt.figure(figsize=(10,10))
plt.subplot(2, 2, 1)
plt.imshow(im1)
plt.clim(-4, 4)
plt.axis("off")
plt.title("original data, 738.4548+-0.1 m/z")
plt.colorbar()
plt.subplot(2, 2, 2)
plt.imshow(im2)
plt.clim(-4, 4)
plt.axis("off")
plt.title("original data, 600.0614+-0.15 m/z")
plt.colorbar()
plt.subplot(2, 2, 3)
plt.imshow(im3)
plt.clim(-4, 4)
plt.axis("off")
plt.title("network prediction, 738.4548+-0.1 m/z")
plt.colorbar()
plt.subplot(2, 2, 4)
plt.imshow(im4)
plt.clim(-4, 4)
plt.axis("off")
plt.title("network prediction, 600.0614+-0.15 m/z")
plt.colorbar()
plt.savefig("prostate_example.png")