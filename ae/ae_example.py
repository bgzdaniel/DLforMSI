from ae_train import *
from load_data import *

data, mz_array, xpos, ypos = load_data()
pixel_count = data.shape[0]
intensity_count = data.shape[1]
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

im1 = get_plot_image(262.177, 0.2)
im2 = get_plot_image(1088.868, 0.249)

im1_top = np.max(im1) * 0.65
im2_top = np.max(im2) * 0.65

model, optimizer, loss_function, device = init_model(intensity_count)
batch = torch.from_numpy(data).to(device)
with torch.no_grad():
    data = model(batch).cpu().numpy()

im3 = get_plot_image(262.177, 0.2)
im4 = get_plot_image(1088.868, 0.249)

plt.figure(figsize=(10,10))

plt.subplot(2, 2, 1)
plt.imshow(im1)
plt.clim(0, im1_top)
plt.axis("off")
plt.title("original data, 262.177m/z+-0.2 m/z")
plt.colorbar()

plt.subplot(2, 2, 2)
plt.imshow(im2)
plt.clim(0, im2_top)
plt.axis("off")
plt.title("original data, 1088.868+-0.249 m/z")
plt.colorbar()

plt.subplot(2, 2, 3)
plt.imshow(im3)
plt.clim(0, im1_top)
plt.axis("off")
plt.title("network prediction, 262.177m/z+-0.2 m/z")
plt.colorbar()

plt.subplot(2, 2, 4)
plt.imshow(im4)
plt.clim(0, im2_top)
plt.axis("off")
plt.title("network prediction, 1088.868+-0.249 m/z")
plt.colorbar()

plt.savefig("worm_example.png")