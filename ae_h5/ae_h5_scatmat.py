from ae_h5_train import *
import seaborn as sns
import pandas as pd
from sklearn.mixture import GaussianMixture

data, mz_array, xpos, ypos = load_data()
pixel_count = data.shape[0]
intensity_count = data.shape[1]

# kmeans
# gaussian mixture
x_size = (np.max(xpos) + 1).astype(int)
y_size = (np.max(ypos) + 1).astype(int)
mz = 738.4548
lower = mz - 0.1
higher = mz + 0.1
idx = np.argwhere((mz_array >= lower) & (mz_array <= higher))
im = np.sum(data[:, idx], 1)
k = 4
labels = GaussianMixture(n_components=k, random_state=0).fit_predict(im)
imgData = np.zeros((x_size, y_size))
imgData[xpos, ypos] = labels
plt.figure()
plt.axis("off")
plt.imshow(imgData, cmap=plt.cm.get_cmap('viridis', k))
plt.clim(0, k)
plt.colorbar()
plt.title(f"{mz}+-0.1 m/z")
plt.savefig("prostate_gaussian_mixture.png")

# scatmat
size = 2000
model, optimizer, loss_function, device = init_model(intensity_count)
idx = np.random.randint(pixel_count, size=(size))
labels = labels[idx].tolist()
batch = torch.from_numpy(data[idx, :]).to(device)
encoding = None
with torch.no_grad():
    encoding = model.module.encode(batch).cpu().numpy().tolist()
for i in range(size):
    encoding[i].append(labels[i])
dataset = encoding
columns = ["dim" + str(i+1) for i in range(latent_size)]
columns.append("label")
df = pd.DataFrame(dataset, columns = columns)
plot = sns.pairplot(df, hue="label", palette="viridis")
fig = plot.fig
fig.savefig("prostate_scatmat.png")