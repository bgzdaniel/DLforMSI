from ae_h5_train import *
import seaborn as sns
import pandas as pd
from sklearn.mixture import GaussianMixture
from matplotlib import colors

data, mz_array, xpos, ypos = load_data()
pixel_count = data.shape[0]
intensity_count = data.shape[1]
model, optimizer, loss_function, device = init_model(intensity_count)
x_size = (np.max(xpos) + 1).astype(int)
y_size = (np.max(ypos) + 1).astype(int)

palette = "tab10"
colormap = [plt.cm.get_cmap(palette)(i) for i in range(0, 7)]
cmap = colors.ListedColormap(colormap)

# guassian mixture
batch = torch.from_numpy(data).to(device)
encoded = None
with torch.no_grad():
    encoded = model.module.encode(batch).cpu().numpy()
k = 6
labels = GaussianMixture(n_components=k, random_state=0).fit_predict(encoded)
imgData = np.zeros((x_size, y_size))
labels += 1
imgData[xpos, ypos] = labels
plt.figure()
plt.axis("off")
plt.imshow(imgData, cmap=cmap)
plt.clim(0, k+1)
ticks = [i for i in range(k+1)]
plt.colorbar(ticks=ticks)
plt.title(f"Gaussian Mixture of Encoded Features")
plt.savefig("prostate_gaussian_mixture.png")

colormap = [plt.cm.get_cmap(palette)(i) for i in range(1, 7)]

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
plot = sns.pairplot(df, hue="label", palette=colormap)
fig = plot.fig
fig.savefig("prostate_scatmat.png")