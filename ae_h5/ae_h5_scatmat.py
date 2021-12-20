from ae_h5_train import *
import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans

data, mz_array, xpos, ypos = load_data()
pixel_count = data.shape[0]
intensity_count = data.shape[1]

# kmeans
mz = 400
lower = mz - 10
higher = mz + 10
ids = np.argwhere((mz_array >= lower) & (mz_array <= higher))
im = np.sum(data[:, ids], 1)
k = 3
kmeans = KMeans(n_clusters=k, random_state=0).fit(im)
labels = kmeans.labels_

# scatmat
size = 3000
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