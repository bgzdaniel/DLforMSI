from ae_h5_conv_train import *
import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans

data, mz_array, xpos, ypos = load_data()
pixel_count = data.shape[0]
intensity_count = data.shape[1]
padded_count = 64000

# kmeans
mz = 738.4548
lower = mz - 0.1
higher = mz + 0.1
ids = np.argwhere((mz_array >= lower) & (mz_array <= higher))
im = np.sum(data[:, ids], 1)
k = 4
kmeans = KMeans(n_clusters=k, random_state=0).fit(im)
labels = kmeans.labels_

# scatmat
size = 128
model, optimizer, loss_function, device = init_model(padded_count)
encoding_list = []
for i in range(30):
    idx = np.random.randint(pixel_count, size=(size))
    batch = torch.from_numpy(data[idx, :]).to(device)
    batch = F.pad(batch, (0, (padded_count-intensity_count)), "constant", 0)
    batch = torch.unsqueeze(batch, 1)
    with torch.no_grad():
        encoding_list.append(np.insert(model.module.encode(batch).cpu().numpy(), latent_size, labels[idx], axis=1))
encoding = encoding_list[0]
for i in range(1, len(encoding_list)):
    encoding = np.concatenate((encoding, encoding_list[i]), axis=0)
dataset = encoding
columns = ["dim" + str(i+1) for i in range(latent_size)]
columns.append("label")
df = pd.DataFrame(dataset, columns = columns)
plot = sns.pairplot(df, hue="label", palette="viridis")
fig = plot.fig
fig.savefig("prostate_scatmat.png")