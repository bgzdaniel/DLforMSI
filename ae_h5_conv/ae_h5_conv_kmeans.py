from ae_h5_conv_train import *
from sklearn.cluster import KMeans
#from sklearn.mixture import GaussianMixture

data, mz_array, xpos, ypos = load_data()
pixel_count = data.shape[0]
intensity_count = data.shape[1]
x_size = (np.max(xpos) + 1).astype(int)
y_size = (np.max(ypos) + 1).astype(int)
mz = 738.4548
lower = mz - 0.1
higher = mz + 0.1
idx = np.argwhere((mz_array >= lower) & (mz_array <= higher))
im = np.sum(data[:, idx], 1)
k = 4
kmeans = KMeans(n_clusters=k, random_state=0).fit(im)
labels = kmeans.labels_
imgData = np.zeros((x_size, y_size))
imgData[xpos, ypos] = labels
plt.figure()
plt.axis("off")
plt.imshow(imgData, cmap=plt.cm.get_cmap('viridis', k))
plt.clim(0, k)
plt.colorbar()
plt.title(f"{mz}+-0.1 m/z")
plt.savefig("prostate_kmeans.png")