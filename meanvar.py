import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

def load_data():
    print("loading spectrum data...")
    f =  h5py.File("training_data/msiPL_Dataset/Prostate/P_1900.h5", "r")
    print(f.keys())
    data = np.transpose(np.array(f["Data"][:])).astype(np.float32)
    mz_array = np.array(f["mzArray"][:]).astype(np.float32)
    xpos = np.array(f["xLocation"][:]).astype(np.float32)
    ypos = np.array(f["yLocation"][:]).astype(np.float32)
    return data, mz_array, xpos.astype(int), ypos.astype(int)

data, mz_array, xpos, ypos = load_data()
meanspec = np.mean(data, 0)
stdspec = np.std(data, 0)
dotsize = stdspec / meanspec

plt.figure(1, figsize=(20, 20))
plt.scatter(meanspec, stdspec, s=dotsize)
plt.xscale("log")
plt.yscale("log")
# plt.xlim(0, 1e6)
# plt.ylim(0, 1e6)
plt.xlabel("mean")
plt.ylabel("std")
plt.savefig("meanvar.png")

plt.figure(2, figsize=(20, 10))
plt.plot(mz_array, meanspec, "b")
plt.fill_between(mz_array, meanspec-stdspec, meanspec+stdspec, color="cornflowerblue", edgecolor="b")
plt.savefig("mzmean.png")

plt.figure(3, figsize=(20, 10))
idx = np.squeeze(np.argwhere(meanspec > 3e5), 1)
new_meanspec = meanspec[idx]
new_stdspec = stdspec[idx]
new_mz_array = mz_array[idx]
plt.plot(new_mz_array, new_meanspec, color="b")
plt.fill_between(new_mz_array, new_meanspec-new_stdspec, new_meanspec+new_stdspec, color="cornflowerblue", edgecolor="b")
plt.xlim(500, 700)
plt.ylim(-1e3, 2e7)
plt.xlabel("mz")
plt.ylabel("mean")
plt.savefig("mzmean2.png")