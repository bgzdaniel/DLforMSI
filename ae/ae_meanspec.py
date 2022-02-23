from ae_train import *
from load_data import *

data, mz_array, _, _ = load_data()
pixel_count = data.shape[0]
intensity_count = data.shape[1]
model, optimizer, loss_function, device = init_model(intensity_count)
input_mean = np.mean(data, 0)
batch = torch.from_numpy(data).to(device)
rec = None
with torch.no_grad():
    rec = model(batch).cpu().numpy()
rec_mean = np.mean(rec, 0)
diff = np.sqrt(np.power(rec_mean - input_mean, 2))
plt.figure(1)
plt.plot(mz_array, input_mean, label="input mean")
plt.plot(mz_array, rec_mean, label="rec mean")
plt.plot(mz_array, diff, "r--", label="diff")
plt.xlabel("m/z")
plt.ylabel("intensity")
plt.legend()
plt.savefig("worm_meanspec.png")
plt.figure(2)
plt.plot(mz_array, input_mean, label="input mean")
plt.plot(mz_array, rec_mean, label="rec mean")
plt.plot(mz_array, diff, "r--", label="diff")
mz = 262.177
lower = mz - 0.2
higher = mz + 0.2
plt.xlim(lower, higher)
plt.ylim(0, 0.2)
plt.xlabel("m/z")
plt.ylabel("intensity")
plt.legend()
plt.savefig("worm_meanspec_withmz.png")