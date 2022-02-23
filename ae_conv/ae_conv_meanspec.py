from ae_conv_train import *
from load_data import *

data, mz_array, _, _ = load_data()
pixel_count = data.shape[0]
intensity_count = data.shape[1]
padded_count = 32000
model, optimizer, loss_function, device = init_model(padded_count)
input_mean = np.mean(data, 0)
rec = torch.zeros(pixel_count, padded_count)
for i in range(0, pixel_count):
    batch = torch.from_numpy(data[i, :]).to(device)
    batch = F.pad(batch, (0, (padded_count-intensity_count)), "constant", 0)
    batch = torch.unsqueeze(torch.unsqueeze(batch, 0), 0)
    with torch.no_grad():
        rec[i, :] = torch.squeeze(torch.squeeze(model(batch), 0), 0)
rec = rec.cpu().numpy()
rec = rec[:, 0:intensity_count]
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
plt.ylim(0, 1.5)
plt.xlabel("m/z")
plt.ylabel("intensity")
plt.legend()
plt.savefig("worm_meanspec_withmz.png")