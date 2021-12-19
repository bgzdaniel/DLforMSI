from ae_h5_train import *

data, mz_array = load_data()
pixel_count = data.shape[0]
intensity_count = data.shape[1]
idx = np.argmax(np.mean(data, 0))
print(idx)
print(mz_array[idx])
plt.figure(1)
plt.imshow(data[:, idx])
plt.savefig("ae_h5_example.png")
#model, optimizer, loss_function, device = init_model(intensity_count)