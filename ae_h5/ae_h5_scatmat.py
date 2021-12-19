from ae_h5_train import *
import seaborn as sns
import pandas as pd

data, _ = load_data()
pixel_count = data.shape[0]
intensity_count = data.shape[1]
model, optimizer, loss_function, device = init_model(intensity_count)
idx = np.random.randint(pixel_count, size=(2000))
batch = torch.from_numpy(data[idx, :]).to(device)
encoding = None
with torch.no_grad():
    encoding = model.encode(batch).cpu().numpy().tolist()
columns = ["dim" + str(i+1) for i in range(5)]
df = pd.DataFrame(encoding, columns = columns)
plot = sns.pairplot(df)
fig = plot.fig
fig.savefig("prostate_scatmat.png")