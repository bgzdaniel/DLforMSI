from ae_conv_train import *
import pandas as pd
import seaborn as sns

lib = CDLL('libM2aiaCoreIO.so')
image = "/home/dbogacz/Development/pyM2aia/tests/training_data/ew_section2_pos.imzML"
params = "m2PeakPicking.txt"

with M2aiaImageHelper(lib, image, params) as helper:
    gen = helper.SpectrumIterator()
    pixel = next(gen)[2]
    intensity_count = len(pixel)
    I = helper.GetImage(1088.868, 0.249, np.float32)
    label1 = sitk.ReadImage("1088_868_label1.nrrd")
    sitk.WriteImage(I, "1088_868.nrrd")
    label1 = sitk.GetArrayFromImage(label1)
    A = sitk.GetArrayFromImage(I)
    x_size = A.shape[1]
    y_size = A.shape[2]
    print(A.shape)
    print(I.GetSize())
    print(intensity_count)

    latent_size = 16
    model, optimizer, loss_function, device = init_model(latent_size)

    aggregate = [0, np.zeros((intensity_count)), np.zeros((intensity_count))]
    gen = helper.SpectrumIterator()
    for data in gen:
        i, xs, ys = data
        aggregate = update(aggregate, ys)
    im_mean, im_variance, im_sampleVariance = finalize(aggregate)
    im_stddev = np.sqrt(im_sampleVariance)

    data = []
    for i in range(2000):
        id = np.random.randint(0, helper.GetNumberOfSpectra())
        xs, ys = helper.GetSpectrum(id)
        x, y, z = helper.GetPixelPosition(id)
        label = str(label1[z, y, x])
        ys -= im_mean # data zero centering
        ys /= (im_stddev + 1e-10) # data normalization
        ys = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(ys).to(device), 0), 0)
        ys = F.pad(ys, (0, 2307), "constant", 0)
        with torch.no_grad():
            ys = torch.squeeze(model.module.encode(ys), 0).cpu().numpy()
        ys = list(ys)
        ys.append(label)
        data.append(ys)

    columns = ["dim" + str(i+1) for i in range(16)]
    columns.append("label")
    df = pd.DataFrame(data, columns = columns)
    plot = sns.pairplot(df, hue="label")
    fig = plot.fig
    fig.savefig("/home/dbogacz/Development/pyM2aia/tests/worm_conv_scatmat.png")