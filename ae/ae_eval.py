from ae.ae_train import *

lib = CDLL('libM2aiaCoreIO.so')
image = "ew_section1_pos.imzML"
params = "m2PeakPicking.txt"

# eval on another imzML file
with M2aiaImageHelper(lib, image, params) as helper:

    gen = helper.SpectrumIterator()
    pixel = next(gen)[2]
    intensity_count = len(pixel)

    # initialize model
    latent_size = 16
    model, optimizer, loss_function, device = initialize_model(intensity_count, latent_size)

    # calculate mean and variance
    aggregate = [0, np.zeros((intensity_count)), np.zeros((intensity_count))]
    gen = helper.SpectrumIterator()
    for data in gen:
        i, xs, ys = data
        aggregate = update(aggregate, ys)
    im_mean, im_variance, im_sampleVariance = finalize(aggregate)
    im_stddev = np.sqrt(im_sampleVariance)

    I = helper.GetImage(1000, 0.54, np.float32)
    A = sitk.GetArrayFromImage(I)
    x_size = A.shape[1]
    y_size = A.shape[2]

    # save encoded image to .nrrd file
    print(f"saving .nrrd image...")
    imgData = np.zeros((1, x_size, y_size, latent_size))
    gen = helper.SpectrumIterator()
    for data in gen:
        id, xs, ys = data
        x, y, z = helper.GetPixelPosition(id)
        ys -= im_mean # data zero centering
        ys /= (im_stddev + 1e-10) # data normalization
        ys = torch.unsqueeze(torch.from_numpy(ys).to(device), 0)
        with torch.no_grad():
            latent_vector = model.module.encode(ys)
        imgData[z, y, x, :] = latent_vector.cpu().numpy()
    im = sitk.GetImageFromArray(imgData)
    sitk.WriteImage(im, "/home/dbogacz/Development/pyM2aia/tests/worm_section1_eval.nrrd")