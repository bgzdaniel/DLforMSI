from vae_conv_train import *

lib = CDLL('libM2aiaCoreIO.so')
image = "ew_section3_pos.imzML"
params = "m2PeakPicking.txt"

with M2aiaImageHelper(lib, image, params) as helper:
    gen = helper.SpectrumIterator()
    pixel = next(gen)[2]
    intensity_count = len(pixel)

    latent_size = 16
    model, optimizer, loss_function, device = init_model(latent_size)
    
    print("calculating input mean...")
    aggregate = [0, np.zeros((intensity_count)), np.zeros((intensity_count))]
    gen = helper.SpectrumIterator()
    for data in gen:
        i, xs, ys = data
        aggregate = update(aggregate, ys)
    im_mean, im_variance, im_sampleVariance = finalize(aggregate)
    im_stddev = np.sqrt(im_sampleVariance)

    def spec_out_iterator():
        gen = helper.SpectrumIterator()
        for data in gen:
            id, xs, ys = data
            ys = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(ys).to(device), 0), 0)
            ys = F.pad(ys, (0, 2307), "constant", 0)
            with torch.no_grad():
                rec = model(ys)
                rec = torch.squeeze(torch.squeeze(rec, 0), 0)[0:-2307].cpu().numpy()
                yield rec

    print("calculating output mean...")
    aggregate = [0, np.zeros((intensity_count)), np.zeros((intensity_count))]
    gen = spec_out_iterator()
    for rec in gen:
        aggregate = update(aggregate, rec)
    out_mean, out_variance, out_sampleVariance = finalize(aggregate)
    out_stddev = np.sqrt(out_sampleVariance)

    print(f"input mean: {im_mean}")
    print(f"output mean: {out_mean}")
    diff = np.sqrt(np.power(out_mean - im_mean, 2))
    print(f"diff: {diff}")

    print(im_mean.shape)
    print(out_mean.shape)
    print(diff.shape)

    plt.figure(1)
    plt.plot(im_mean[::1000], label="input mean")
    plt.plot(out_mean[::1000], label="output mean")
    plt.plot(diff[::1000], "r--", label="square error")
    plt.legend()
    plt.savefig("mean_spectrum_diff.png")