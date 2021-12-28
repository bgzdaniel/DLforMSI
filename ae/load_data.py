from pyM2aia import M2aiaImageHelper
from ctypes import CDLL
import numpy as np
import SimpleITK as sitk

lib = CDLL('libM2aiaCoreIO.so')
image = "/home/dbogacz/Development/pyM2aia/tests/training_data/ew_section2_pos.imzML"
params = "../m2PeakPicking.txt"

def get_shape():
    with M2aiaImageHelper(lib, image, params) as helper:
        gen = helper.SpectrumIterator()
        pixel = next(gen)[2]
        intensity_count = len(pixel)
        I = helper.GetImage(1000, 0.54, np.float32)
        A = sitk.GetArrayFromImage(I)
        x_size = A.shape[1]
        y_size = A.shape[2]
        return x_size, y_size, intensity_count

def load_data():
    x_size, y_size, intensity_count = get_shape()
    pixel_count = x_size * y_size
    data = np.zeros((pixel_count, intensity_count))
    mz_array = np.zeros(intensity_count)
    xpos = np.zeros(pixel_count)
    ypos = np.zeros(pixel_count)
    with M2aiaImageHelper(lib, image, params) as helper:
        gen = helper.SpectrumIterator()
        xs = None
        for pixel in gen:
            id, xs, ys = pixel
            y, x, z = helper.GetPixelPosition(id)
            data[id, :] = ys
            xpos[id] = x
            ypos[id] = y
        mz_array = xs
        data = data.astype(np.float32)
        mz_array = mz_array.astype(np.float32)
        xpos = xpos.astype(int)
        ypos = ypos.astype(int)
    # data_mean = np.mean(data, 0)
    # data -= data_mean
    data_std = np.std(data, 0)
    data /= (data_std + 1e-10)
    return data, mz_array, xpos, ypos

