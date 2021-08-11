import numpy as np


RGB_TO_YUV = np.array([
    [ 0.299,     0.587,     0.114],
    [-0.168736, -0.331264,  0.5],
    [ 0.5,      -0.418688, -0.081312]])
YUV_TO_RGB = np.array([
    [1.0,  0.0,      1.402],
    [1.0, -0.34414, -0.71414],
    [1.0,  1.772,    0.0]])
YUV_OFFSET = np.array([0, 128.0, 128.0]).reshape(1, 1, -1)

def rgb2yuv(im):
    return np.tensordot(im, RGB_TO_YUV, ([2], [1])) + YUV_OFFSET

def yuv2rgb(im):
    return np.tensordot(im.astype(float) - YUV_OFFSET, YUV_TO_RGB, ([2], [1]))