import numpy as np

def maxpool(x, kernel_size, stride):
    """
    执行二维最大池化操作。
    """
    H, W = x.shape
    out_H = (H - kernel_size) // stride + 1
    out_W = (W - kernel_size) // stride + 1
    out = np.zeros((out_H, out_W))

    for i in range(out_H):
        for j in range(out_W):
            h_start = i * stride
            w_start = j * stride
            window = x[h_start:h_start + kernel_size, w_start:w_start + kernel_size]
            out[i, j] = np.max(window)
    
    return out
