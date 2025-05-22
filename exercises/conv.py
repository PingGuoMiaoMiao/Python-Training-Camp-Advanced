import numpy as np

def conv2d(x, kernel):
    """
    执行二维卷积操作 (无填充, 步幅为 1)。

    Args:
        x (np.array): 输入二维数组, 形状 (H, W)。
        kernel (np.array): 卷积核二维数组, 形状 (kH, kW)。

    Return:
        np.array: 卷积结果, 形状 (out_H, out_W)。
                  out_H = H - kH + 1
                  out_W = W - kW + 1
    """
    H, W = x.shape
    kH, kW = kernel.shape

    out_H = H - kH + 1
    out_W = W - kW + 1

    # 初始化输出数组
    out = np.zeros((out_H, out_W))

    # 卷积操作
    for i in range(out_H):
        for j in range(out_W):
            patch = x[i:i+kH, j:j+kW]          # 对应区域
            out[i, j] = np.sum(patch * kernel) # 元素乘积之和

    return out
