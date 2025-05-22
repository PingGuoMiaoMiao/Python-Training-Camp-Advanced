import numpy as np

def smooth_l1(x, sigma=1.0):
    sigma2 = sigma ** 2
    abs_x = np.abs(x)
    threshold = 1.0 / sigma2

    loss = np.where(
        abs_x < threshold,
        0.5 * (sigma * x) ** 2,
        abs_x - 0.5 / sigma2
    )
    return loss
