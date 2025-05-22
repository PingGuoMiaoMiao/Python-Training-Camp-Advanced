import numpy as np

def leaky_relu(x, alpha=0.01):
    """
    计算 Leaky ReLU 激活函数。
    对于每个元素 x_i，返回 max(alpha * x_i, x_i)。
    """
    return np.maximum(alpha * x, x)
