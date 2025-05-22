import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    计算交叉熵损失。

    Args:
        y_true (np.array): 真实标签，(N,) 或 (N, C)。
        y_pred (np.array): 模型预测概率，(N, C)。

    Return:
        float: 平均交叉熵损失。
    """
    N = y_pred.shape[0]           # 样本数
    C = y_pred.shape[1]           # 类别数

    # 如果 y_true 是类别索引，转换为独热编码
    if y_true.ndim == 1:
        y_true = np.eye(C)[y_true]

    # 防止 log(0)
    y_pred = np.clip(y_pred, 1e-12, 1.0)

    # 计算交叉熵损失
    loss = -np.sum(y_true * np.log(y_pred))

    # 返回平均损失
    return loss / N
