import numpy as np


def do_normalization(data):
    """
    :param data:    原始一维数据
    :return:        返回归一化后[0, 1]区间的一维数据
    """
    _range = np.max(data) - np.min(data)

    return (data - np.min(data)) / _range


def normalization(data):
    """
    对二维数组进行归一化
    """
    data_shape = data.shape
    row_num = data_shape[0]
    # 逐行归一化
    data_norm = []
    for row in range(row_num):
        temp_norm = do_normalization(data[row])
        data_norm.append(temp_norm)

    return np.array(data_norm)

