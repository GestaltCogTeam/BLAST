import numpy as np
import pandas as pd

def test_volatility(timeseries):
    """
    TODO: 使用该指标
    """
    # 最小归一化，起点为0
    timeseries = timeseries - min(timeseries)
    # 计算均值
    mean = np.mean(timeseries)
    if mean >= 1e-5:
        # 计算方差
        variance = np.var(timeseries)
        # 计算标准差
        std_dev = np.std(timeseries)
        # 计算极差
        range_ = np.ptp(timeseries)
        # 计算平均绝对偏差
        mad = np.mean(np.abs(timeseries - mean))
        # 计算变异系数
        cv = np.around(std_dev / mean, 4)
    else:
        mean = variance = std_dev = range_ = mad = cv = 0
    return mean, variance, std_dev, range_, mad, cv
