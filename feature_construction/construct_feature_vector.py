import warnings
warnings.filterwarnings("ignore")


import numpy as np


def transform_metrics_to_vector(metrics):
    """将时间序列的指标转化为向量

    Args:
        metrics (dict): 某一条时间序列的指标
        default_value (float, optional): 默认值(TODO: 更多文档). Defaults to 0.0.

    Returns:
        np.array: 向量
    """
    row_list = []
    # 平稳性：
    stationarity = float(metrics.get('stationarity-stationarity')[0])
    stationarity = one_hot_two(stationarity) # 11.26 对平稳性加权，因为这个东西确实不好学，降低后面在降维时的比重
    row_list.append(stationarity)

    # 趋势：
    trend = float(metrics.get('trend-trend')[0])
    trend = one_hot_three(trend)
    row_list.append(trend)
    tau_value = float(metrics.get('trend-tau')[0])
    tau = bin(tau_value, -1, 1, 0.1)
    row_list.append(tau)

    # 11.26 季节性大改
    count = int(metrics.get('seasonality-count')[0])
    seasonalstrength = metrics.get('seasonality-seasonalstrength')[0]
    if abs(tau_value) == 1:
        count = 0
        seasonalstrength = 0
    count -= 1
    count_lst = one_hot_four(count)
    row_list.append(count_lst)
    seasonalstrength_lst = bin(seasonalstrength, 0, 1, 0.1)
    row_list.append(seasonalstrength_lst)

    # 11.26 波动率大改，仅采纳变异系数cv
    '''std = float(metrics.get('volatillity-std', [default_value])[0])
    std = bin(std, 0, 0.5, 0.1) # 11.26 delete
    row_list.append(std)'''
    cv = float(metrics.get('volatility-cv')[0])
    cv = bin(cv, 0, 1.2, 0.2)
    row_list.append(cv)
    '''mad = float(metrics.get('volatillity-mad', [default_value])[0])
    mad = bin(mad, 0, 0.5, 0.1) # 11.26 delete
    row_list.append(mad)'''

    # 记忆性
    hurst = float(metrics.get('memory-hurst')[0])
    hurst = bin(hurst, 0, 1, 0.1)
    row_list.append(hurst)

    # 方差齐异性
    lm = float(metrics.get('scedasticity-LM')[0])
    lm = one_hot_two(lm)
    row_list.append(lm)

    # 异常
    out = float(metrics.get('outliers-count')[0])
    out = bin(out, 0, 0.16, 0.04)
    row_list.append(out)
    row_list = np.concatenate(row_list)
    
    return row_list

def one_hot_two(value):
    """
    独热码，0-1的独热码
    """
    vector = np.array([1, 0]) if value == 0 else np.array([0, 1])
    return vector

def one_hot_three(value):
    """
    独热码，针对-1/0/1的独热码
    """
    if value == -1:
        vector = np.array([1, 0, 0])
    elif value == 0:
        vector = np.array([0, 1, 0])
    else:
        vector = np.array([0, 0, 1])
    return vector

def one_hot_four(value):
    """
    独热码，针对-1/0/1/2的独热码
    """
    if value == -1:
        vector = np.array([1, 0, 0, 0])
    elif value == 0:
        vector = np.array([0, 1, 0, 0])
    elif value == 1:
        vector = np.array([0, 0, 1, 0])
    elif value == 2:
        vector = np.array([0, 0, 0, 1])
    else:
        raise ValueError("Invalid value for one_hot_four")
    return vector

def bin(value, th1 = 0.0, th2 = 1.0, grid = 0.1, weights_action = False):
    """
    自定义分箱，输入最小值（下界），最大值（上界），粒度，是否加权的操作
    """
    bins = np.arange(th1, th2, grid)

    # 使用digitize函数将值分箱，并处理异常情况
    if value < th1:
        bin_index = 0
    elif value > th2:
        bin_index = len(bins) - 1
    else:
        bin_index = np.digitize(value, bins) - 1  # digitize返回的索引从1开始，所以减1

    # 创建一个高维数组，长度为分箱的数量
    vector = np.zeros(len(bins))

    # 将对应的分箱位置设为1
    vector[bin_index] = 1
    if weights_action:
        weights = (th2 - th1) / grid
        vector /= weights
    return vector

def deal_with_seasonality(count, freq, seasonalstrength):
    indices_to_remove = [i for i, value in enumerate(seasonalstrength) if value == 1]
    # 季节性强度的值为1说明是窗口
    # 反向遍历索引列表并删除对应位置的元素
    for index in sorted(indices_to_remove, reverse=True):
        del seasonalstrength[index]
        del freq[index]
    # 更新count
    count = count - len(indices_to_remove)
    data = [count, freq, seasonalstrength]
    return data
