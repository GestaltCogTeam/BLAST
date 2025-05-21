import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stl.mstl import MSTL
import warnings
warnings.filterwarnings("ignore")
from scipy.fft import fft, ifft

def check_period(data, k=3):
    # 计算快速傅里叶变换（FFT）
    # window = np.hamming(len(data))
    # data = data * window
    fft_result = np.fft.fft(data)
    power_spectrum = np.abs(fft_result)
    frequencies = np.fft.fftfreq(len(data))

    # 只取正频率部分
    positive_frequencies = frequencies[2: len(frequencies) // 2]
    valid_indices = positive_frequencies > 0
    positive_power_spectrum = power_spectrum[2: len(power_spectrum) // 2]
    positive_power_spectrum = positive_power_spectrum[valid_indices]

    # 找到功率谱(幅度)的前k个最大值及其对应的频率
    topk_indices = np.argsort(positive_power_spectrum)[-k:][::-1]
    topk_frequencies = positive_frequencies[topk_indices]
    topk_power_values = positive_power_spectrum[topk_indices]
    topk_periods = np.where(topk_frequencies != 0, 1 / topk_frequencies, np.inf)

    # 将周期进行四舍五入并转换为整数，并去除重复项，同时保持顺序
    topk_periods_rounded = np.round(topk_periods).astype(int)
    _, unique_indices = np.unique(topk_periods_rounded, return_index=True)
    topk_periods_int = topk_periods_rounded[np.sort(unique_indices)]
    filtered_lists = [(x, y, z, v) for x, y, z, v in
                      zip(topk_frequencies, topk_power_values, topk_periods, topk_periods_int)]
    return filtered_lists


def seasonal_strength(data, period, dump=1):
    """
    TODO: 使用该指标
    计算时间序列的季节性强度

    参数:
    data (numpy.ndarray): 输入的时间序列数据
    period (int): 季节周期

    返回:
    seasonal_strength (float): 季节性强度
    """
    if data.shape[0] > 200:
        dump = 5
    series = pd.Series(data)

    # 进行时间序列分解
    stl = STL(series, period=period, trend_jump=dump, seasonal_jump=dump)
    result = stl.fit()

    # 计算季节性强度
    seasonal = result.seasonal
    trend = result.trend
    resid = result.resid

    seasonal_strength = max(0, 1 - np.var(resid) / np.var(seasonal + resid))
    return seasonal_strength

def multiseasonal_strength(data, period, dump=1):
    """
    TODO: 使用该指标
    计算时间序列的季节性强度

    参数:
    data (numpy.ndarray): 输入的时间序列数据
    period (int): 季节周期

    返回:
    seasonal_strength (float): 季节性强度
    """
    if data.shape[0] > 200:
        dump = 5
    series = pd.Series(data)
    # 进行时间序列分解
    stl = MSTL(series, periods=period, stl_kwargs = {'trend_jump': dump, 'seasonal_jump': dump})
    result = stl.fit()

    # 计算季节性强度
    seasonal = np.sum(result.seasonal, axis=1)
    trend = result.trend
    resid = result.resid
    seasonal_strength = max(0, 1 - np.var(resid) / np.var(seasonal + resid))

    return seasonal_strength


def test_seasonality(data, k=3):
    """
        TODO: 使用该指标
        对时间序列进行季节性/周期信息计算。
        通过计算功率谱（幅度谱）的最大值，确定所在的周期，取前K大的周期，K默认为3

        参数:
        timeseries (pd.Series): 时间序列数据

        返回:
        长度为7的输出：
            topk_frequencies, 幅度（能量谱）最大所在的频率
            topk_power_values,  最大的幅度（能量谱）的数值，
            topk_periods, 最大幅度所在的周期，是频率的倒数，为浮点数
            topk_periods_int, 最大幅度所在的周期，转为整数
            ocsb_test_result, ocsb测试数值，因复杂度高而丢弃
            ch_test_result, ch测试数值，因复杂度高而丢弃
            SeasonalStrength：季节性强度（关键） （季节性强度与幅度概念与数值不一样）
            它们的长度为主要周期的数量，<=3

    """

    length = data.shape[0] // 2
    series = pd.Series(data)
    # 进行时间序列分解
    stl = STL(series, period=length)
    result = stl.fit()
    # 计算季节性强度
    seasonal = result.seasonal
    filtered_lists = check_period(seasonal, k = k)

    # 分别提取过滤后的列表
    topk_power_values = [x[1] for x in filtered_lists]
    topk_periods_int = [x[3] for x in filtered_lists]

    filtered_indices0 = [i for i, value in enumerate(topk_periods_int ) if value < length - 1]
    if len(filtered_indices0) == 0:
        Number_periods = 0
        SeasonalStrength = 0
        topk_periods_int2 = []
    else:
        topk_periods_int = [topk_periods_int[i] for i in filtered_indices0]
        topk_power_values = [topk_power_values[i] for i in filtered_indices0]
        max_value = max(topk_power_values)
        threshold = max_value * 0.2
        # 筛选满足条件的元素
        filtered_indices = [i for i, value in enumerate(topk_power_values) if value >= threshold]
        Number_periods = len(filtered_indices)
        topk_periods_int2 = [topk_periods_int[i] for i in filtered_indices]
        # 计算季节性强度
        if len(topk_periods_int2) == 1:
            SeasonalStrength = seasonal_strength(data, topk_periods_int2[0])
        else:
            SeasonalStrength = multiseasonal_strength(data, topk_periods_int2)

    return Number_periods, topk_periods_int2, SeasonalStrength
