import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from scipy import stats
import pymannkendall as mk
import statsmodels.api as sm
from hurst import compute_Hc
import statsmodels.stats.api as sms
import fathon
from fathon import fathonUtils as fu


def generate_and_analyze_hurst(data, seed=42):
    """
    TODO: 使用该指标
    生成随机时间序列并计算 Hurst 指数

    参数:
    length (int): 时间序列的长度，默认为10000
    seed (int): 随机种子，默认为42

    返回:
    H (float): 计算得到的 Hurst 指数
    c (numpy.ndarray): 生成的随机时间序列
    data_reg (tuple): Hurst 指数分析的回归数据
    """
    data = data + 2e-5
    # 计算 Hurst 指数
    try:
        H, c, data_reg = compute_Hc(data, kind='change', simplified=False)
    except:
        H = 0
        c = data_reg = None

    return H, c, data_reg
