import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def extract_true_indices(boolean_array):
    """
    Extract the indices and count of True values in a boolean array.

    Parameters:
    boolean_array (numpy array): The input boolean array.

    Returns:
    dict: A dictionary with the count of True values and their indices.
    """
    true_indices = np.where(boolean_array)[0]
    true_count = len(true_indices)
    return {'True Count': true_count, 'Indices': true_indices}

def z_score_detection(time_series, threshold=1.645):
    """
    TODO: 使用该指标
    Z-score detection for outliers in a time series.

    Parameters:
    time_series (numpy array): The input time series data.
    threshold (float): The Z-score threshold for detecting outliers.

    Returns:
    numpy array: Boolean array indicating the presence of outliers.
    """
    mean = np.mean(time_series)
    std = np.std(time_series)
    # 计算上下阈值
    if std != 0:
        z_scores = (time_series - mean) / std
        outliers = np.abs(z_scores) > threshold
        # outlier_indices = np.where(outliers)[0]
        result = np.around(extract_true_indices(outliers)['True Count'] / z_scores.shape[0], 4)
    else:
        outliers = np.zeros_like(time_series)
        result = 0
    return result
