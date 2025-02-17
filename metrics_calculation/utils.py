
import numpy as np
from scipy.interpolate import interp1d

def interpolate(data):
    _data = np.where(np.isinf(data), np.nan, data) # replace inf with nan
    valid_indices = np.isfinite(_data)
    x_valid = np.where(valid_indices)[0]
    y_valid = _data[valid_indices]
    interp_data = interp1d(x_valid, y_valid, kind='linear', fill_value='extrapolate')
    return interp_data(np.arange(len(data)))

def get_valid_slice(timeseries):
    # valid index: non-nan and non-zero
    a = (~np.isnan(timeseries))
    b = (timeseries != 0)
    valid_indices = np.where(a & b)[0]
    if len(valid_indices) == 0:
        return np.array([1])  # 如果没有找到有效值，返回空数组
    # 取出第一次和最后一次出现非NaN和非0值的索引
    start_idx = valid_indices[0]
    end_idx = valid_indices[-1] + 1  # 切片时需要包含最后一个索引
    return timeseries[start_idx:end_idx]

def zscore_norm(data): # 标准化
    mean = np.nanmean(data)
    std = np.nanstd(data)
    if std == 0:
        std = 1
    return (data - mean) / std
