import numpy as np
import copy

from utils import get_valid_slice, interpolate, zscore_norm
from metrics import test_stationarity, mannkenall, test_seasonality, generate_and_analyze_hurst, lm_test, test_volatility, z_score_detection


def process_time_series(time_series, data_name, node_id, channel, multi_sample=False, segment_size=2039):
    """
    
    Args:
        time_series (np.array): 1d time series data, shape (L,).
        data_name (str): data part name
        node_id (int): node id in the data part
        channel (int): channel id in the data part
    
    Returns:
        dict: 
            - key: data part name_node_id_channel_id
            - values: dict of metrics calculation results
    """
    # test:
    segment_size = 2039

    # get valid slice
    time_series = get_valid_slice(time_series)
    if len(time_series) < 512:
        key = data_name + '_' + str(node_id) + '_' + str(channel)
        return {key: None}

    # NOTE: NaN values may exist in the time series data
    # fill NaN values in the metrics calculation process
    time_series = interpolate(time_series)

    sample_list = []

    if len(time_series) <= segment_size:
        sample_list.append(time_series)
    elif not multi_sample:
        # faster
        idx = np.random.choice(len(time_series) - segment_size, 1)
        idx = [0]
        for i in idx:
            sample_list.append(time_series[i:i+segment_size])
    else:
        if len(time_series) > segment_size and len(time_series) <= 10000:
            idx = np.random.choice(len(time_series) - segment_size, 1)
            for i in idx:
                sample_list.append(time_series[i:i+segment_size])
        elif len(time_series) > 10000 and len(time_series) <= 20000:
            idx = np.random.choice(len(time_series) - segment_size, 2)
            for i in idx:
                sample_list.append(time_series[i:i+segment_size])
        else:
            idx = np.random.choice(len(time_series) - segment_size, 3)
            for i in idx:
                sample_list.append(time_series[i:i+segment_size])

    review_results = []
    for ts in sample_list:
        assert np.isnan(ts).sum()  == 0, 'NaN values exist in the time series data.'
        try:
            review_result = compute_indicators(ts)
        except Exception as e:
            print(e)
            print('Error in computing indicators for time series data.')
            # save ts
            import pickle
            with open('error_ts.pkl', 'wb') as f:
                pickle.dump(ts, f)
            raise e
        review_results.append(review_result)

    # merge review_results:
    # - float: average
    # - int: majority vote
    
    review_result = {}
    # stationarity
    stationarity_list = [review_result['stationarity']['stationarity'] for review_result in review_results]
    unique_values, counts = np.unique(stationarity_list, return_counts=True)
    most_frequent_value = int(unique_values[np.argmax(counts)])
    review_result['stationarity'] = {'stationarity': most_frequent_value}
    # trend
    trend_trend_list = [review_result['trend']['trend'] for review_result in review_results]
    unique_values, counts = np.unique(trend_trend_list, return_counts=True)
    most_frequent_value = int(unique_values[np.argmax(counts)])
    trend_tau_list = [review_result['trend']['tau'] for review_result in review_results]
    review_result['trend'] = {'trend': most_frequent_value, 'tau': np.mean(trend_tau_list)}
    # seasonality
    seasonality_count_list = [review_result['seasonality']['count'] for review_result in review_results]
    unique_values, counts = np.unique(seasonality_count_list, return_counts=True)
    most_frequent_value = int(unique_values[np.argmax(counts)])
    seasonality_freq_list = [review_result['seasonality']['seasonalstrength'] for review_result in review_results]
    review_result['seasonality'] = {'count': most_frequent_value, 'seasonalstrength': np.mean(seasonality_freq_list)}
    # volatility
    volatility_std_list = [review_result['volatility']['cv'] for review_result in review_results]
    volatility_std_raw_list = [review_result['volatility']['cv_raw'] for review_result in review_results]
    review_result['volatility'] = {'cv': np.mean(volatility_std_list), 'cv_raw': np.mean(volatility_std_raw_list)}
    # memory
    memory_hurst_list = [review_result['memory']['hurst'] for review_result in review_results]
    review_result['memory'] = {'hurst': np.mean(memory_hurst_list)}
    # scedasticity
    scedasticity_list = [review_result['scedasticity']['LM'] for review_result in review_results]
    unique_values, counts = np.unique(scedasticity_list, return_counts=True)
    most_frequent_value = int(unique_values[np.argmax(counts)])
    review_result['scedasticity'] = {'LM': most_frequent_value}
    # outliers
    outliers_count_list = [review_result['outliers']['count'] for review_result in review_results]
    review_result['outliers'] = {'count': np.mean(outliers_count_list)}
    
    key = data_name + '_' + str(node_id) + '_' + str(channel)
    return {key: review_result}

def compute_indicators(time_series):

    raw_time_series = copy.deepcopy(time_series)
    time_series = zscore_norm(time_series) ## 11.26改--li

    # 取出非nan的数据
    time_series = time_series.astype(float)
    if np.sum(time_series) == 0:
        time_series += 1e-8

    if np.all(time_series == time_series[0]): # constant
        result_dict = {'stationarity': {'stationarity':1},
                        'trend' : {'trend': 0, 'tau':0},
                        'seasonality' : {'count':0, 'freq':[], 'seasonalstrength':0},
                        'volatility' : {'cv':0, 'cv_raw': 0},
                        'memory' : {'hurst':0},
                        'scedasticity' : {'LM':1},
                        'outliers' : {'count':0}}
        return result_dict

    stationarity = {}
    stationarity['stationarity'] = test_stationarity(copy.deepcopy(time_series))

    trend = {}
    mannkenall_result = list(mannkenall(copy.deepcopy(time_series)))
    if mannkenall_result[0] == 'no trend': trend['trend'] = 0
    elif mannkenall_result[0] == 'increasing': trend['trend'] = 1
    else: trend['trend'] = -1
    trend['tau'] = float(np.around(mannkenall_result[4], 4))

    seasonality = {}
    seasonality_result = test_seasonality(copy.deepcopy(time_series))
    seasonality['count'] = int(seasonality_result[0])
    seasonality['freq'] = [int(i) for i in seasonality_result[1]]
    seasonality['seasonalstrength'] = float(np.around(seasonality_result[2], 4))

    vol_result = test_volatility(copy.deepcopy(time_series))
    vol_result_raw = test_volatility(copy.deepcopy(raw_time_series))
    volatility = {'cv': float(np.around(vol_result[-1],4)), 'cv_raw': float(np.around(vol_result_raw[-1],4))} # NOTE: 对于已经归一化的数据来说，这个数值其实没有太大意义

    hurst = generate_and_analyze_hurst(copy.deepcopy(time_series))[0]
    memory = {'hurst': float(np.around(hurst, 4))}

    sce_res1 = lm_test(copy.deepcopy(time_series))['p-value']
    scedasticity = {'LM': 0 if sce_res1 <= 0.05 else 1}

    count = z_score_detection(copy.deepcopy(time_series))
    outliers = {'count': float(np.around(count, 4))}

    result_dict = {
        'stationarity' :  stationarity,
        'trend' : trend,
        'seasonality' : seasonality,
        'volatility' : volatility,
        'memory' : memory,
        'scedasticity' : scedasticity,
        'outliers' : outliers
    }
    return result_dict
