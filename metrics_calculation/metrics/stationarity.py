import numpy as np
import warnings
import statsmodels.tsa.stattools as ts
warnings.filterwarnings("ignore")


def test_stationarity(timeseries):
    
    if np.std(timeseries) == 0:
        return 1 # is stationary
    try:
        timeseries2 = np.diff(timeseries)

        result_ori = ts.adfuller(timeseries, regression='ct', autolag='AIC')
        result_ori2 = ts.adfuller(timeseries, regression='ctt', autolag='BIC')

        resultdf = ts.adfuller(timeseries2, regression='n', autolag='AIC')
        resultdf2 = ts.adfuller(timeseries2, regression='c', autolag='BIC')

        adf_resultdf = ts.kpss(timeseries, regression='ct')
        adf_result_ori = ts.kpss(timeseries, regression='c')

        if resultdf[1] < 0.05 and resultdf2[1] < 0.05 and result_ori[1] < 0.05 and result_ori2[1] < 0.05 \
                and adf_resultdf[1] > 0.05 and adf_result_ori[1] > 0.05:
            is_stationarity = 1
        else:
            is_stationarity = 0
    except:
        is_stationarity = 0
    return is_stationarity
