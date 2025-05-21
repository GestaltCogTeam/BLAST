import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_arch

def lm_test(time_series):
    """
    TODO: 使用该指标
    Lagrange Multiplier (LM) test for heteroscedasticity in a time series.

    Parameters:
    time_series (numpy array): The input time series data.

    Returns:
    dict: Test statistic and p-value.
    """
    # Fit an OLS model
    ols_model = sm.OLS(time_series, sm.add_constant(np.arange(len(time_series)))).fit()
    # Perform the LM test
    #print(np.isnan(ols_model.resid))
    where_are_nan = np.isnan(ols_model.resid)
    ols_model.resid[where_are_nan] = 0
    try:
        lm_test_stat, lm_p_value, _, _ = het_arch(ols_model.resid)
    except:
        lm_test_stat, lm_p_value = 0, 0
    return {'LM Test Statistic': lm_test_stat, 'p-value': lm_p_value}

