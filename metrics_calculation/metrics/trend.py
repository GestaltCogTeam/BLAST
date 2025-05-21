import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from scipy import stats
import pymannkendall as mk
import statsmodels.api as sm

def mannkenall(timeseries):
    result = mk.original_test(timeseries)
    return result
