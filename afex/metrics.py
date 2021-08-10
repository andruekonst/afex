import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def mean_absolute_percentage_error(y_true, y_pred):
    epsilon = np.finfo(np.float64).eps
    mape = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), epsilon)
    return np.average(mape)


def calc_metrics(y_true, y_pred):
    return dict(
        r2=r2_score(y_true, y_pred),
        mape=mean_absolute_percentage_error(y_true, y_pred),
        mse=mean_squared_error(y_true, y_pred),
        mae=mean_absolute_error(y_true, y_pred),
    )
