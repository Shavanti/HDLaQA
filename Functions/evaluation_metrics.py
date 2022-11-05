from scipy import stats
import numpy as np
import glob
import scipy.io as sio


def regular_metrics(predictions, scores):
    """
    Regular evaluation metrics: SROCC, KROCC, PLCC, RMSE, MAE
    """

    SROCC = stats.spearmanr(predictions, scores)[0]
    PLCC = stats.pearsonr(predictions, scores)[0]

    return SROCC, PLCC














