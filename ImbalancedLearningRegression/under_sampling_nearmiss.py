## load dependencies - third party
import numpy as np
import pandas as pd
import random as rd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler


## edit normal observations
def under_sampling_nearmiss(
    
    ## arguments / inputs
    data,           ## training set
    index,          ## index of input data
    estimator,      ## KNN classifier
    rare_indices    ## indices of samples in the minority set
    
    ):
    
    """
    
    """

    ## store dimensions of data subset
    n = len(data)
    d = len(data.columns)
    
    ## store original data types
    feat_dtypes_orig = [None] * d
    
    for j in range(d):
        feat_dtypes_orig[j] = data.iloc[:, j].dtype
    
    ## find non-negative numeric features
    feat_non_neg = [] 
    num_dtypes = ["int64", "float64"]
    
    for j in range(d):
        if data.iloc[:, j].dtype in num_dtypes and any(data.iloc[:, j] > 0):
            feat_non_neg.append(j)
    
    ## create copy of data containing variation
    data_var = data.copy()
    
