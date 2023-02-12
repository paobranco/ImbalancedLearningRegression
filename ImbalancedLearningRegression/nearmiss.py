## load dependencies - third party
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# load dependencies - internal
from ImbalancedLearningRegression.phi import phi
from ImbalancedLearningRegression.phi_ctrl_pts import phi_ctrl_pts

def nearmiss(
  
   ## main arguments / inputs
    data,                     ## training set (pandas dataframe)
    y,                        ## response variable y by name (string)
    version = 1,              ## version (1, 2, or 3), default 1
    samp_method = "balance",  ## over / under sampling ("balance" or extreme")
    drop_na_col = True,       ## auto drop columns with nan's (bool)
    drop_na_row = True,       ## auto drop rows with nan's (bool)         
  
    ## phi relevance function arguments / inputs
    rel_thres = 0.5,          ## relevance threshold considered rare (pos real)
    rel_method = "auto",      ## relevance method ("auto" or "manual")
    rel_xtrm_type = "both",   ## distribution focus ("high", "low", "both")
    rel_coef = 1.5,           ## coefficient for box plot (pos real)
    rel_ctrl_pts_rg = None,   ## input for "manual" rel method  (2d array)

    ## KNeighborsClassifier attribute
    k = 3,                    ## the number of neighbors used for K-NN
    n_jobs = 1,               ## the number of parallel jobs to run for neighbors search

    ## user-defined KNeighborsClassifier
    k_neighbors_classifier = None  ## user-defined estimator allowing more non-default attributes
                                   ## will ignore k and n_jobs values if not None

):
  
  """
  
  NearMiss-1: Majority class examples with minimum average distance to three closest minority class examples.
  NearMiss-2: Majority class examples with minimum average distance to three furthest minority class examples.
  NearMiss-3: Majority class examples with minimum distance to each minority class example.
  
  """
  
  ## pre-process missing values
    if bool(drop_na_col) == True:
        data = data.dropna(axis = 1)  ## drop columns with nan's
    
    if bool(drop_na_row) == True:
        data = data.dropna(axis = 0)  ## drop rows with nan's
    
    ## quality check for missing values in dataframe
    if data.isnull().values.any():
        raise ValueError("cannot proceed: data cannot contain NaN values")
    
    ## quality check for y
    if isinstance(y, str) is False:
        raise ValueError("cannot proceed: y must be a string")
    
    if y in data.columns.values is False:
        raise ValueError("cannot proceed: y must be an header name (string) \
               found in the dataframe")
    
    ## quality check for sampling method
    if samp_method in ["balance", "extreme"] is False:
        raise ValueError("samp_method must be either: 'balance' or 'extreme'")
    
    ## quality check for relevance threshold parameter
    if rel_thres == None:
        raise ValueError("cannot proceed: relevance threshold required")
    
    if rel_thres > 1 or rel_thres <= 0:
        raise ValueError("rel_thres must be a real number number: 0 < R < 1")

    ## quality check for k
    if type(k) != int or k <= 0:
        raise ValueError("k must be a positive integer")

    ## quality check for n_jobs
    if type(n_jobs) != int:
        raise ValueError("n_jobs must be an integer")
    
    ## store data dimensions
    n = len(data)
    d = len(data.columns)
    
    ## store original data types
    feat_dtypes_orig = [None] * d
    
    for j in range(d):
        feat_dtypes_orig[j] = data.iloc[:, j].dtype
    
    ## determine column position for response variable y
    y_col = data.columns.get_loc(y)
    
    ## move response variable y to last column
    if y_col < d - 1:
        cols = list(range(d))
        cols[y_col], cols[d - 1] = cols[d - 1], cols[y_col]
        data = data[data.columns[cols]]
    
    ## store original feature headers and
    ## encode feature headers to index position
    feat_names = list(data.columns)
    data.columns = range(d)
    
    ## sort response variable y by ascending order
    y = pd.DataFrame(data[d - 1])
    y_sort = y.sort_values(by = d - 1)
    y_sort = y_sort[d - 1]

    ## k-NN classifier
    estimator = KNeighborsClassifier(n_neighbors = k, n_jobs = n_jobs) if k_neighbors_classifier == None else k_neighbors_classifier
    
    ## -------------------------------- phi --------------------------------- ##
    ## calculate parameters for phi relevance function
    ## (see 'phi_ctrl_pts()' function for details)
    phi_params = phi_ctrl_pts(
        
        y = y_sort,                ## y (ascending)
        method = rel_method,       ## defaults "auto" 
        xtrm_type = rel_xtrm_type, ## defaults "both"
        coef = rel_coef,           ## defaults 1.5
        ctrl_pts = rel_ctrl_pts_rg ## user spec
    )
    
    ## calculate the phi relevance function
    ## (see 'phi()' function for details)
    y_phi = phi(
        
        y = y_sort,                ## y (ascending)
        ctrl_pts = phi_params      ## from 'phi_ctrl_pts()'
    )
    
    ## phi relevance quality check
    if all(i == 0 for i in y_phi):
        raise ValueError("redefine phi relevance function: all points are 1")
    
    if all(i == 1 for i in y_phi):
        raise ValueError("redefine phi relevance function: all points are 0")
    ## ---------------------------------------------------------------------- ##

