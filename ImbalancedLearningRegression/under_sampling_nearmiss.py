## load dependencies - third party
import numpy as np
import pandas as pd
import random as rd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from ImbalancedLearningRegression.dist_metrics import euclidean_dist, heom_dist, overlap_dist


## edit normal observations
def under_sampling_nearmiss(
    
    ## arguments / inputs
    data,           ## training set
    index,          ## index of input data
    rare_indices,    ## indices of samples in the minority set
    version,          ## which version of nearmiss undersampling is being used
    perc              ## percentage of each bin to be undersampled
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
    
    ## create global feature list by column index
    feat_list = list(data.columns.values)
    
    ## create nominal feature list and
    ## label encode nominal / categorical features
    ## (strictly label encode, not one hot encode) 
    feat_list_nom = []
    nom_dtypes = ["object", "bool", "datetime64"]
    
    # Unknown warning, may be handled later
    pd.options.mode.chained_assignment = None

    for j in range(d):
        if data.dtypes[j] in nom_dtypes:
            feat_list_nom.append(j)
            data.iloc[:, j] = pd.Categorical(pd.factorize(
                data.iloc[:, j])[0])
    
    data = data.apply(pd.to_numeric)
    
    ## create numeric feature list
    feat_list_num = list(set(feat_list) - set(feat_list_nom))
    
    ## calculate ranges for numeric / continuous features
    ## (includes label encoded features)
    feat_ranges = list(np.repeat(1, d))
    
    if len(feat_list_nom) > 0:
        for j in feat_list_num:
            feat_ranges[j] = max(data.iloc[:, j]) - min(data.iloc[:, j])
    else:
        for j in range(d):
            feat_ranges[j] = max(data.iloc[:, j]) - min(data.iloc[:, j])
    
    ## subset feature ranges to include only numeric features
    ## (excludes label encoded features)
    feat_ranges_num = [feat_ranges[i] for i in feat_list_num]
    
    ## subset data by either numeric / continuous or nominal / categorical
    data_num = data.iloc[:, feat_list_num]
    data_nom = data.iloc[:, feat_list_nom]
    
    ## get number of features for each data type
    feat_count_num = len(feat_list_num)
    feat_count_nom = len(feat_list_nom)


######################################################################################

    rare = [] # array of the values of all rare indices
    for i in rare_indices:
        if(rare_indices[i] == 1):
            rare.append(data.iloc[i])


    dist_matrix = np.ndarray(shape = (n, n)) # all distances from each majority point to every minority point
    av_dist = []

    ## loop through the majority set
    for i in index:
        for j in rare:
            dist_matrix[i][j] = (abs(data.iloc[i [:,-1]] - rare[i [:,-1]] ), i)


    if version == 1:
        for i in index:
            closest = sorted(dist_matrix[i])[:3]

            av_dist[i] = (closest[0] + closest[1] + closest[2]) / 3 # 3 closest rare values 

    elif version == 2:
        for i in index:
            closest = sorted(dist_matrix[i])[-3:]

            av_dist[i] = (closest[0] + closest[1] + closest[2]) / 3 # 3 farthest rare values

    else:
        for i in index:
            closest = sum(dist_matrix[i]) # all rare values

            av_dist[i] = closest / len(rare)


    np.argsort(av_dist) # sorts average distances by index

    ## indices of results
    chosen_indices = list() #list of samples that will be kept

    n_under = int(len(index)  * perc)   ## number of samples to be removed - referenced from under_sampling_random

    for i in index: # for all indices in majority set
        if i > n_under: # dont select indices that should be removed
            chosen_indices.append(av_dist[i]) # index of majority sample with smallest average distance


    ## conduct under sampling and store modified training set
    data_new = pd.DataFrame()
    data_new = pd.concat([data.iloc[chosen_indices], data_new], ignore_index = True)
    

    #############################################################################################


    ## replace label encoded values with original values
    for j in feat_list_nom:
        code_list = data.iloc[:, j].unique()
        cat_list = data_var.iloc[:, j].unique()
        
        for x in code_list:
            data_new.iloc[:, j] = data_new.iloc[:, j].replace(x, cat_list[x])
    
    ## convert negative values to zero in non-negative features
    for j in feat_non_neg:
        # data_new.iloc[:, j][data_new.iloc[:, j] < 0] = 0
        data_new.iloc[:, j] = data_new.iloc[:, j].clip(lower = 0)
    
    ## return over-sampling results dataframe
    return data_new
