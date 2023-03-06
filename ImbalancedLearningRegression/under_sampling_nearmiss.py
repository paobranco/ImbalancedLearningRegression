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
    version          ## which version of nearmiss undersampling is being used
    ):
    
    """
    under-sample observations and is the primary function underlying the
    under-sampling technique utilized in the higher main function 'nearmiss()', the
    4 step procedure for generating synthetic observations is:
    
    1) pre-processing: label encodes nominal / categorical features, and subsets 
    the training set into two data sets by data type: numeric / continuous, and 
    nominal / categorical
    
    2) under-sampling: ENN, which apply ENN rule to choose a subset of a majority 
    dataset, whose target values agree with K-NN prediction
    
    3) post processing: restores original values for label encoded features, 
    converts any interpolated negative values to zero in the case of non-negative 
    features
    
    returns a pandas dataframe containing synthetic observations of the training
    set which are then returned to the higher main function 'enn()'
    
    ref:
    
    Branco, P., Torgo, L., Ribeiro, R. (2017).
    SMOGN: A Pre-Processing Approach for Imbalanced Regression.
    Proceedings of Machine Learning Research, 74:36-50.
    http://proceedings.mlr.press/v74/branco17a/branco17a.pdf.
    
    Branco, P., Ribeiro, R., Torgo, L. (2017). 
    Package 'UBL'. The Comprehensive R Archive Network (CRAN).
    https://cran.r-project.org/web/packages/UBL/UBL.pdf.
    Branco, P., Torgo, L., & Ribeiro, R. P. (2019). 
    Pre-processing approaches for imbalanced distributions in regression. 
    Neurocomputing, 343, 76-99. 
    https://www.sciencedirect.com/science/article/abs/pii/S0925231219301638
    Wilson, D. L. (1972). 
    Asymptotic properties of nearest neighbor rules using edited data. 
    IEEE Transactions on Systems, Man, and Cybernetics, (3), 408-421.
    https://ieeexplore.ieee.org/abstract/document/4309137
    Kunz, N., (2019). SMOGN. 
    https://github.com/nickkunz/smogn
    """

    ## store dimensions of data subset
    n = len(data)
    d = len(data.columns)
    r_i = len(rare_indices)
    
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

    
    print(rare_indices)


    # undersample points with minimum av distance to 3 nearest neighbors
    if version == 1:
        estimator = KNeighborsClassifier(n_neighbors = 3, n_jobs = 1)

                
    # calculate list of distances to 3 furthest minority samples from majority point, then undersample
    elif version == 2:

        # creats estimator of all n nearest distances, as well as the nearest n-3 distances, then 
        # uses the difference in lists to know the 3 furthest distances
        estimator1 = KNeighborsClassifier(n_neighbors = r_i, n_jobs = 1)
        estimator2 = KNeighborsClassifier(n_neighbors = r_i - 3, n_jobs = 1)
        estimator = list(set(estimator1) - set(estimator2))
        

        """
        dist_matrix = np.ndarray(shape = (n, n))
    
        for i in tqdm(range(n), ascii = True, desc = "dist_matrix"):
            for j in range(n):
                
                ## utilize euclidean distance given that 
                ## data is all numeric / continuous
                if feat_count_nom == 0:
                    dist_matrix[i][j] = euclidean_dist(
                        a = data_num.iloc[i],
                        b = data_num.iloc[j],
                        d = feat_count_num
                    )
                
        furthest = dist_matrix[-3:]
        """



    # undersample points with minimum av distance to all nearest neighbors
    else:
        estimator = KNeighborsClassifier(n_neighbors = r_i, n_jobs = 1)

        


    ## loop through the majority set
    for i in index:
        train_X = data_X[:i] + data_X[i+1:]
        train_y = class_y[:i] + class_y[i+1:]
        min_max_scaler = MinMaxScaler()
        train_X_minmax = min_max_scaler.fit_transform(train_X)
        estimator.fit(train_X_minmax, train_y)
        predict_X = min_max_scaler.transform(data.iloc[i,:(d-1)].values.reshape(1,-1))
        predict_y = estimator.predict(predict_X)
        if predict_y == 0:
            chosen_indices.append(i)

            
    ## indices of results
    chosen_indices = list()

    ## list representations
    data_X = data.iloc[:,:(d-1)].values.tolist()
    class_y = [(1 if i in rare_indices else 0) for i in range(n)]


    ## conduct under sampling and store modified training set
    data_new = pd.DataFrame()
    data_new = pd.concat([data.iloc[chosen_indices], data_new], ignore_index = True)
    
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
