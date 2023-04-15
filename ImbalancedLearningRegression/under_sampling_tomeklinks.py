## load dependencies - third party
import numpy as np
import pandas as pd
import random as rd
from tqdm import tqdm

## load dependencies - internal
from ImbalancedLearningRegression.utils.dist_metrics import euclidean_dist, heom_dist, overlap_dist

## under-sampling by removing Tomek’s links
def under_sampling_tomeklinks(
    
    ## arguments / inputs
    data,       ## training set
    label,      ## label for each observation in the dataset
    option,     ## user's choice of undersampling which class(es). Default is set to "majority"
                ## option = "both" if user wants to undersample majority and minority classes
    k = 1       ## num of neighs for over-sampling. k is constant 1 since tomeklinks search for nearest neighbour only.
    ):
    
    """
    Removing observations that are considered as Tomek’s links. It is the primary function 
    underlying the under-sampling technique utilized in the higher main function 'tomeklinks()', 
    the 4 step procedure for removing Tomek’s links. is:
    
    1) pre-processing: temporarily removes features without variation, label 
    encodes nominal / categorical features, and subsets the training set into 
    two data sets by data type: numeric / continuous, and nominal / categorical
    
    2) distances: calculates the cartesian distances between all observations, 
    distance metric automatically determined by data type (euclidean distance 
    for numeric only data, heom distance for both numeric and nominal data, and 
    hamming distance for nominal only data) and determine k nearest neighbors
    
    3) under-sampling: 'tomeklinks' is used to search if two observations are both the 
    nearest neighbor of each other and if the two observations belong to different classes
    
    'tomeklinks' only applies to numeric / continuous features, 
    for nominal / categorical features, synthetic values are generated at random 
    from sampling observed values found within the same feature
    
    4) post processing: restores original values for label encoded features, 
    reintroduces constant features previously removed, converts any interpolated
    negative values to zero in the case of non-negative features
    
    returns a pandas dataframe containing synthetic observations of the training
    set which are then returned to the higher main function 'tomeklinks()'
    
    ref:
    
    Branco, P., Torgo, L., Ribeiro, R. (2017).
    SMOGN: A Pre-Processing Approach for Imbalanced Regression.
    Proceedings of Machine Learning Research, 74:36-50.
    http://proceedings.mlr.press/v74/branco17a/branco17a.pdf.
    
    Branco, P., Ribeiro, R., Torgo, L. (2017). 
    Package 'UBL'. The Comprehensive R Archive Network (CRAN).
    https://cran.r-project.org/web/packages/UBL/UBL.pdf.
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
    
    ## find features without variation (constant features)
    feat_const = data.columns[data.nunique() == 1]
    
    ## temporarily remove constant features
    if len(feat_const) > 0:
        
        ## create copy of orignal data and omit constant features
        data_orig = data.copy()
        data = data.drop(data.columns[feat_const], axis = 1)
        
        ## store list of features with variation
        feat_var = list(data.columns.values)
        
        ## reindex features with variation
        for i in range(d - len(feat_const)):
            data.rename(columns = {
                data.columns[i]: i
                }, inplace = True)
        
        ## store new dimension of feature space
        d = len(data.columns)
    
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
    
    ## calculate distance between observations based on data types
    ## store results over null distance matrix of n x n
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
            
            ## utilize heom distance given that 
            ## data contains both numeric / continuous 
            ## and nominal / categorical
            if feat_count_nom > 0 and feat_count_num > 0:
                dist_matrix[i][j] = heom_dist(
                    
                    ## numeric inputs
                    a_num = data_num.iloc[i],
                    b_num = data_num.iloc[j],
                    d_num = feat_count_num,
                    ranges_num = feat_ranges_num,
                    
                    ## nominal inputs
                    a_nom = data_nom.iloc[i],
                    b_nom = data_nom.iloc[j],
                    d_nom = feat_count_nom
                )
            
            ## utilize hamming distance given that 
            ## data is all nominal / categorical
            if feat_count_num == 0:
                dist_matrix[i][j] = overlap_dist(
                    a = data_nom.iloc[i],
                    b = data_nom.iloc[j],
                    d = feat_count_nom
                )
    
    ## determine indicies of k nearest neighbors (k always equal to 1)
    ## and convert knn index list to matrix
    knn_index = [None] * n
    
    for i in range(n):
        knn_index[i] = np.argsort(dist_matrix[i])[0:k + 1]
    
    knn_matrix = np.array(knn_index)

    ## store indicies where two observations are each other's nearest neighbor
    temp = []
    for i in range(n):
        for j in range(n):
            if knn_matrix[i][0] == knn_matrix[j][1] and knn_matrix[i][1] == knn_matrix[j][0]:
                temp.append(knn_matrix[i])

    ## store indicies that belong to tomeklinks AND majority class to tomeklink_majority
    ## store indicies that belong to tomeklinks AND minority class to tomeklink_minority
    tomeklink_majority = []
    tomeklink_minority = []
    for i in range(len(temp)):
        if label[temp[i][0]] != label[temp[i][1]]:
            if label[temp[i][0]] == -1:
                tomeklink_majority.append(temp[i][0])
            else:
                tomeklink_minority.append(temp[i][0])

    ## find all index of the dataset
    all_index = []
    for i in range(n):
        all_index.append(i)

    ## find the index to be undersampled according to user's choice
    remove_index = []
    if option == "majority" or option == "not_minority":
        remove_index = tomeklink_majority
    
    if option == "minority" or option == "not_majority":
        remove_index = tomeklink_minority
    
    if option == "both" or option == "all":
        remove_index = tomeklink_majority + tomeklink_minority


    ## find the non-intersecting values of all_index and remove_index  
    new_index = np.setxor1d(all_index, remove_index)

    ## create null matrix to store new synthetic observations
    synth_matrix = np.ndarray(shape = (len(new_index), d))
    
    # added
    ## store data in the synthetic matrix
    count = 0 
    for i in tqdm(new_index, ascii = True, desc = "new_index"):
        for attr in range(d):
            synth_matrix[count, attr] = (data.iloc[i, attr])
        count = count + 1
        

    ## convert synthetic matrix to dataframe
    data_new = pd.DataFrame(synth_matrix)
    
    ## synthetic data quality check
    if sum(data_new.isnull().sum()) > 0:
        raise ValueError("oops! synthetic data contains missing values")
    
    ## replace label encoded values with original values
    for j in feat_list_nom:
        code_list = data.iloc[:, j].unique()
        cat_list = data_var.iloc[:, j].unique()
        
        for x in code_list:
            data_new.iloc[:, j] = data_new.iloc[:, j].replace(x, cat_list[x])
    
    ## reintroduce constant features previously removed
    if len(feat_const) > 0:
        data_new.columns = feat_var
        
        for j in range(len(feat_const)):
            data_new.insert(
                loc = int(feat_const[j]),
                column = feat_const[j], 
                value = np.repeat(
                    data_orig.iloc[0, feat_const[j]], 
                    len(synth_matrix))
            )
    
    ## convert negative values to zero in non-negative features
    for j in feat_non_neg:
        # data_new.iloc[:, j][data_new.iloc[:, j] < 0] = 0
        data_new.iloc[:, j] = data_new.iloc[:, j].clip(lower = 0)
    
    ## return under-sampling results dataframe
    return data_new
