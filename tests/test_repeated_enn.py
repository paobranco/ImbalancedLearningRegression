import pandas
from sklearn.neighbors import KNeighborsClassifier
from ImbalancedLearningRegression.repeated_enn import repeated_enn

## user-defined estimator
customized_estimator = KNeighborsClassifier(n_neighbors = 5, leaf_size = 60, metric = "manhattan", n_jobs = 2)

## housing
housing = pandas.read_csv(
    "https://raw.githubusercontent.com/paobranco/ImbalancedLearningRegression/master/data/housing.csv"
)

housing_basic = enn(
    data = housing, 
    y = "SalePrice" 
)

housing_extreme = enn(
    data = housing, 
    y = "SalePrice", 
    samp_method = "extreme"
)

housing_k = enn(
    data = housing, 
    y = "SalePrice", 
    k = 6
)

housing_k_neighbors_classifier = enn(
    data = housing, 
    y = "SalePrice", 
    k_neighbors_classifier = customized_estimator
)


housing_combined = enn(
    data = housing, 
    y = "SalePrice",
    samp_method = "extreme", 
    rel_thres = 0.8,
    k = 6
)
