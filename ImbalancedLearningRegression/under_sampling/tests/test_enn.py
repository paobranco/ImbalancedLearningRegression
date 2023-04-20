## Third Party Dependencies
from pytest import raises
from pandas import read_csv
from sklearn.neighbors import KNeighborsClassifier

## Internal Dependencies
from ImbalancedLearningRegression.under_sampling.enn import ENN, RepeatedENN

def test_repeated_enn_fit_resample():
    neighbour_classifier = KNeighborsClassifier(n_neighbors = 1, n_jobs = 1)
    repeated_enn = RepeatedENN(neighbour_classifier = neighbour_classifier)
    data = read_csv("https://raw.githubusercontent.com/paobranco/ImbalancedLearningRegression/master/data/College.csv")
    response_variable = "Grad.Rate"
    
    new_data = repeated_enn.fit_resample(data = data, response_variable = response_variable)

def test_enn_fit_resample():
    neighbour_classifier = KNeighborsClassifier(n_neighbors = 1, n_jobs = 1)
    enn = ENN(neighbour_classifier = neighbour_classifier)
    data = read_csv("https://raw.githubusercontent.com/paobranco/ImbalancedLearningRegression/master/data/College.csv")
    response_variable = "Grad.Rate"
    
    new_data = enn.fit_resample(data = data, response_variable = response_variable)