## Third Party Dependencies
from pytest import raises
from pandas import read_csv
from sklearn.neighbors import KNeighborsClassifier

## Internal Dependencies
from ImbalancedLearningRegression.under_sampling.cnn import CNN

def test_fit_resample():
    neighbour_classifier = KNeighborsClassifier(n_neighbors = 1, n_jobs = 1)
    cnn = CNN(neighbour_classifier = neighbour_classifier, rel_thres = 0.75)
    data = read_csv("https://raw.githubusercontent.com/paobranco/ImbalancedLearningRegression/master/data/College.csv")
    response_variable = "Grad.Rate"
    
    new_data  = cnn.fit_resample(data = data, response_variable = response_variable)