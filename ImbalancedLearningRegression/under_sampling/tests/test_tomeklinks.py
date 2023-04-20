## Third Party Dependencies
from pytest import raises
from pandas import read_csv

## Internal Dependencies
from ImbalancedLearningRegression.under_sampling.tomeklinks import TomekLinks

def test_fit_resample():
    tomeklinks = TomekLinks()
    data = read_csv("https://raw.githubusercontent.com/paobranco/ImbalancedLearningRegression/master/data/College.csv")
    response_variable = "Grad.Rate"
    
    new_data = tomeklinks.fit_resample(data = data, response_variable = response_variable)