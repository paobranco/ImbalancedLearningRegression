## Third Party Dependencies
from pytest import raises
from pandas import read_csv

## Internal Dependencies
from ImbalancedLearningRegression.under_sampling.random_under_sampler import RandomUnderSampler

def test_fit_resample():
    random_under_sampler = RandomUnderSampler()
    data = read_csv("https://raw.githubusercontent.com/paobranco/ImbalancedLearningRegression/master/data/College.csv")
    response_variable = "Grad.Rate"
    
    new_data  = random_under_sampler.fit_resample(data = data, response_variable = response_variable)

    diff = new_data.merge(data, how='outer', indicator=True)
    diff = diff.loc[diff['_merge'] == 'left_only']
    diff = diff.drop(columns='_merge')
    assert diff.empty == True, "fit_resample generated a synthetic sample that is not part of the original dataset"