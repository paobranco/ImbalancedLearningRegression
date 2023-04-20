## Third Party Dependencies
from pytest import raises
from pandas import read_csv

## Internal Dependencies
from ImbalancedLearningRegression.combined.gaussian_noise import GaussianNoise

# SMOTE Tests
def test_smote_fit_resample() -> None:
    gaussian_noise = GaussianNoise()
    data = read_csv("https://raw.githubusercontent.com/paobranco/ImbalancedLearningRegression/master/data/College.csv")
    response_variable = "Grad.Rate"
    
    new_data = gaussian_noise.fit_resample(data = data, response_variable = response_variable)