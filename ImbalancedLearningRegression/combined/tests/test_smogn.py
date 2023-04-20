## Third Party Dependencies
from pytest import raises
from pandas import read_csv

## Internal Dependencies
from ImbalancedLearningRegression.combined.smogn import SMOGN

# SMOTE Tests
def test_smote_fit_resample() -> None:
    smogn = SMOGN()
    data = read_csv("https://raw.githubusercontent.com/paobranco/ImbalancedLearningRegression/master/data/College.csv")
    response_variable = "Grad.Rate"
    
    new_data = smogn.fit_resample(data = data, response_variable = response_variable)