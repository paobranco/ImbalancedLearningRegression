## Third Party Dependencies
from pytest import raises
from pandas import read_csv

## Internal Dependencies
from ImbalancedLearningRegression.over_sampling.adasyn import ADASYN

def test_fit_resample() -> None:
    adasyn = ADASYN()
    data = read_csv("https://raw.githubusercontent.com/paobranco/ImbalancedLearningRegression/master/data/College.csv")
    response_variable = "Grad.Rate"
    
    new_data  = adasyn.fit_resample(data = data, response_variable = response_variable)

def test_validate_neighbours() -> None:
    adasyn = ADASYN()
    data = read_csv("https://raw.githubusercontent.com/paobranco/ImbalancedLearningRegression/master/data/College.csv")
    adasyn._validate_neighbours(data = data)
    
    # Test Exception if neighbours > num of points
    with raises(ValueError):
        adasyn.neighbours = 100000000
        adasyn._validate_neighbours(data = data)

# Test Setters

def test_neighbours() -> None:
    adasyn = ADASYN()
    adasyn.neighbours = 10
    assert adasyn.neighbours == 10
    adasyn.neighbours = 5
    assert adasyn.neighbours == 5
    with raises(TypeError):
        adasyn.neighbours = "foobar" # type: ignore