## Third Party Dependencies
from pytest import raises
from pandas import read_csv

## Standard Library Dependencies
from typing        import Any
from unittest.mock import patch, _patch

## Internal Dependencies
from ImbalancedLearningRegression.over_sampling.smote import BaseSMOTE, SMOTE, SVMSMOTE

# BaseSMOTE Tests
def test_validate_neighbours() -> None:
    base_smote, p = create_basesmote()

    destroy_basesmote(p)


# SMOTE Tests
def test_smote_fit_resample() -> None:
    smote = SMOTE()
    data = read_csv("https://raw.githubusercontent.com/paobranco/ImbalancedLearningRegression/master/data/College.csv")
    response_variable = "Grad.Rate"
    
    new_data = smote.fit_resample(data = data, response_variable = response_variable)

def test_smote_validate_neighbours() -> None:
    smote = SMOTE()
    data = read_csv("https://raw.githubusercontent.com/paobranco/ImbalancedLearningRegression/master/data/College.csv")
    smote._validate_neighbours(data = data)
    
    # Test Exception if neighbours > num of points
    with raises(ValueError):
        smote.neighbours = 100000000
        smote._validate_neighbours(data = data)

# Test Setters

def test_smote_neighbours() -> None:
    smote = SMOTE()
    smote.neighbours = 10
    assert smote.neighbours == 10
    smote.neighbours = 5
    assert smote.neighbours == 5
    with raises(TypeError):
        smote.neighbours = "foobar" # type: ignore

# SVMSMOTE Tests
def test_svm_smote_fit_resample() -> None:
    svm_smote = SVMSMOTE()
    data = read_csv("https://raw.githubusercontent.com/paobranco/ImbalancedLearningRegression/master/data/College.csv")
    response_variable = "Grad.Rate"
    
    new_data = svm_smote.fit_resample(data = data, response_variable = response_variable)

def test_svm_smote_validate_neighbours() -> None:
    svm_smote = SVMSMOTE()
    data = read_csv("https://raw.githubusercontent.com/paobranco/ImbalancedLearningRegression/master/data/College.csv")
    svm_smote._validate_neighbours(data = data)
    
    # Test Exception if neighbours > num of points
    with raises(ValueError):
        svm_smote.neighbours = 100000000
        svm_smote._validate_neighbours(data = data)

# Test Setters

def test_svm_smote_neighbours() -> None:
    svm_smote = SVMSMOTE()
    svm_smote.neighbours = 10
    assert svm_smote.neighbours == 10
    svm_smote.neighbours = 5
    assert svm_smote.neighbours == 5
    with raises(TypeError):
        svm_smote.neighbours = "foobar" # type: ignore

# Helper Functions
def create_basesmote() -> tuple[BaseSMOTE, "_patch[Any]"]:
    p = patch.multiple(BaseSMOTE, __abstractmethods__=set())
    p.start()
    base_smote = BaseSMOTE() # type: ignore
    return base_smote, p 

def destroy_basesmote(p: "_patch[Any]") -> None:
    p.stop()