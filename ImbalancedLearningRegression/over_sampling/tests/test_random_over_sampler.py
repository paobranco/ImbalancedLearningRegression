## Third Party Dependencies
from pytest import raises
from pandas import read_csv

## Internal Dependencies
from ImbalancedLearningRegression.over_sampling.random_over_sampler import RandomOverSampler

## TODO: Create tests for _oversample and _random_oversample.
## If test_fit_resample passes, then both methods work as expected
## but for complete unittest coverage, it's a good thing to add.

def test_fit_resample():
    random_over_sampler = RandomOverSampler()
    data = read_csv("https://raw.githubusercontent.com/paobranco/ImbalancedLearningRegression/master/data/College.csv")
    response_variable = "Grad.Rate"
    
    new_data  = random_over_sampler.fit_resample(data = data, response_variable = response_variable)

    diff = new_data.merge(data, how='outer', indicator=True)
    diff = diff.loc[diff['_merge'] == 'left_only']
    diff = diff.drop(columns='_merge')
    assert diff.empty == True, "fit_resample generated a synthetic sample that is not part of the original dataset"

def test_validate_perc_oversampling() -> None:
    random_over_sampler = RandomOverSampler()

    # Test Function is manual_perc == False and perc_oversampling == -1
    random_over_sampler._validate_perc_oversampling()

    random_over_sampler.manual_perc = True
    # Test Exception if manual_perc == True and perc_oversampling == -1
    with raises(ValueError):
        random_over_sampler._validate_perc_oversampling()

    random_over_sampler.perc_oversampling = -0.5
    # Test Exception if manual_perc == True and perc_oversampling <= 0
    with raises(ValueError):
        random_over_sampler._validate_perc_oversampling()

# Test Setters

def test_replace() -> None:
    random_over_sampler = RandomOverSampler()
    random_over_sampler.replace = True
    assert random_over_sampler.replace == True
    random_over_sampler.replace = False
    assert random_over_sampler.replace == False
    with raises(TypeError):
        random_over_sampler.replace = "foobar" # type: ignore

def test_manual_perc() -> None:
    random_over_sampler = RandomOverSampler()
    random_over_sampler.manual_perc = True
    assert random_over_sampler.manual_perc == True
    random_over_sampler.manual_perc = False
    assert random_over_sampler.manual_perc == False
    with raises(TypeError):
        random_over_sampler.manual_perc = "foobar" # type: ignore

def test_perc_oversampling() -> None:
    random_over_sampler = RandomOverSampler()
    random_over_sampler.perc_oversampling = -1
    assert random_over_sampler.perc_oversampling == -1
    random_over_sampler.perc_oversampling = 2.5
    assert random_over_sampler.perc_oversampling == 2.5
    with raises(TypeError):
        random_over_sampler.perc_oversampling = "foobar" # type: ignore