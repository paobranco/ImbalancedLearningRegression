## Third Party Dependencies
from pytest         import raises
from numpy          import nan, random, sort, insert
from pandas         import DataFrame, read_csv, concat
from pandas.testing import assert_series_equal, assert_frame_equal

## Standard Library Dependencies
from typing        import Any
from pathlib       import Path
from unittest.mock import patch, _patch

## Internal Dependencies
from ImbalancedLearningRegression.base  import BaseSampler
from ImbalancedLearningRegression.utils import (
    phi,
    phi_ctrl_pts,
    SAMPLE_METHOD,
    RELEVANCE_METHOD,
    RELEVANCE_XTRM_TYPE
)

# Test _validate_type function with every used built-in, third-party and internally defined types
def test_validate_type() -> None:
    # Create BaseSampler
    base_sampler, p = create_basesampler()

    # Test with built-in Types
    base_sampler._validate_type(value = 4, dtype = (int, ),                 msg = "Failed _validate_type test with integer as type.")
    base_sampler._validate_type(value = True, dtype = (bool, ),             msg = "Failed _validate_type test with boolean as type.")
    base_sampler._validate_type(value = -2.37, dtype = (float, ),           msg = "Failed _validate_type test with float as type.")
    base_sampler._validate_type(value = [], dtype = (list, ),               msg = "Failed _validate_type test with list as type.")
    base_sampler._validate_type(value = DataFrame(), dtype = (DataFrame, ), msg = "Failed _validate_type test with DataFrame as type.")

    # Test with Package Defined Types
    base_sampler._validate_type(value = SAMPLE_METHOD.BALANCE,    dtype = (SAMPLE_METHOD, ),       msg = "Failed _validate_type test with SAMPLE_METHOD as type.")
    base_sampler._validate_type(value = RELEVANCE_METHOD.AUTO,    dtype = (RELEVANCE_METHOD, ),    msg = "Failed _validate_type test with RELEVANCE_METHOD as type.")
    base_sampler._validate_type(value = RELEVANCE_XTRM_TYPE.BOTH, dtype = (RELEVANCE_XTRM_TYPE, ), msg = "Failed _validate_type test with RELEVANCE_XTRM_TYPE as type.")

    # Test Exceptions
    with raises(TypeError):
        base_sampler._validate_type(value = SAMPLE_METHOD.BALANCE, dtype = (int, ),                 
        msg = "Failed _validate_type exception test with SAMPLE_METHOD compared with int as type.")

    with raises(TypeError):
        base_sampler._validate_type(value = RELEVANCE_METHOD.AUTO, dtype = (SAMPLE_METHOD, ),       
        msg = "Failed _validate_type exception test with RELEVANCE_METHOD compared with SAMPLE_METHOD as type.")

    with raises(TypeError):
        base_sampler._validate_type(value = DataFrame(), dtype = (RELEVANCE_XTRM_TYPE, ), 
        msg = "Failed _validate_type exception test with DataFrame compared with RELEVANCE_XTRM_TYPE as type.")

    destroy_basesampler(p)

def test_validate_data() -> None:
    # Create BaseSampler
    base_sampler, p = create_basesampler()

    base_sampler._validate_data(data = DataFrame())
    with raises(TypeError):
        base_sampler._validate_data(data = 4) # type: ignore

    destroy_basesampler(p)

def test_validate_response_variable() -> None:
    # Create BaseSampler
    base_sampler, p = create_basesampler()

    # Test expected behavior.
    base_sampler._validate_response_variable(data = DataFrame(columns = ["foobar"], dtype = int), response_variable = "foobar")
    # Test response_variable is not a str raises exception.
    with raises(TypeError):
        base_sampler._validate_response_variable(data = DataFrame(), response_variable = -2.35) # type: ignore
    # Test response_variable is not a column in the DataFrame.
    with raises(ValueError):
        base_sampler._validate_response_variable(data = DataFrame(), response_variable = "foobar")
    # Test response_variable column is in the DataFrame but the column type is not numeric
    with raises(ValueError):
        base_sampler._validate_response_variable(data = DataFrame(columns = ["foobar"], dtype = object), response_variable = "foobar")

    destroy_basesampler(p)

def test_preprocess_nan() -> None:
    # Create BaseSampler
    base_sampler, p = create_basesampler()

    num_rows = 10
    data = {
        "to_drop_1": insert(random.randint(0, 1000, size=num_rows).astype('float64'), 2, nan), 
        "to_keep_1": random.randint(0, 1000, size=num_rows + 1).tolist(), 
        "to_drop_2": insert(random.randint(0, 1000, size=num_rows).astype('float64'), 2, nan), 
        "to_keep_2": random.randint(0, 1000, size=num_rows + 1).tolist(), 
        "to_drop_3": insert(random.randint(0, 1000, size=num_rows).astype('float64'), 2, nan)
    }
    data = DataFrame.from_dict(data = data)
    data = base_sampler._preprocess_nan(data = data)
    assert data.columns.to_list() == ["to_keep_1", "to_keep_2"]

    destroy_basesampler(p)

def test_create_new_data() -> None:
    # Create BaseSampler
    base_sampler, p = create_basesampler()

    response_variable = "response_variable"
    response_variable_values = random.randint(0, 1000, size=10).tolist()
    response_variable_sorted = sort(response_variable_values)
    response_variable_values.sort()
    data = {
        "a": random.randint(0, 1000, size=10).tolist(), 
        "b": random.randint(0, 1000, size=10).tolist(), 
        "response_variable": response_variable_values,             
        "d": random.randint(0, 1000, size=10).tolist(), 
        "e": random.randint(0, 1000, size=10).tolist()
    }
    data = DataFrame.from_dict(data = data)
    data, sorted = base_sampler._create_new_data(data = data, response_variable = response_variable)
    assert data.iloc[:, len(data.columns)-1].to_list() == response_variable_values
    assert sorted.tolist() == response_variable_sorted.tolist()

    destroy_basesampler(p)

def test_validate_relevance() -> None:
    # Create BaseSampler
    base_sampler, p = create_basesampler()

    # Test expected bahvior.
    base_sampler._validate_relevance(relevances = [0.5, 0.4])
    # Test list of relevances is only 0s raises exception
    with raises(ValueError):
        base_sampler._validate_relevance(relevances = [0,0,0,0])
    # Test list of relevances is only 1s raises exception
    with raises(ValueError):
        base_sampler._validate_relevance(relevances = [1,1,1,1])

    destroy_basesampler(p)

def test_phi_ctrl_pts() -> None:
    # Create BaseSampler
    base_sampler, p = create_basesampler()

    data = read_csv("https://raw.githubusercontent.com/paobranco/ImbalancedLearningRegression/master/data/College.csv")
    response_variable = "Grad.Rate"
    
    # Validate variables and Pre-Process Data
    base_sampler._validate_relevance_method()
    base_sampler._validate_data(data = data)
    base_sampler._validate_response_variable(data = data, response_variable = response_variable)
    data = base_sampler._preprocess_nan(data = data)

    # Manipulate Data
    _ , response_variable_sorted = base_sampler._create_new_data(data = data, response_variable = response_variable)
    relevance_params = phi_ctrl_pts(response_variable = response_variable_sorted)
    method   = RELEVANCE_METHOD.AUTO
    num_pts  = 3
    ctrl_pts = [18.0, 1, 0, 65.0, 0, 0, 100.0, 1, 0]
    assert relevance_params["method"]   == method
    assert relevance_params["num_pts"]  == num_pts
    assert relevance_params["ctrl_pts"] == ctrl_pts

    destroy_basesampler(p)

def test_phi() -> None:
    # Create BaseSampler
    base_sampler, p = create_basesampler()

    data = read_csv("https://raw.githubusercontent.com/paobranco/ImbalancedLearningRegression/master/data/College.csv")
    response_variable = "Grad.Rate"
    
    # Validate variables and Pre-Process Data
    base_sampler._validate_relevance_method()
    base_sampler._validate_data(data = data)
    base_sampler._validate_response_variable(data = data, response_variable = response_variable)
    data = base_sampler._preprocess_nan(data = data)

    # Manipulate Data
    _ , response_variable_sorted = base_sampler._create_new_data(data = data, response_variable = response_variable)
    relevance_params = phi_ctrl_pts(response_variable = response_variable_sorted)
    relevances       = phi(response_variable = response_variable_sorted, relevance_parameters = relevance_params)

    # Load Expected Relevances and Assert validity relevances
    with Path("ImbalancedLearningRegression/tests/test_phi_expected_values.txt").open('r') as f:
        expected_relevances = [float(line.strip()) for line in f.readlines()]
        assert relevances == expected_relevances

    destroy_basesampler(p)

def test_identify_intervals() -> None:
    # Create BaseSampler
    base_sampler, p = create_basesampler()

    data = read_csv("https://raw.githubusercontent.com/paobranco/ImbalancedLearningRegression/master/data/College.csv")
    response_variable = "Grad.Rate"
    
    # Validate variables and Pre-Process Data
    base_sampler._validate_relevance_method()
    base_sampler._validate_data(data = data)
    base_sampler._validate_response_variable(data = data, response_variable = response_variable)
    data = base_sampler._preprocess_nan(data = data)

    # Manipulate Data
    _, response_variable_sorted = base_sampler._create_new_data(data = data, response_variable = response_variable)
    relevance_params = phi_ctrl_pts(response_variable = response_variable_sorted)
    relevances       = phi(response_variable = response_variable_sorted, relevance_parameters = relevance_params)
    intervals, perc  = base_sampler._identify_intervals(response_variable_sorted = response_variable_sorted, relevances = relevances)
    
    # Load Expected Percentages and Intervals
    expected_perc = [4.389830508474576, 0.44655172413793104, 1.8768115942028984]
    interval_1    = read_csv("ImbalancedLearningRegression/tests/test_identify_intervals_interval_1_expected_values.txt", header = None)
    interval_2    = read_csv("ImbalancedLearningRegression/tests/test_identify_intervals_interval_2_expected_values.txt", header = None)
    interval_3    = read_csv("ImbalancedLearningRegression/tests/test_identify_intervals_interval_3_expected_values.txt", header = None)

    # Assert their Validity
    assert perc == expected_perc
    assert_series_equal(intervals[0], interval_1[0], check_index = False, check_names = False)
    assert_series_equal(intervals[1], interval_2[0], check_index = False, check_names = False)
    assert_series_equal(intervals[2], interval_3[0], check_index = False, check_names = False)

    destroy_basesampler(p)

def test_preprocess_synthetic_data() -> None:
    # Create BaseSampler
    base_sampler, p = create_basesampler()

    data = DataFrame({
        'Nomological': ['Alice', 'Bob', 'Charlie', 'David', 'Emily'],
        'Constant':    ['foobar', 'foobar', 'foobar', 'foobar', 'foobar'],
        'Numerical':   [25, 32, 18, 47, 29]})

    expected_pre_numerical_synth_data = DataFrame({
        0: ['Alice', 'Bob', 'Charlie', 'David', 'Emily'],
        1: [25, 32, 18, 47, 29]})

    expected_synth_data = DataFrame({
        0: [0, 1, 2, 3, 4],
        1: [25, 32, 18, 47, 29]})

    synth_data, pre_numerical_synth_data = base_sampler._preprocess_synthetic_data(data = data, indices = data.index)
    
    assert_frame_equal(pre_numerical_synth_data, expected_pre_numerical_synth_data)
    assert_frame_equal(synth_data, expected_synth_data)

    destroy_basesampler(p)

def test_format_synthetic_data() -> None:
    # Create BaseSampler
    base_sampler, p = create_basesampler()

    # Test with no constant column
    data = DataFrame({
        'Nomological': ['Alice', 'Bob', 'Charlie', 'David', 'Emily'],
        'Numerical':   [25, 32, 18, 47, 29]})

    synth_data, pre_numerical_synth_data = base_sampler._preprocess_synthetic_data(data = data, indices = data.index)
    synth_data = base_sampler._format_synthetic_data(data = data, synth_data = synth_data, pre_numerical_processed_data = pre_numerical_synth_data)
    assert_frame_equal(synth_data, data)

    # Test with constant column
    data['Constant'] = ['foobar', 'foobar', 'foobar', 'foobar', 'foobar']

    synth_data, pre_numerical_synth_data = base_sampler._preprocess_synthetic_data(data = data, indices = data.index)
    synth_data = base_sampler._format_synthetic_data(data = data, synth_data = synth_data, pre_numerical_processed_data = pre_numerical_synth_data)
    assert_frame_equal(synth_data, data)

    # Test when a negative integer is added in an otherwise non-negative column
    expected_data = concat([DataFrame({'Nomological': ['Alice'], 'Numerical': [0], 'Constant': ['foobar']}), data])
    synth_data, pre_numerical_synth_data = base_sampler._preprocess_synthetic_data(data = data, indices = data.index)
    synth_data = concat([DataFrame({0: [0], 1: [-5]}), synth_data])
    synth_data = base_sampler._format_synthetic_data(data = data, synth_data = synth_data, pre_numerical_processed_data = pre_numerical_synth_data)
    assert_frame_equal(synth_data, expected_data)

    destroy_basesampler(p)
    
def test_format_new_data() -> None:
    # Create BaseSampler
    base_sampler, p = create_basesampler()

    response_variable = "response_variable"
    response_variable_values = random.randint(0, 1000, size=10).tolist()
    data = {
        "a": random.randint(0, 1000, size=10).tolist(), 
        "b": random.randint(0, 1000, size=10).tolist(), 
        "response_variable": response_variable_values,             
        "d": random.randint(0, 1000, size=10).tolist(), 
        "e": random.randint(0, 1000, size=10).tolist()
    }
    original_data = DataFrame.from_dict(data = data)
    new_data, _ = base_sampler._create_new_data(data = original_data, response_variable = response_variable)
    new_data    = base_sampler._format_new_data(new_data = new_data, original_data = original_data, response_variable = response_variable)
    assert_frame_equal(new_data, original_data)

    destroy_basesampler(p)

def test_fit_resample() -> None:
    # Create BaseSampler
    base_sampler, p = create_basesampler()

    with raises(NotImplementedError):
        base_sampler.fit_resample(data = DataFrame(), response_variable = "foobar")

    destroy_basesampler(p)
    

# Test Setters
def test_drop_na_row() -> None:
    # Create BaseSampler
    base_sampler, p = create_basesampler()

    base_sampler.drop_na_row = True
    assert True == base_sampler.drop_na_row
    base_sampler.drop_na_row = False
    assert False == base_sampler.drop_na_row
    with raises(TypeError):
        base_sampler.drop_na_row = 4 # type: ignore

    destroy_basesampler(p)

def test_drop_na_col() -> None:
    # Create BaseSampler
    base_sampler, p = create_basesampler()

    base_sampler.drop_na_col = True
    assert True == base_sampler.drop_na_col
    base_sampler.drop_na_col = False
    assert False == base_sampler.drop_na_col
    with raises(TypeError):
        base_sampler.drop_na_col = 4 # type: ignore

    destroy_basesampler(p)

def test_samp_method() -> None:
    # Create BaseSampler
    base_sampler, p = create_basesampler()

    base_sampler.samp_method = SAMPLE_METHOD.BALANCE
    assert base_sampler.samp_method == SAMPLE_METHOD.BALANCE
    base_sampler.samp_method = SAMPLE_METHOD.EXTREME
    assert base_sampler.samp_method == SAMPLE_METHOD.EXTREME
    with raises(TypeError):
        base_sampler.samp_method = True # type: ignore

    destroy_basesampler(p)

def test_rel_thres() -> None:
    # Create BaseSampler
    base_sampler, p = create_basesampler()

    base_sampler.rel_thres = 0.6
    assert base_sampler.rel_thres == 0.6
    with raises(TypeError):
        base_sampler.rel_thres = None # type: ignore
    with raises(TypeError):
        base_sampler.rel_thres = 1
    with raises(ValueError):
        base_sampler.rel_thres = -1.2

    destroy_basesampler(p)

def test_rel_method() -> None:
    # Create BaseSampler
    base_sampler, p = create_basesampler()

    base_sampler.rel_method = RELEVANCE_METHOD.AUTO
    assert base_sampler.rel_method == RELEVANCE_METHOD.AUTO
    base_sampler.rel_method = RELEVANCE_METHOD.MANUAL
    assert base_sampler.rel_method == RELEVANCE_METHOD.MANUAL
    with raises(TypeError):
        base_sampler.rel_method = True # type: ignore

    destroy_basesampler(p)

def test_rel_xtrm_type() -> None:
    # Create BaseSampler
    base_sampler, p = create_basesampler()

    base_sampler.rel_xtrm_type = RELEVANCE_XTRM_TYPE.BOTH
    assert base_sampler.rel_xtrm_type == RELEVANCE_XTRM_TYPE.BOTH
    base_sampler.rel_xtrm_type = RELEVANCE_XTRM_TYPE.HIGH
    assert base_sampler.rel_xtrm_type == RELEVANCE_XTRM_TYPE.HIGH
    base_sampler.rel_xtrm_type = RELEVANCE_XTRM_TYPE.LOW
    assert base_sampler.rel_xtrm_type == RELEVANCE_XTRM_TYPE.LOW
    with raises(TypeError):
        base_sampler.rel_xtrm_type = True # type: ignore

    destroy_basesampler(p)

def test_rel_coef() -> None:
    # Create BaseSampler
    base_sampler, p = create_basesampler()

    base_sampler.rel_coef = 2.5
    assert base_sampler.rel_coef == 2.5
    base_sampler.rel_coef = 1
    assert base_sampler.rel_coef == 1
    with raises(TypeError):
        base_sampler.rel_thres = True 
    with raises(TypeError):
        base_sampler.rel_thres = None # type: ignore

    destroy_basesampler(p)

def test_rel_ctrl_pts_rg() -> None:
    # Create BaseSampler
    base_sampler, p = create_basesampler()

    base_sampler.rel_ctrl_pts_rg = None
    assert base_sampler.rel_ctrl_pts_rg == None
    base_sampler.rel_ctrl_pts_rg = [[1,2,3], [1.1,1.5,-1.5]]
    assert base_sampler.rel_ctrl_pts_rg == [[1,2,3], [1.1,1.5,-1.5]]
    with raises(TypeError):
        base_sampler.rel_ctrl_pts_rg = True # type: ignore
    with raises(TypeError):
        base_sampler.rel_ctrl_pts_rg = []
    with raises(TypeError):
        base_sampler.rel_ctrl_pts_rg = [[1,2, False]]
    with raises(TypeError):
        base_sampler.rel_ctrl_pts_rg = [[1,2], 1, 2] # type: ignore

    destroy_basesampler(p)

# Helper Functions
def create_basesampler() -> tuple[BaseSampler, "_patch[Any]"]:
    p = patch.multiple(BaseSampler, __abstractmethods__=set())
    p.start()
    base_sampler = BaseSampler() # type: ignore
    return base_sampler, p 

def destroy_basesampler(p: "_patch[Any]") -> None:
    p.stop()