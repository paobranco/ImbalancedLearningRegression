## Third Party Dependencies
from numpy          import nan, random, sort, insert
from pandas         import DataFrame, read_csv
from pandas.testing import assert_series_equal, assert_frame_equal

## Standard Library Dependencies
from pathlib       import Path
from unittest      import TestCase, main
from unittest.mock import patch

## Internal Dependencies
from ImbalancedLearningRegression.utils.phi import phi
from ImbalancedLearningRegression.utils.phi_ctrl_pts import phi_ctrl_pts
from ImbalancedLearningRegression.base import BaseSampler
from ImbalancedLearningRegression.utils.models import SAMPLE_METHOD, RELEVANCE_METHOD, RELEVANCE_XTRM_TYPE

class TestBaseSampler(TestCase):
    def setUp(self) -> None:
        self.p = patch.multiple(BaseSampler, __abstractmethods__=set())
        self.p.start()
        self.base_sampler = BaseSampler() # type: ignore
        return super().setUp()

    def tearDown(self) -> None:
        self.p.stop()
        return super().tearDown()

    def test_validate_type(self) -> None:
        # Test with built-in Types
        self.base_sampler._validate_type(value = 4, dtype = (int, ),                 msg = "Failed _validate_type test with integer as type.")
        self.base_sampler._validate_type(value = True, dtype = (bool, ),             msg = "Failed _validate_type test with boolean as type.")
        self.base_sampler._validate_type(value = -2.37, dtype = (float, ),           msg = "Failed _validate_type test with float as type.")
        self.base_sampler._validate_type(value = [], dtype = (list, ),               msg = "Failed _validate_type test with list as type.")
        self.base_sampler._validate_type(value = DataFrame(), dtype = (DataFrame, ), msg = "Failed _validate_type test with DataFrame as type.")

        # Test with Package Defined Types
        self.base_sampler._validate_type(value = SAMPLE_METHOD.BALANCE,    dtype = (SAMPLE_METHOD, ),       msg = "Failed _validate_type test with SAMPLE_METHOD as type.")
        self.base_sampler._validate_type(value = RELEVANCE_METHOD.AUTO,    dtype = (RELEVANCE_METHOD, ),    msg = "Failed _validate_type test with RELEVANCE_METHOD as type.")
        self.base_sampler._validate_type(value = RELEVANCE_XTRM_TYPE.BOTH, dtype = (RELEVANCE_XTRM_TYPE, ), msg = "Failed _validate_type test with RELEVANCE_XTRM_TYPE as type.")

        # Test Exceptions
        with self.assertRaises(TypeError):
            self.base_sampler._validate_type(value = SAMPLE_METHOD.BALANCE, dtype = (int, ),                 
            msg = "Failed _validate_type exception test with SAMPLE_METHOD compared with int as type.")

        with self.assertRaises(TypeError):
            self.base_sampler._validate_type(value = RELEVANCE_METHOD.AUTO, dtype = (SAMPLE_METHOD, ),       
            msg = "Failed _validate_type exception test with RELEVANCE_METHOD compared with SAMPLE_METHOD as type.")

        with self.assertRaises(TypeError):
            self.base_sampler._validate_type(value = DataFrame(), dtype = (RELEVANCE_XTRM_TYPE, ), 
            msg = "Failed _validate_type exception test with DataFrame compared with RELEVANCE_XTRM_TYPE as type.")

    def test_validate_data(self) -> None:
        self.base_sampler._validate_data(data = DataFrame())
        with self.assertRaises(TypeError):
            self.base_sampler._validate_data(data = 4) # type: ignore

    def test_validate_response_variable(self) -> None:
        # Test expected behavior.
        self.base_sampler._validate_response_variable(data = DataFrame(columns = ["foobar"], dtype = int), response_variable = "foobar")
        # Test response_variable is not a str raises exception.
        with self.assertRaises(TypeError):
            self.base_sampler._validate_response_variable(data = DataFrame(), response_variable = -2.35) # type: ignore
        # Test response_variable is not a column in the DataFrame.
        with self.assertRaises(ValueError):
            self.base_sampler._validate_response_variable(data = DataFrame(), response_variable = "foobar")
        # Test response_variable column is in the DataFrame but the column type is not numeric
        with self.assertRaises(ValueError):
            self.base_sampler._validate_response_variable(data = DataFrame(columns = ["foobar"], dtype = object), response_variable = "foobar")

    def test_preprocess_nan(self) -> None:
        num_rows = 10
        data = {
            "to_drop_1": insert(random.randint(0, 1000, size=num_rows).astype('float64'), 2, nan), 
            "to_keep_1": random.randint(0, 1000, size=num_rows + 1).tolist(), 
            "to_drop_2": insert(random.randint(0, 1000, size=num_rows).astype('float64'), 2, nan), 
            "to_keep_2": random.randint(0, 1000, size=num_rows + 1).tolist(), 
            "to_drop_3": insert(random.randint(0, 1000, size=num_rows).astype('float64'), 2, nan)
        }
        data = DataFrame.from_dict(data = data)
        data = self.base_sampler._preprocess_nan(data = data)
        self.assertListEqual(data.columns.to_list(), ["to_keep_1", "to_keep_2"])

    def test_create_new_data(self) -> None:
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
        data, sorted = self.base_sampler._create_new_data(data = data, response_variable = response_variable)
        self.assertListEqual(data.iloc[:, len(data.columns)-1].to_list(), response_variable_values)
        self.assertListEqual(sorted.tolist(), response_variable_sorted.tolist())

    def test_validate_relevance(self) -> None:
        # Test expected bahvior.
        self.base_sampler._validate_relevance(relevances = [0.5, 0.4])
        # Test list of relevances is only 0s raises exception
        with self.assertRaises(ValueError):
            self.base_sampler._validate_relevance(relevances = [0,0,0,0])
        # Test list of relevances is only 1s raises exception
        with self.assertRaises(ValueError):
            self.base_sampler._validate_relevance(relevances = [1,1,1,1])


    def test_phi_ctrl_pts(self) -> None:
        data = read_csv("https://raw.githubusercontent.com/paobranco/ImbalancedLearningRegression/master/data/College.csv")
        response_variable = "Grad.Rate"
        
        # Validate variables and Pre-Process Data
        self.base_sampler._validate_relevance_method()
        self.base_sampler._validate_data(data = data)
        self.base_sampler._validate_response_variable(data = data, response_variable = response_variable)
        data = self.base_sampler._preprocess_nan(data = data)

        # Manipulate Data
        new_data, response_variable_sorted = self.base_sampler._create_new_data(data = data, response_variable = response_variable)
        relevance_params = phi_ctrl_pts(response_variable = response_variable_sorted)
        method   = RELEVANCE_METHOD.AUTO
        num_pts  = 3
        ctrl_pts = [18.0, 1, 0, 65.0, 0, 0, 100.0, 1, 0]
        self.assertEqual(relevance_params["method"], method)
        self.assertEqual(relevance_params["num_pts"] ,num_pts)
        self.assertListEqual(relevance_params["ctrl_pts"], ctrl_pts)

    def test_phi(self) -> None:
        data = read_csv("https://raw.githubusercontent.com/paobranco/ImbalancedLearningRegression/master/data/College.csv")
        response_variable = "Grad.Rate"
        
        # Validate variables and Pre-Process Data
        self.base_sampler._validate_relevance_method()
        self.base_sampler._validate_data(data = data)
        self.base_sampler._validate_response_variable(data = data, response_variable = response_variable)
        data = self.base_sampler._preprocess_nan(data = data)

        # Manipulate Data
        new_data, response_variable_sorted = self.base_sampler._create_new_data(data = data, response_variable = response_variable)
        relevance_params = phi_ctrl_pts(response_variable = response_variable_sorted)
        relevances       = phi(response_variable = response_variable_sorted, relevance_parameters = relevance_params)

        # Load Expected Relevances and Assert validity relevances
        with Path("ImbalancedLearningRegression/tests/test_phi_expected_values.txt").open('r') as f:
            expected_relevances = [float(line.strip()) for line in f.readlines()]
            self.assertListEqual(relevances, expected_relevances)  

    def test_identify_intervals(self) -> None:
        data = read_csv("https://raw.githubusercontent.com/paobranco/ImbalancedLearningRegression/master/data/College.csv")
        response_variable = "Grad.Rate"
        
        # Validate variables and Pre-Process Data
        self.base_sampler._validate_relevance_method()
        self.base_sampler._validate_data(data = data)
        self.base_sampler._validate_response_variable(data = data, response_variable = response_variable)
        data = self.base_sampler._preprocess_nan(data = data)

        # Manipulate Data
        new_data, response_variable_sorted = self.base_sampler._create_new_data(data = data, response_variable = response_variable)
        relevance_params = phi_ctrl_pts(response_variable = response_variable_sorted)
        relevances       = phi(response_variable = response_variable_sorted, relevance_parameters = relevance_params)
        indicies, perc   = self.base_sampler._identify_intervals(response_variable_sorted = response_variable_sorted, relevances = relevances)
        
        # Load Expected Percentages and Intervals
        expected_perc = [4.389830508474576, 0.44655172413793104, 1.8768115942028984]
        interval_1 = read_csv("ImbalancedLearningRegression/tests/test_identify_intervals_interval_1_expected_values.txt", header = None)
        interval_2 = read_csv("ImbalancedLearningRegression/tests/test_identify_intervals_interval_2_expected_values.txt", header = None)
        interval_3 = read_csv("ImbalancedLearningRegression/tests/test_identify_intervals_interval_3_expected_values.txt", header = None)

        # Assert their Validity
        self.assertListEqual(perc, expected_perc)
        assert_series_equal(indicies[0], interval_1[0], check_index = False, check_names = False)
        assert_series_equal(indicies[1], interval_2[0], check_index = False, check_names = False)
        assert_series_equal(indicies[2], interval_3[0], check_index = False, check_names = False)

    def test_format_new_data(self) -> None:
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
        new_data, _ = self.base_sampler._create_new_data(data = original_data, response_variable = response_variable)
        new_data    = self.base_sampler._format_new_data(new_data = new_data, original_data = original_data, response_variable = response_variable)
        assert_frame_equal(new_data, original_data)

    # Setters Tests

    def test_drop_na_row(self) -> None:
        self.base_sampler.drop_na_row = True
        self.assertTrue(self.base_sampler.drop_na_row)
        self.base_sampler.drop_na_row = False
        self.assertFalse(self.base_sampler.drop_na_row)
        with self.assertRaises(TypeError):
            self.base_sampler.drop_na_row = 4 # type: ignore

    def test_drop_na_col(self) -> None:
        self.base_sampler.drop_na_col = True
        self.assertTrue(self.base_sampler.drop_na_col)
        self.base_sampler.drop_na_col = False
        self.assertFalse(self.base_sampler.drop_na_col)
        with self.assertRaises(TypeError):
            self.base_sampler.drop_na_col = 4 # type: ignore

    def test_samp_method(self) -> None:
        self.base_sampler.samp_method = SAMPLE_METHOD.BALANCE
        self.assertEqual(self.base_sampler.samp_method, SAMPLE_METHOD.BALANCE)
        self.base_sampler.samp_method = SAMPLE_METHOD.EXTREME
        self.assertEqual(self.base_sampler.samp_method, SAMPLE_METHOD.EXTREME)
        with self.assertRaises(TypeError):
            self.base_sampler.samp_method = True # type: ignore

    def test_rel_thres(self) -> None:
        self.base_sampler.rel_thres = 0.6
        self.assertEqual(self.base_sampler.rel_thres, 0.6)
        with self.assertRaises(TypeError):
            self.base_sampler.rel_thres = None # type: ignore
        with self.assertRaises(TypeError):
            self.base_sampler.rel_thres = 1
        with self.assertRaises(ValueError):
            self.base_sampler.rel_thres = -1.2


    def test_rel_method(self) -> None:
        self.base_sampler.rel_method = RELEVANCE_METHOD.AUTO
        self.assertEqual(self.base_sampler.rel_method, RELEVANCE_METHOD.AUTO)
        self.base_sampler.rel_method = RELEVANCE_METHOD.MANUAL
        self.assertEqual(self.base_sampler.rel_method, RELEVANCE_METHOD.MANUAL)
        with self.assertRaises(TypeError):
            self.base_sampler.rel_method = True # type: ignore


    def test_rel_xtrm_type(self) -> None:
        self.base_sampler.rel_xtrm_type = RELEVANCE_XTRM_TYPE.BOTH
        self.assertEqual(self.base_sampler.rel_xtrm_type, RELEVANCE_XTRM_TYPE.BOTH)
        self.base_sampler.rel_xtrm_type = RELEVANCE_XTRM_TYPE.HIGH
        self.assertEqual(self.base_sampler.rel_xtrm_type, RELEVANCE_XTRM_TYPE.HIGH)
        self.base_sampler.rel_xtrm_type = RELEVANCE_XTRM_TYPE.LOW
        self.assertEqual(self.base_sampler.rel_xtrm_type, RELEVANCE_XTRM_TYPE.LOW)
        with self.assertRaises(TypeError):
            self.base_sampler.rel_xtrm_type = True # type: ignore

    def test_rel_coef(self) -> None:
        self.base_sampler.rel_coef = 2.5
        self.assertEqual(self.base_sampler.rel_coef, 2.5)
        self.base_sampler.rel_coef = 1
        self.assertEqual(self.base_sampler.rel_coef, 1)
        with self.assertRaises(TypeError):
            self.base_sampler.rel_thres = True 
        with self.assertRaises(TypeError):
            self.base_sampler.rel_thres = None # type: ignore


    def test_rel_ctrl_pts_rg(self) -> None:
        self.base_sampler.rel_ctrl_pts_rg = None
        self.assertIsNone(self.base_sampler.rel_ctrl_pts_rg)
        self.base_sampler.rel_ctrl_pts_rg = [[1,2,3], [1.1,1.5,-1.5]]
        self.assertEqual(self.base_sampler.rel_ctrl_pts_rg, [[1,2,3], [1.1,1.5,-1.5]])
        with self.assertRaises(TypeError):
            self.base_sampler.rel_ctrl_pts_rg = True # type: ignore
        with self.assertRaises(TypeError):
            self.base_sampler.rel_ctrl_pts_rg = []
        with self.assertRaises(TypeError):
            self.base_sampler.rel_ctrl_pts_rg = [[1,2, False]]
        with self.assertRaises(TypeError):
            self.base_sampler.rel_ctrl_pts_rg = [[1,2], 1, 2] # type: ignore

if __name__ == "__main__":
    main()
