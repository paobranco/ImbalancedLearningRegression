# Third Party Imports
from pandas import DataFrame

# Standard Library Imports
from unittest      import TestCase, main
from unittest.mock import patch

# Package Module Imports
from ImbalancedLearningRegression.base import BaseSampler
from ImbalancedLearningRegression.utils.enums import SAMPLE_METHOD, RELEVANCE_METHOD, RELEVANCE_XTRM_TYPE

class TestBaseSampler(TestCase):
    def setUp(self) -> None:
        self.p = patch.multiple(BaseSampler, __abstractmethods__=set())
        self.p.start()
        self.base_sampler = BaseSampler() # type: ignore
        return super().setUp()

    def tearDown(self) -> None:
        self.p.stop()
        return super().tearDown()

    def test_preprocess_nan(self) -> None:   
        pass 

    def test_validate_data(self) -> None:
        self.base_sampler._validate_data(data = DataFrame())
        with self.assertRaises(TypeError):
            self.base_sampler._validate_data(data = 4) # type: ignore

    def test_validate_response_variable(self) -> None:
        # Test expected behavior.
        self.base_sampler._validate_response_variable(data = DataFrame(columns = ["foobar"]), response_variable = "foobar")
        # Test response_variable is not a str raises exception.
        with self.assertRaises(TypeError):
            self.base_sampler._validate_response_variable(data = DataFrame(), response_variable = -2.35) # type: ignore
        # Test response_variable is not a column in the DataFrame.
        with self.assertRaises(ValueError):
            self.base_sampler._validate_response_variable(data = DataFrame(), response_variable = "foobar")

    def test_validate_relevance(self) -> None:
        # Test expected bahvior.
        self.base_sampler._validate_relevance(relevances = [0.5, 0.4])
        # Test list of relevances is only 0s raises exception
        with self.assertRaises(ValueError):
            self.base_sampler._validate_relevance(relevances = [0,0,0,0])
        # Test list of relevances is only 1s raises exception
        with self.assertRaises(ValueError):
            self.base_sampler._validate_relevance(relevances = [1,1,1,1])

    def test_classify_data(self) -> None:
        pass

    def test_create_new_data(self) -> None:
        pass

    def test_format_new_data(self) -> None:
        pass

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