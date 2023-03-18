import unittest
from unittest.mock import patch

from ImbalancedLearningRegression.base import BaseSampler
from ImbalancedLearningRegression.utils.enums import SAMPLE_METHOD, RELEVANCE_METHOD, RELEVANCE_XTRM_TYPE

class TestBaseSampler(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        p = patch.multiple(BaseSampler, __abstractmethods__=set())
        p.start()
        cls.base_sampler = BaseSampler() # type: ignore

    def test_preprocess_nan(self) -> None:   
        pass 

    def test_validate_data(self) -> None:
        pass

    def test_validate_response_variable(self) -> None:
        pass

    def test_validate_relevance(self) -> None:
        pass

    def test_create_new_data(self) -> None:
        pass
    
    def test_format_new_data(self) -> None:
        pass


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
    unittest.main()
