from pandas import DataFrame
from abc import abstractmethod
from ..base import BaseSampler

class BaseUnderSampler(BaseSampler):

    def __init__(self, data: DataFrame, response_variable: str, drop_na_row: bool = True, drop_na_col: bool = True, 
                 samp_method: str = "balance", rel_thres: float = 0.5, rel_method: str = "auto", rel_xtrm_type: str = "both", 
                 rel_coef: float = 1.5, rel_ctrl_pts_rg: list[list[float]] = None) -> None:
        
        super.__init__(data = data, response_variable = response_variable, drop_na_row = drop_na_row, drop_na_col = drop_na_col, samp_method = samp_method,
        rel_thres = rel_thres, rel_method = rel_method, rel_xtrm_type = rel_xtrm_type, rel_coef = rel_coef, rel_ctrl_pts_rg = rel_ctrl_pts_rg)

    @abstractmethod
    def _undersample(self):
        pass