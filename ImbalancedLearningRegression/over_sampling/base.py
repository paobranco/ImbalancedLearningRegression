## Third Party Dependencies
from pandas import DataFrame, Series

## Standard Library Dependencies
from typing import Any
from abc    import abstractmethod

## Internal Dependencies
from ImbalancedLearningRegression.base         import BaseSampler
from ImbalancedLearningRegression.utils.models import SAMPLE_METHOD, RELEVANCE_METHOD, RELEVANCE_XTRM_TYPE

class BaseOverSampler(BaseSampler):

    def __init__(self, drop_na_row: bool = True, drop_na_col: bool = True, samp_method: SAMPLE_METHOD = SAMPLE_METHOD.BALANCE, 
                 rel_thres: float = 0.5, rel_method: RELEVANCE_METHOD = RELEVANCE_METHOD.AUTO, rel_xtrm_type: RELEVANCE_XTRM_TYPE = RELEVANCE_XTRM_TYPE.BOTH, 
                 rel_coef: float | int = 1.5, rel_ctrl_pts_rg: list[list[float | int]] | None = None) -> None:
        
        super().__init__(drop_na_row = drop_na_row, drop_na_col = drop_na_col, samp_method = samp_method, rel_thres = rel_thres,
        rel_method = rel_method, rel_xtrm_type = rel_xtrm_type, rel_coef = rel_coef, rel_ctrl_pts_rg = rel_ctrl_pts_rg)

    @abstractmethod
    def fit_resample(self, data: DataFrame, response_variable: str) -> DataFrame:
        raise NotImplementedError("BaseOverSampler must never call fit_resample as it's just a base abstract class.")

    @abstractmethod
    def _oversample(self, data: DataFrame, indicies: dict[int, "Series[Any]"], perc: list[float]) -> DataFrame:
        raise NotImplementedError("BaseOverSampler must never call _oversample as it's just a base abstract class.")