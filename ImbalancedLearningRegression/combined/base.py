## Third Party Dependencies
from pandas import DataFrame, Series

## Standard Library Dependencies
from typing import Any
from abc    import abstractmethod

## Internal Dependencies
from ImbalancedLearningRegression.base         import BaseSampler
from ImbalancedLearningRegression.utils.models import SAMPLE_METHOD, RELEVANCE_METHOD, RELEVANCE_XTRM_TYPE

class BaseCombinedSampler(BaseSampler):

    def __init__(self, drop_na_row: bool = True, drop_na_col: bool = True, samp_method: SAMPLE_METHOD = SAMPLE_METHOD.BALANCE, 
                 rel_thres: float = 0.5, rel_method: RELEVANCE_METHOD = RELEVANCE_METHOD.AUTO, rel_xtrm_type: RELEVANCE_XTRM_TYPE = RELEVANCE_XTRM_TYPE.BOTH, 
                 rel_coef: float | int = 1.5, rel_ctrl_pts_rg: list[list[float | int]] | None = None, under_samp: bool = True, 
                 pert: int | float = 0.02, replace: bool = False) -> None:
        
        super().__init__(drop_na_row = drop_na_row, drop_na_col = drop_na_col, samp_method = samp_method, rel_thres = rel_thres,
        rel_method = rel_method, rel_xtrm_type = rel_xtrm_type, rel_coef = rel_coef, rel_ctrl_pts_rg = rel_ctrl_pts_rg)

        self.under_samp = under_samp
        self.pert       = pert
        self.replace    = replace

    @abstractmethod
    def fit_resample(self, data: DataFrame, response_variable: str) -> DataFrame:
        raise NotImplementedError("BaseCombinedSampler must never call fit_resample as it's just a base abstract class.")

    @abstractmethod
    def _sample(self, data: DataFrame, indices: dict[int, "Series[Any]"], perc: list[float]) -> DataFrame:
        raise NotImplementedError("BaseCombinedSampler must never call _sample as it's just a base abstract class.")

    # Define Setters and Getters for BaseCombinedSampler

    @property
    def under_samp(self) -> bool:
        return self._under_samp

    @under_samp.setter
    def under_samp(self, under_samp: bool) -> None:
        self._validate_type(value = under_samp, dtype = (bool, ), msg = f"under_samp should be a bool. Passed: {under_samp}")
        self._under_samp = under_samp

    @property
    def pert(self) -> int | float:
        return self._pert

    @pert.setter
    def pert(self, pert: int | float) -> None:
        self._validate_type(value = pert, dtype = (int, float), msg = f"pert should be a bool. Passed: {pert}")
        if pert > 1 or pert <= 0:
            raise ValueError("pert must be a real number number: 0 < R < 1")
        self._pert = pert     

    @property
    def replace(self) -> bool:
        return self._replace

    @replace.setter
    def replace(self, replace: bool) -> None:
        self._validate_type(value = replace, dtype = (bool, ), msg = f"replace should be a boolean. Passed: {replace}")
        self._replace = replace