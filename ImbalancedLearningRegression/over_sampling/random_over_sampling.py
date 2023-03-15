from pandas import DataFrame
from abc import abstractmethod
from .base import BaseOverSampler
from ..utils.enums import SAMPLE_METHOD, RELEVANCE_METHOD, RELEVANCE_XTRM_TYPE

class RandomOverSampler(BaseOverSampler):

    def __init__(self, drop_na_row: bool = True, drop_na_col: bool = True, samp_method: SAMPLE_METHOD = SAMPLE_METHOD.BALANCE, 
                 rel_thres: float = 0.5, rel_method: RELEVANCE_METHOD = RELEVANCE_METHOD.AUTO, rel_xtrm_type: RELEVANCE_XTRM_TYPE = RELEVANCE_XTRM_TYPE.BOTH, 
                 rel_coef: float = 1.5, rel_ctrl_pts_rg: list[list[float | int]] | None = None, replace: bool = True, manual_perc: bool = False,
                 perc_oversampling: float = -1) -> None:
        
        super().__init__(drop_na_row = drop_na_row, drop_na_col = drop_na_col, samp_method = samp_method,
        rel_thres = rel_thres, rel_method = rel_method, rel_xtrm_type = rel_xtrm_type, rel_coef = rel_coef, rel_ctrl_pts_rg = rel_ctrl_pts_rg)

        self.replace           = replace
        self.manual_perc       = manual_perc
        self.perc_oversampling = perc_oversampling

    def fit_resample(self):
        pass

    def _oversample(self):
        pass

    def _validate_perc_oversampling(self):
        if self.manual_perc:
            if self.perc_oversampling == -1:
                raise ValueError("cannot proceed: require percentage of over-sampling if manual_perc == True")
            if self.perc_oversampling <= 0:
                raise ValueError("percentage of over-sampling must be a positve real number")

    @property
    def replace(self) -> bool:
        return self._replace

    @replace.setter
    def replace(self, replace: bool) -> None:
        self._validate_type(value = replace, dtype = bool, msg = "replace should be a boolean.")
        self._replace = replace

    @property
    def manual_perc(self) -> bool:
        return self._manual_perc

    @manual_perc.setter
    def manual_perc(self, manual_perc: bool) -> None:
        self._validate_type(value = manual_perc, dtype = bool, msg = "manual_perc should be a boolean.")
        self._manual_perc = manual_perc

    @property
    def perc_oversampling(self) -> float:
        return self._perc_oversampling

    @perc_oversampling.setter
    def perc_oversampling(self, perc_oversampling: float) -> None:
        self._validate_type(value = perc_oversampling, dtype = float, msg = "perc_oversampling should be a float.")
        self._perc_oversampling = perc_oversampling