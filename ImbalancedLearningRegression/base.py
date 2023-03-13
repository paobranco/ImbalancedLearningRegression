from abc import ABC, abstractmethod
from pandas import DataFrame

class BaseSampler(ABC):

    def __init__(self, data: DataFrame, response_variable: str, drop_na_row: bool = True, drop_na_col: bool = True,
                 samp_method: str = "balance", rel_thres: float = 0.5, rel_method: str = "auto", rel_xtrm_type: str = "both", 
                 rel_coef: float = 1.5, rel_ctrl_pts_rg: list[list[float]] = None) -> None:
        
        self.data              = data
        self.drop_na_row       = drop_na_row 
        self.drop_na_col       = drop_na_col 
        self.samp_method       = samp_method
        self.response_variable = response_variable

        self.rel_thres       = rel_thres 
        self.rel_method      = rel_method
        self.rel_xtrm_type   = rel_xtrm_type
        self.rel_coef        = rel_coef
        self.rel_ctrl_pts_rg = rel_ctrl_pts_rg

    def _preprocess_nan(self) -> None:
        if self.data.isnull().values.any():
            raise ValueError("cannot proceed: data cannot contain NaN values")
            
        if self.drop_na_col == True:
            self.data = self.data.dropna(axis = 1)  ## drop columns with nan's

        if self.drop_na_row == True:
            self.data = self.data.dropna(axis = 0)  ## drop rows with nan's
        
    def _validate_response_variable(self) -> None:
        if self.response_variable in self.data.columns.values is False:
            raise ValueError("cannot proceed: response_variable must be a header name (string) \
                found in the dataframe")

    @abstractmethod
    def fit_resample(self):
        raise NotImplementedError("BaseSampler must never call fit_resample as it's just a base abstract class.")

    @property
    def data(self) -> DataFrame:
        return self._data

    @data.setter
    def data(self, data: DataFrame) -> None:
        if not isinstance(data, DataFrame):
            raise TypeError("data must be a Pandas Dataframe.")
        self._data = data

    @property
    def samp_method(self) -> str:
        return self._samp_method

    @samp_method.setter
    def samp_method(self, samp_method: str) -> None:
        if not isinstance(samp_method, str):
            raise TypeError("samp_method must be a string.")
        self._samp_method = samp_method

    @property 
    def response_variable(self) -> str:
        return self._response_variable

    @response_variable.setter
    def response_variable(self, response_variable: str) -> None:
        if not isinstance(response_variable, str):
            raise TypeError("response_variable must be a string.")
        self._response_variable = response_variable

    @property 
    def drop_na_row(self) -> bool:
        return self._drop_na_row

    @drop_na_row.setter
    def drop_na_row(self, drop_na_row: bool) -> None:
        if not isinstance(drop_na_row, bool):
            raise TypeError("drop_na_row must be a boolean.")
        self._drop_na_row = drop_na_row

    @property
    def drop_na_col(self) -> bool:
        return self._drop_na_col

    @drop_na_col.setter
    def drop_na_col(self, drop_na_col: bool) -> None:
        if not isinstance(drop_na_col, bool):
            raise TypeError("drop_na_col must be a boolean.")
        self._drop_na_col = drop_na_col

    @property 
    def rel_thres(self) -> float:
        return self._rel_thres

    @rel_thres.setter
    def rel_thres(self, rel_thres: float) -> None:
        if rel_thres == None:
            raise ValueError("cannot proceed: rel_thres is required but is None.")

        if not isinstance(rel_thres, float):
            raise TypeError("rel_thresh must be a float.")
            
        if rel_thres > 1 or rel_thres <= 0:
            raise ValueError("rel_thres must be a real number number: 0 < R < 1")
        self._rel_thres = rel_thres

    @property
    def rel_method(self) -> str:
        return self._rel_method

    @rel_method.setter
    def rel_method(self, rel_method: str) -> None:
        if not isinstance(rel_method, str):
            raise TypeError("rel_method must be a string.")
        self._rel_method = rel_method

    @property
    def rel_xtrm_type(self) -> str:
        return self._rel_xtrm_type

    @rel_xtrm_type.setter
    def rel_xtrm_type(self, rel_xtrm_type: str) -> None:
        if not isinstance(rel_xtrm_type, str):
            raise TypeError("rel_xtrm_type must be a string.")
        self._rel_xtrm_type = rel_xtrm_type

    @property
    def rel_coef(self) -> float:
        return self._rel_coef

    @rel_coef.setter
    def rel_coef(self, rel_coef: float) -> None:
        if not isinstance(rel_coef, float):
            raise TypeError("rel_coef must be a float.")
        self._rel_coef = rel_coef

    @property 
    def rel_ctrl_pts_rg(self) -> list[list[float]]:
        return self._rel_ctrl_pts_rg

    @rel_ctrl_pts_rg.setter
    def rel_ctrl_pts_rg(self, rel_ctrl_pts_rg: list[list[float]]) -> None:
        if not isinstance(rel_ctrl_pts_rg, list[list[float]]):
            raise TypeError("rel_ctrl_pts_rg must be a 2D array of floats.")
        self._rel_ctrl_pts_rg = rel_ctrl_pts_rg