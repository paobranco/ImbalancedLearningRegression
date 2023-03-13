from abc import ABC, abstractmethod
from pandas import DataFrame

class BaseSampler(ABC):

    def __init__(self, data: DataFrame, response_variable: str, drop_na_row: bool, drop_na_col: bool) -> None:
        self.data              = data
        self.drop_na_row       = drop_na_row 
        self.drop_na_col       = drop_na_col 
        self.response_variable = response_variable

    def _preprocess_nan(self) -> None:
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

    def drop_na_col(self, drop_na_col: bool) -> None:
        if not isinstance(drop_na_col, bool):
            raise TypeError("drop_na_col must be a boolean.")
        self._drop_na_col = drop_na_col