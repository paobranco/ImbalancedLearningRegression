## Third Party Dependencies
from pandas           import DataFrame, Series
from pandas.api.types import is_numeric_dtype

## Standard Library Dependencies
from typing import Any
from abc    import ABC, abstractmethod

## Internal Dependencies
from ImbalancedLearningRegression.utils.models import SAMPLE_METHOD, RELEVANCE_METHOD, RELEVANCE_XTRM_TYPE

class BaseSampler(ABC):

    def __init__(self, drop_na_row: bool = True, drop_na_col: bool = True, samp_method: SAMPLE_METHOD = SAMPLE_METHOD.BALANCE, 
                 rel_thres: float = 0.5, rel_method: RELEVANCE_METHOD = RELEVANCE_METHOD.AUTO, rel_xtrm_type: RELEVANCE_XTRM_TYPE = RELEVANCE_XTRM_TYPE.BOTH, 
                 rel_coef: float | int = 1.5, rel_ctrl_pts_rg: list[list[float | int]] | None = None) -> None:
        
        self.drop_na_row     = drop_na_row 
        self.drop_na_col     = drop_na_col 
        self.samp_method     = samp_method

        self.rel_thres       = rel_thres 
        self.rel_method      = rel_method
        self.rel_xtrm_type   = rel_xtrm_type
        self.rel_coef        = rel_coef
        self.rel_ctrl_pts_rg = rel_ctrl_pts_rg

    def _validate_type(self, value: Any, dtype: tuple[type, ...], msg: str) -> None:
        if type(value) not in dtype:
            raise TypeError(msg)

    def _validate_relevance_method(self):
        if self.rel_method == RELEVANCE_METHOD.MANUAL and self.rel_ctrl_pts_rg is None:
            raise ValueError("rel_ctrl_pts_rg cannot be None while using a manual relevance method.")

    def _validate_data(self, data: DataFrame) -> None:
        self._validate_type(value = data, dtype = (DataFrame, ), msg = "data must be a Pandas Dataframe.")    

    def _validate_response_variable(self, data: DataFrame, response_variable: str) -> None:
        self._validate_data(data = data)
        self._validate_type(value = response_variable, dtype = (str, ), msg = "response_variable must be a string.")

        if not response_variable in data.columns.values:
            raise ValueError("response_variable must be a header name (string) found in the dataframe")

        if not is_numeric_dtype(data[response_variable]):
            raise ValueError("response_variable column in the dataframe must be specified and numeric.")       

    def _preprocess_nan(self, data: DataFrame) -> DataFrame:   
        if self.drop_na_col == True:
            data = data.dropna(axis = 1)  ## drop columns with nan's

        if self.drop_na_row == True:
            data = data.dropna(axis = 0)  ## drop rows with nan's

        if data.isnull().values.any():
            raise ValueError("cannot proceed: data cannot contain NaN values")

        return data

    def _create_new_data(self, data: DataFrame, response_variable: str) -> tuple[DataFrame, "Series[Any]"]:
        ## determine column position for response variable
        response_col_pos = data.columns.get_loc(response_variable)

        ## move response variable to last column
        if response_col_pos < len(data.columns) - 1:
            cols = list(data.columns)
            cols[response_col_pos], cols[len(data.columns) - 1] = cols[len(data.columns) - 1], cols[response_col_pos]
            data = data[cols]

        ## store original feature headers and
        ## encode feature headers to index position
        data.columns = [str(num) for num in range(len(data.columns))]

        ## sort response variable by ascending order
        response_col = DataFrame(data[str(len(data.columns) - 1)])
        response_col_sorted = response_col.sort_values(by = str(data.columns[len(data.columns) - 1]))
        response_col_sorted = response_col_sorted[str(len(data.columns) - 1)]
        
        return data, response_col_sorted     

    def _validate_relevance(self, relevances: list[float]) -> None:
        if all(i == 0 for i in relevances):
            raise ValueError("redefine phi relevance function: all points are 1")

        if all(i == 1 for i in relevances):
            raise ValueError("redefine phi relevance function: all points are 0")   

    def _identify_intervals(self, response_variable_sorted: "Series[Any]", relevances: list[float]):
        ## determine bin (rare or normal) by interval classification
        intervals = [0]

        for i in range(0, len(response_variable_sorted) - 1):
            if ((relevances[i] >= self.rel_thres and relevances[i + 1] < self.rel_thres) or 
            (relevances[i] < self.rel_thres and relevances[i + 1] >= self.rel_thres)):
                intervals.append(i + 1)

        intervals.append(len(response_variable_sorted))

        ## determine indicies for each interval classification
        interval_indicies = {}

        for i in range(len(intervals) - 1):
            interval_indicies.update({i: response_variable_sorted.iloc[intervals[i]:intervals[i + 1]]})

        ## calculate over / under sampling percentage according to
        ## bump class and user specified method ("balance" or "extreme")
        samples_to_intervals = round(len(response_variable_sorted) / (len(intervals) - 1))
        s_perc = []
        scale = []
        obj = []
        
        if self.samp_method == SAMPLE_METHOD.BALANCE:
            for i in interval_indicies:
                s_perc.append(samples_to_intervals / len(interval_indicies[i]))
                
        if self.samp_method == SAMPLE_METHOD.EXTREME:
            for i in interval_indicies:
                scale.append(samples_to_intervals ** 2 / len(interval_indicies[i]))
            scale = (len(intervals) - 1) * samples_to_intervals / sum(scale)
            
            for i in interval_indicies:
                obj.append(round(samples_to_intervals ** 2 / len(interval_indicies[i]) * scale, 2))
                s_perc.append(round(obj[i] / len(interval_indicies[i]), 1))

        return interval_indicies, s_perc

    def _format_new_data(self, new_data: DataFrame, original_data: DataFrame, response_variable: str) -> DataFrame:
        original_dtypes = [original_data.iloc[:, j].dtype for j in range(len(original_data.columns))]
        response_col_pos = original_data.columns.get_loc(response_variable)
        
        ## restore response variable y to original position
        if response_col_pos < len(original_data.columns) - 1:
            cols = [str(num) for num in range(len(original_data.columns))]
            cols[response_col_pos], cols[len(original_data.columns) - 1] = cols[len(original_data.columns) - 1], cols[response_col_pos]
            new_data = new_data[cols]

        ## rename feature headers to originals
        new_data.columns = list(original_data.columns)
        
        ## restore original data types
        for j in range(len(original_data.columns)):
            new_data.iloc[:, j] = new_data.iloc[:, j].astype(original_dtypes[j])
        
        ## return modified training set
        return new_data

    @abstractmethod
    def fit_resample(self):
        raise NotImplementedError("BaseSampler must never call fit_resample as it's just a base abstract class.")

    # Define Setters and Getters for BaseSampler

    @property
    def samp_method(self) -> SAMPLE_METHOD:
        return self._samp_method

    @samp_method.setter
    def samp_method(self, samp_method: SAMPLE_METHOD) -> None:
        self._validate_type(value = samp_method, dtype = (SAMPLE_METHOD, ), msg = f"samp_method must be an enum of type SAMPLE_METHOD. Passed: '{samp_method}'")
        self._samp_method = samp_method

    @property 
    def drop_na_row(self) -> bool:
        return self._drop_na_row

    @drop_na_row.setter
    def drop_na_row(self, drop_na_row: bool) -> None:
        self._validate_type(value = drop_na_row, dtype = (bool, ), msg = f"drop_na_row must be a boolean. Passed: '{drop_na_row}'")
        self._drop_na_row = drop_na_row

    @property
    def drop_na_col(self) -> bool:
        return self._drop_na_col

    @drop_na_col.setter
    def drop_na_col(self, drop_na_col: bool) -> None:
        self._validate_type(value = drop_na_col, dtype = (bool, ), msg = f"drop_na_col must be a boolean. Passed: '{drop_na_col}'")
        self._drop_na_col = drop_na_col

    @property 
    def rel_thres(self) -> float:
        return self._rel_thres

    @rel_thres.setter
    def rel_thres(self, rel_thres: float) -> None:
        self._validate_type(value = rel_thres, dtype = (float, ), msg = f"rel_thresh must be a float. Passed: '{rel_thres}'")
            
        if rel_thres > 1 or rel_thres <= 0:
            raise ValueError(f"rel_thres must be a real number number: 0 < R < 1. Passed: '{rel_thres}'")
        self._rel_thres = rel_thres

    @property
    def rel_method(self) -> RELEVANCE_METHOD:
        return self._rel_method

    @rel_method.setter
    def rel_method(self, rel_method: RELEVANCE_METHOD) -> None:
        self._validate_type(value = rel_method, dtype = (RELEVANCE_METHOD, ), msg = f"rel_method must be an enum of type RELEVANCE_METHOD. Passed: '{rel_method}'")
        self._rel_method = rel_method

    @property
    def rel_xtrm_type(self) -> RELEVANCE_XTRM_TYPE:
        return self._rel_xtrm_type

    @rel_xtrm_type.setter
    def rel_xtrm_type(self, rel_xtrm_type: RELEVANCE_XTRM_TYPE) -> None:
        self._validate_type(value = rel_xtrm_type, dtype = (RELEVANCE_XTRM_TYPE, ), msg = f"rel_xtrm_type must be an enum of type RELEVANCE_XTRM_TYPE. Passed: '{rel_xtrm_type}'")
        self._rel_xtrm_type = rel_xtrm_type

    @property
    def rel_coef(self) -> float:
        return self._rel_coef

    @rel_coef.setter
    def rel_coef(self, rel_coef: float | int) -> None:
        self._validate_type(value = rel_coef, dtype = (float, int), msg = f"rel_coef must be a float or int. Passed: '{rel_coef}'")
        self._rel_coef = rel_coef

    @property 
    def rel_ctrl_pts_rg(self) -> list[list[float | int]] | None:
        return self._rel_ctrl_pts_rg

    @rel_ctrl_pts_rg.setter
    def rel_ctrl_pts_rg(self, rel_ctrl_pts_rg: list[list[float | int]] | None) -> None:
        if rel_ctrl_pts_rg is not None:
            self._validate_type(value = rel_ctrl_pts_rg, dtype = (list, ), msg = "rel_ctrl_pts_rg must be 'None' or a 2D array of floats.")
            if len(rel_ctrl_pts_rg) == 0:
                raise TypeError(f"rel_ctrl_pts_rg must be 'None' or a 2D array of floats. You passed an empty 1D array.")
            for pts in rel_ctrl_pts_rg:
                self._validate_type(value = pts, dtype = (list, ), msg = f"rel_ctrl_pts_rg must be 'None' or a 2D array of floats. Passed: '{pts}'")
                any(self._validate_type(value = pt, dtype = (float, int), 
                msg = f"rel_ctrl_pts_rg must be 'None' or a 2D array of floats. Contains element: '{pt}'") for pt in pts)
        self._rel_ctrl_pts_rg = rel_ctrl_pts_rg