## Third Party Dependencies
from numpy            import repeat
from pandas           import DataFrame, Series, Index, Categorical, factorize, to_numeric
from pandas.api.types import is_numeric_dtype

## Standard Library Dependencies
from typing import Any
from abc    import ABC, abstractmethod

## Internal Dependencies
from ImbalancedLearningRegression.utils import SAMPLE_METHOD, RELEVANCE_METHOD, RELEVANCE_XTRM_TYPE

class BaseSampler(ABC):
    """Base class for all sampler algorithms.

    Warning: This class should not be used directly. Use the derived classes instead.
    
    Parameters
    ----------
    drop_na_row: bool, default = True
        Whether rows with Null values will be dropped in data set.

    drop_na_col: bool, default = True
        Whether columns with Null values will be dropped in data set.

    samp_method: SAMPLE_METHOD, default = SAMPLE_METHOD.BALANCE
        Sampling information to resample the data set.

        Possible choices are:

            ``SAMPLE_METHOD.BALANCE``: A balanced amount of resampling. The resampling percentage
                is determined by the 'average ratio of points to rare/majority intervals' to the
                particular interval's number of points.

            ``SAMPLE_METHOD.EXTREME``: A more extreme amount of resampling. The resampling percentage
                is determined by a more extreme (in terms of value) and complex ratio than BALANCE.

    rel_thresh: float, default = 0.5 must be in interval (0, 1]
        This is the threshold used to determine whether an interval is a minority or majority interval.

    rel_method: RELEVANCE_METHOD, default = RELEVANCE_METHOD.AUTO
        Whether minority and majority intervals will be determined using internally computed parameters
        or by using parameters further defined by the user.

        Possible choices are:

            ``RELEVANCE_METHOD.AUTO``: Intervals are determined without further user input.

            ``RELEVANCE_METHOD.MANUAL``: Intervals are determined by using pre-computed points provided
                by the user.

    rel_xtrm_type: RELEVANCE_XTRM_TYPE, default = RELEVANCE_XTRM_TYPE.BOTH
        Whether minority and majority intervals will include the head/tail ends samples of the distribution.

        Possible choices are:

            ``RELEVANCE_XTRM_TYPE.BOTH``: Will include all points in their respective intervals.

            ``RELEVANCE_XTRM_TYPE.HIGH``: Will include only centre and tail end in their respective intervals.

            ``RELEVANCE_XTRM_TYPE.LOW``: Will include only centre and head end in their respective intervals.

    rel_coef: int or float, default = 1.5, must be positive greater than 0
        The coefficient used in box_plot_stats to determine the different quartile points as part of the 
        different intervals calculations.

    rel_ctrl_pts_rg: (2D array of floats or int) or None, default = None
        The pre-computed control points used in the manual calculation of the intervals.
        Used only if rel_method is set to RELEVANCE_METHOD.MANUAL.
    
    """
    def __init__(
        self, 
        drop_na_row:     bool = True, 
        drop_na_col:     bool = True, 
        samp_method:     SAMPLE_METHOD = SAMPLE_METHOD.BALANCE, 
        rel_thres:       float = 0.5, 
        rel_method:      RELEVANCE_METHOD = RELEVANCE_METHOD.AUTO, 
        rel_xtrm_type:   RELEVANCE_XTRM_TYPE = RELEVANCE_XTRM_TYPE.BOTH, 
        rel_coef:        float | int = 1.5, 
        rel_ctrl_pts_rg: list[list[float | int]] | None = None
    ) -> None:
        
        self.drop_na_row     = drop_na_row 
        self.drop_na_col     = drop_na_col 
        self.samp_method     = samp_method

        self.rel_thres       = rel_thres 
        self.rel_method      = rel_method
        self.rel_xtrm_type   = rel_xtrm_type
        self.rel_coef        = rel_coef
        self.rel_ctrl_pts_rg = rel_ctrl_pts_rg

    def _validate_type(self, value: Any, dtype: tuple[type, ...], msg: str) -> None:
        """Validates whether value is of one of the type in tuple dtype.
        
        Parameters
        ----------
        value: Object of any type
            Parameter for which the type is verified.

        dtype: tuple of types, 
            Where one the types in the tuple should correpond to value, otherwise raises exception

        msg: str
            Error message used when raising Exception if no types in dtype matches type of value
        """
        if type(value) not in dtype:
            raise TypeError(msg)

    def _validate_relevance_method(self) -> None:
        """Validates rel_ctrl_pts_rg if RELEVANCE_METHOD.MANUAL is selected.
        """
        if self.rel_method == RELEVANCE_METHOD.MANUAL and self.rel_ctrl_pts_rg is None:
            raise ValueError("rel_ctrl_pts_rg cannot be None while using a manual relevance method.")

    def _validate_data(self, data: DataFrame) -> None:
        """Validates if data set is a DataFrame.
        
        Parameters
        ----------
        data: DataFrame
            Data set to be resampled.
        """
        self._validate_type(value = data, dtype = (DataFrame, ), msg = "data must be a Pandas Dataframe.")    

    def _validate_response_variable(self, data: DataFrame, response_variable: str) -> None:
        """Validates whether dependent variable matches the specifications for resampling.
        
        Parameters
        ----------
        data: DatFrame
            Data set to be resampled.

        response_variable: str
            Header of the column which represents the dependent variable.
        """
        self._validate_data(data = data)
        self._validate_type(value = response_variable, dtype = (str, ), msg = "response_variable must be a string.")

        if not response_variable in data.columns.values:
            raise ValueError("response_variable must be a header name (string) found in the dataframe")

        if not is_numeric_dtype(data[response_variable]):
            raise ValueError("response_variable column in the dataframe must be specified and numeric.")       

    def _preprocess_nan(self, data: DataFrame) -> DataFrame:
        """Removes rows and columns with Null values per drop_na_col and drop_na_row variables.
        
        Parameters
        ----------
        data: DataFrame
            Data set to be resampled.

        Returns
        -------
        data: DataFrame
            Data set where rows and columns with Null values may be removed.
        """   
        if self.drop_na_col == True:
            data = data.dropna(axis = 1)  ## drop columns with nan's

        if self.drop_na_row == True:
            data = data.dropna(axis = 0)  ## drop rows with nan's

        if data.isnull().values.any():
            raise ValueError("cannot proceed: data cannot contain NaN values")

        return data

    def _create_new_data(self, data: DataFrame, response_variable: str) -> tuple[DataFrame, "Series[Any]"]:
        """Creates a DataFrame which matches the standards required for resampling.

        Parameters
        ----------
        data: DataFrame
            DataFrame containing the data set to be resampled.

        response_variable: str
            The header of the column which corresponds to the dependent variable in the data set to be resampled.

        Returns
        -------
        new_data: DataFrame
            The DataFrame that will be used throughout execution for resampling.

        response_col_sorted: Series
            A Series that contains the sorted values of the dependent column.

        """
        ## Create new DataFrame
        new_data = data.copy()

        ## determine column position for response variable
        response_col_pos = new_data.columns.get_loc(response_variable)

        ## move response variable to last column
        if response_col_pos < len(new_data.columns) - 1:
            cols = list(new_data.columns)
            cols[response_col_pos], cols[len(new_data.columns) - 1] = cols[len(new_data.columns) - 1], cols[response_col_pos]
            new_data = new_data[cols]

        ## store original feature headers and
        ## encode feature headers to index position
        new_data.columns = [num for num in range(len(new_data.columns))]

        ## sort response variable by ascending order
        response_col = DataFrame(new_data[len(new_data.columns) - 1])
        response_col_sorted = response_col.sort_values(by = (new_data.columns[len(new_data.columns) - 1]).tolist())
        response_col_sorted = response_col_sorted[len(new_data.columns) - 1]
        
        return new_data, response_col_sorted     

    def _validate_relevance(self, relevances: list[float]) -> None:
        """Validates that not all samples are part of one majority/minority interval.
        
        Parameters
        ----------
        relevances: list[float]
            List of relevances, one relevance per sample in the data set.
        """
        if all(i == 0 for i in relevances):
            raise ValueError("redefine phi relevance function: all points are 1")

        if all(i == 1 for i in relevances):
            raise ValueError("redefine phi relevance function: all points are 0")   

    def _identify_intervals(self, response_variable_sorted: "Series[Any]", relevances: list[float]) -> tuple[dict[int, "Series[Any]"], list[float]]:
        """Identify the different minority and majority intervals in the data set.

        Parameters
        ----------
        response_variable_sorted: Series
            A Series that contains the sorted values of the dependent column.

        relevances: List of floats
            A list that contains the calculated relevances for every samples.

        Returns
        -------
        intervals: Dictionary with integers for keys and Series for values. 
            Dictionary where the key is the number corresponding to an interval
            and the value is a Series that contains the samples that makeup the interval.


        percs: List of floats
            List that contains the computed resampling percentages for each intervals.

        """
        ## determine bin (rare or normal) by interval classification
        interval_indices = [0]

        for i in range(len(response_variable_sorted) - 1):
            if ((relevances[i] >= self.rel_thres and relevances[i + 1] < self.rel_thres) or 
            (relevances[i] < self.rel_thres and relevances[i + 1] >= self.rel_thres)):
                interval_indices.append(i + 1)

        interval_indices.append(len(response_variable_sorted))

        ## determine the samples for each interval classification
        intervals: dict[int, "Series[Any]"] = {}

        for i in range(len(interval_indices) - 1):
            intervals.update({i: response_variable_sorted.iloc[interval_indices[i]:interval_indices[i + 1]]})

        ## calculate over / under sampling percentage according to
        ## bump class and user specified method ("balance" or "extreme")
        ratio_samples_to_intervals = round(len(response_variable_sorted) / (len(interval_indices) - 1))
        percs: list[float] = []
        scale = []
        obj   = []
        
        if self.samp_method == SAMPLE_METHOD.BALANCE:
            for interval in intervals.values():
                percs.append(ratio_samples_to_intervals / len(interval))  

        elif self.samp_method == SAMPLE_METHOD.EXTREME:
            for interval in intervals.values():
                scale.append(ratio_samples_to_intervals ** 2 / len(interval))
            scale = (len(interval_indices) - 1) * ratio_samples_to_intervals / sum(scale)
            
            for idx, interval in intervals.items():
                obj.append(round(ratio_samples_to_intervals ** 2 / len(interval) * scale, 2))
                percs.append(round(obj[idx] / len(interval), 1))

        return intervals, percs

    def _preprocess_synthetic_data(self, data: DataFrame, indices: Index) -> tuple[DataFrame, DataFrame]:
        """Pre-processes the DataFrame that will contain the resampled data set.

        Parameters
        ----------
        data: DataFrame
            DataFrame that contains the data set that matches our specifications for resampling.

        indicies: Index
            The Index that contains the indicies for all the points that makeup this interval that will be resampled.
        
        Returns
        -------
        preprocessed_data: DataFrame
            Pre-Processed DataFrame that contains the points that will be used for resampling this interval.

        pre_numerical_processed_data: DataFrame
            Pre-Processed DataFrame that still has unmodified nomological columns, which will be used for 
            formatting the DataFrame that contains resampled values.

        """
        preprocessed_data: DataFrame = data.loc[indices]

        ## find features without variation (constant features)
        feat_const = preprocessed_data.columns[preprocessed_data.nunique() == 1]

        ## temporarily remove constant features
        preprocessed_data = preprocessed_data.drop(feat_const, axis = 1)

        ## reindex features with variation
        for idx, column in enumerate(preprocessed_data.columns):
            preprocessed_data.rename(columns = { column : idx }, inplace = True)

        pre_numerical_processed_data = preprocessed_data.copy()
        
        ## create nominal and numeric feature list and
        ## label encode nominal / categorical features
        ## (strictly label encode, not one hot encode) 
        nom_dtypes = ["object", "bool", "datetime64"]

        for idx, column in enumerate(preprocessed_data.columns):
            if preprocessed_data[column].dtype in nom_dtypes:
                preprocessed_data.isetitem(idx, Categorical(factorize(preprocessed_data.iloc[:, idx])[0]))

        preprocessed_data = preprocessed_data.apply(to_numeric)

        return preprocessed_data, pre_numerical_processed_data

    def _format_synthetic_data(self, data: DataFrame, synth_data: DataFrame, pre_numerical_processed_data: DataFrame) -> DataFrame:
        """Formats the resampled DataFrame for that particular interval.

        Parameters
        ----------
        data: DataFrame
            DataFrame that contains all of the samples of the data set.

        synth_data: DataFrame
            DataFrame that contains the post-resampled samples for that particular interval.

        pre_numerical_processed_data: DataFrame
            Pre-Processed DataFrame that still has unmodified nomological columns, which is
            used to format the synth_data back into the format of the original data set.

        Returns
        -------
        synth_data:
            DataFrame that contains post-resampled samples for that particular interval,
            but matches format of the original data set.

        """
        nom_dtypes = ["object", "bool", "datetime64"]
        num_dtypes = ["int64", "float64"]
        const_cols = data.columns[data.nunique() == 1]

        for column in pre_numerical_processed_data.columns:
            ## convert encoded nomological column's values back to previous values
            if pre_numerical_processed_data[column].dtype in nom_dtypes: 
                encoded_list = synth_data.loc[:, column].unique()                   # list of all encoded values for that column 
                cat_list     = pre_numerical_processed_data.loc[:, column].unique() # list of all uncoded values for that column

                for encoded in encoded_list:
                    # replace all instances of the encoded value by its unencoded value
                    synth_data.loc[:, column] = synth_data.loc[:, column].replace(to_replace = encoded, value = cat_list[int(encoded)])
            
            ## convert negative values to zero in non-negative features
            elif pre_numerical_processed_data[column].dtype in num_dtypes and (pre_numerical_processed_data[column] > 0).any():
                synth_data.loc[:, column] = synth_data.loc[:, column].clip(lower = 0)
        
        ## remove constant columns from the original data set copied by value
        ## and use the remaining column headers and set the resampled DataFrame
        ## columns to the original dataset
        synth_data.columns = data.drop(const_cols, axis = 1).columns 
        ## reintroduce constant features previously removed
        for column in const_cols:
            synth_data.insert(
                loc = data.columns.get_loc(column),
                column = column,
                value = repeat(data.loc[0, column], len(synth_data)))
        
        ## return over-sampling results dataframe
        return synth_data

    def _format_new_data(self, new_data: DataFrame, original_data: DataFrame, response_variable: str) -> DataFrame:
        """Formats the whole resampled data set to match the original data set.

        Parameters
        ----------
        new_data: DataFrame
            The DataFrame that contains the resampled data set, which will be formatted.

        original_data: DataFrame
            The DataFrame that contains the original data set.

        response_variable: str
            The header of the column which corresponds to the dependent variable in the data set to be resampled.

        Returns
        -------
        new_data: DataFrame
            Formatted resampled DataFrame.

        """
        response_col_pos = original_data.columns.get_loc(response_variable)
        
        ## restore response variable y to original position
        if response_col_pos < len(original_data.columns) - 1:
            cols = [num for num in range(len(original_data.columns))]
            cols[response_col_pos], cols[len(original_data.columns) - 1] = cols[len(original_data.columns) - 1], cols[response_col_pos]
            new_data = new_data[cols]

        ## rename feature headers to originals
        new_data.columns = original_data.columns
        
        ## restore original data types
        for idx, column in enumerate(original_data.columns):
            new_data.isetitem(idx, new_data.loc[:, column].astype(original_data[column].dtype))
        
        ## return modified training set
        return new_data

    @abstractmethod
    def fit_resample(self, data: DataFrame, response_variable: str) -> DataFrame:
        """Resample the data set.

        Parameters
        ----------
        data: DataFrame
            DataFrame that contains the data set to be resampled.

        response_variable: str
            String that contains the header for the column that contains the dependent variable.

        Returns
        -------
        new_data: DataFrame
            DataFrame that contains the resampled data set.

        """
        raise NotImplementedError("BaseSampler must never call fit_resample as it's just a base abstract class.")

    ## Define Setters and Getters for BaseSampler

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