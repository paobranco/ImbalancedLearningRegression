## Third Party Dependencies
from pandas import DataFrame, Series

## Standard Library Dependencies
from typing import Any
from abc    import abstractmethod

## Internal Dependencies
from ImbalancedLearningRegression.base  import BaseSampler
from ImbalancedLearningRegression.utils import SAMPLE_METHOD, RELEVANCE_METHOD, RELEVANCE_XTRM_TYPE

class BaseCombinedSampler(BaseSampler):
    """Base class for all combinedsampler algorithms.

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

    under_samp: bool, default = True
        Whether the algorithm will undersample majority intervals, alongside oversampler minority intervals.

    pert: int or float, default = 0.02
        The degree of variance the algorithm will use for generating synthetic data for oversampling.
    
    replace: bool, default = False
        Whether the same sample can be reused when generating different synthetic samples.
    
    """
    def __init__(
        self, 
        drop_na_row: bool = True, 
        drop_na_col: bool = True, 
        samp_method: SAMPLE_METHOD = SAMPLE_METHOD.BALANCE, 
        rel_thres: float = 0.5, 
        rel_method: RELEVANCE_METHOD = RELEVANCE_METHOD.AUTO, 
        rel_xtrm_type: RELEVANCE_XTRM_TYPE = RELEVANCE_XTRM_TYPE.BOTH, 
        rel_coef: float | int = 1.5, 
        rel_ctrl_pts_rg: list[list[float | int]] | None = None, 
        under_samp: bool = True, 
        pert: int | float = 0.02, 
        replace: bool = False) -> None:
        
        super().__init__(drop_na_row = drop_na_row, drop_na_col = drop_na_col, samp_method = samp_method, rel_thres = rel_thres,
        rel_method = rel_method, rel_xtrm_type = rel_xtrm_type, rel_coef = rel_coef, rel_ctrl_pts_rg = rel_ctrl_pts_rg)

        self.under_samp = under_samp
        self.pert       = pert
        self.replace    = replace

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
        raise NotImplementedError("BaseCombinedSampler must never call fit_resample as it's just a base abstract class.")

    @abstractmethod
    def _sample(self, data: DataFrame, indices: dict[int, "Series[Any]"], perc: list[float]) -> DataFrame:
        """Resample the data set for a combinedsampler algorithm.

        Parameters
        ----------
        data: DataFrame
            DataFrame that contains the data set to be resampled.

        indicies: dict[int, "Series[Any]"]
            Dictionary that contains the specific interval's index as key and a Series containing the samples as values.

        perc: list[float]
            List of floats that contains the resampling percentages for each interval, the interval's index matches the 
            key in indicies.

        Returns
        -------
        new_data: DataFrame
            DataFrame that contains the resampled data set.

        """
        raise NotImplementedError("BaseCombinedSampler must never call _sample as it's just a base abstract class.")

    ## Define Setters and Getters for BaseCombinedSampler

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