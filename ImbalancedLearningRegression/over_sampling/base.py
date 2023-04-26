## Third Party Dependencies
from pandas import DataFrame, Series

## Standard Library Dependencies
from typing import Any
from abc    import abstractmethod

## Internal Dependencies
from ImbalancedLearningRegression.base  import BaseSampler
from ImbalancedLearningRegression.utils import SAMPLE_METHOD, RELEVANCE_METHOD, RELEVANCE_XTRM_TYPE

class BaseOverSampler(BaseSampler):
    """Base class for all oversampler algorithms.

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
        
        super().__init__(drop_na_row = drop_na_row, drop_na_col = drop_na_col, samp_method = samp_method, rel_thres = rel_thres,
        rel_method = rel_method, rel_xtrm_type = rel_xtrm_type, rel_coef = rel_coef, rel_ctrl_pts_rg = rel_ctrl_pts_rg)

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
        raise NotImplementedError("BaseOverSampler must never call fit_resample as it's just a base abstract class.")

    @abstractmethod
    def _oversample(self, data: DataFrame, indices: dict[int, "Series[Any]"], perc: list[float]) -> DataFrame:
        """Oversample the data set for an oversampler algorithm.

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
            DataFrame that contains the oversampled data set.

        """
        raise NotImplementedError("BaseOverSampler must never call _oversample as it's just a base abstract class.")