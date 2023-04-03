## Third Party Dependencies
import numpy as np
from   pandas import Series

## Standard Library Dependencies
from typing import Any

## Internal Dependencies
from ImbalancedLearningRegression.utils.box_plot_stats import box_plot_stats
from ImbalancedLearningRegression.utils.models import RelevanceParameters, RELEVANCE_METHOD, RELEVANCE_XTRM_TYPE

## calculate parameters for phi relevance function
def phi_ctrl_pts(
    
    ## arguments / inputs
    response_variable: "Series[Any]",                          ## response variable
    method: RELEVANCE_METHOD = RELEVANCE_METHOD.AUTO,          ## relevance method (AUTO or MANUAl)
    xtrm_type: RELEVANCE_XTRM_TYPE = RELEVANCE_XTRM_TYPE.BOTH, ## distribution focus (HIGH, LOW, BOTH)
    coef: int | float = 1.5,                                   ## coefficient for box plot
    ctrl_pts: list[list[float | int]] | None = None            ## input for "manual" rel method
    ) -> RelevanceParameters:
    
    """ 
    generates the parameters required for the 'phi()' function, specifies the 
    regions of interest or 'relevance' in the response variable y, the notion 
    of relevance can be associated with rarity
    
    controls how the relevance parameters are calculated by selecting between 
    two methods, either "auto" or "manual"
    
    the "auto" method calls the function 'phi_extremes()' and calculates the 
    relevance parameters by the values beyond the interquartile range
    
    the "manual" method calls the function 'phi_range()' and determines the 
    relevance parameters by user specification (the use of a domain expert 
    is recommended for utilizing this method)
    
    returns a dictionary containing 3 items "method", "num_pts", "ctrl_pts": 
    1) the "method" item contains a chartacter string simply indicating the 
    method used calculate the relevance parameters (control points) either 
    "auto" or "manual"
    
    2) the "num_pts" item contains a positive integer simply indicating the 
    number of relevance parameters returned, typically 3
    
    3) the "ctrl_pts" item contains an array indicating the regions of 
    interest in the response variable y and their corresponding relevance 
    values mapped to either 0 or 1, expressed as [y, 0, 1]
    
    ref:
    
    Branco, P., Ribeiro, R., Torgo, L. (2017).
    Package 'UBL'. The Comprehensive R Archive Network (CRAN).
    https://cran.r-project.org/web/packages/UBL/UBL.pdf.
    
    Ribeiro, R. (2011). Utility-Based Regression.
    (PhD Dissertation, Dept. Computer Science, 
    Faculty of Sciences, University of Porto).
    """
    
    ## conduct 'extremes' method (default)
    if method == RELEVANCE_METHOD.AUTO:
        relevance_parameters = phi_extremes(response_variable, xtrm_type, coef)
    ## conduct 'range' method
    else:
        relevance_parameters = phi_range(ctrl_pts)
    
    ## return phi relevance parameters dictionary
    return relevance_parameters

## calculates phi parameters for statistically extreme values
def phi_extremes(response_variable: "Series[Any]", xtrm_type: RELEVANCE_XTRM_TYPE, coef: float | int) -> RelevanceParameters:
    
    """ 
    assigns relevance to the most extreme values in the distribution of response 
    variable y according to the box plot stats generated from 'box_plot_stat()'
    """
    
    ## create 'pts' variable
    pts = []
    
    ## calculates statistically extreme values by
    ## box plot stats in the response variable y
    ## (see function 'boxplot_stats()' for details)
    bx_plt_st = box_plot_stats(response_variable, coef)
    
    ## calculate range of the response variable y
    rng = [response_variable.min(), response_variable.max()]
    
    ## adjust low
    if xtrm_type in [RELEVANCE_XTRM_TYPE.BOTH, RELEVANCE_XTRM_TYPE.LOW] and any(bx_plt_st["xtrms"]
    < bx_plt_st["stats"][0]):
        pts.extend([bx_plt_st["stats"][0], 1, 0])
   
    ## min
    else:
        pts.extend([rng[0], 0, 0])
    
    ## median
    if bx_plt_st["stats"][2] != rng[0]:
        pts.extend([bx_plt_st["stats"][2], 0, 0])
    
    ## adjust high
    if xtrm_type in [RELEVANCE_XTRM_TYPE.BOTH, RELEVANCE_XTRM_TYPE.HIGH] and any(bx_plt_st["xtrms"]
    > bx_plt_st["stats"][4]):
        pts.extend([bx_plt_st["stats"][4], 1, 0])
    
    ## max
    else:
        if bx_plt_st["stats"][2] != rng[1]:
            pts.extend([rng[1], 0, 0])
    
    ## store phi relevance parameter dictionary
    relevance_parameters: RelevanceParameters = {
        'method':   RELEVANCE_METHOD.AUTO, 
        'num_pts':  round(len(pts) / 3), 
        'ctrl_pts': pts
    }
    
    ## return relevance parameters
    return relevance_parameters

## calculates phi parameters for user specified range
def phi_range(ctrl_pts: list[list[float | int]] | None) -> RelevanceParameters:
    
    """
    assigns relevance to values in the response variable y according to user 
    specification, when specifying relevant regions use matrix format [x, y, m]
    
    x is an array of relevant values in the response variable y, y is an array 
    of values mapped to 1 or 0, and m is typically an array of zeros
    
    m is the phi derivative adjusted afterward by the phi relevance function to 
    interpolate a smooth and continous monotonically increasing function
    
    example:
    [[15, 1, 0],
    [30, 0, 0],
    [55, 1, 0]]
    """
    
    ## set pts to the numpy 2d array (matrix) representation of 'ctrl_pts'
    pts = np.array(ctrl_pts)
    
    ## quality control checks for user specified phi relevance values
    if np.isnan(pts).any() or np.size(pts, axis = 1) > 3 or np.size(
    pts, axis = 1) < 2 or not isinstance(pts, np.ndarray):
        raise ValueError("ctrl_pts must be given as a matrix in the form: [x, y, m]" 
              "or [x, y]")
    
    elif (pts[1: ,[1, ]] > 1).any() or (pts[1: ,[1, ]] < 0).any():
        raise ValueError("phi relevance function only maps values: [0, 1]")
    
    ## store number of control points
    else:
        dx = pts[1:,[0,]] - pts[0:-1,[0,]]
    
    ## quality control check for dx
    if np.isnan(dx).any() or dx.any() == 0:
        raise ValueError("x must strictly increase (not na)")
    
    ## sort control points from lowest to highest
    else:
        pts = pts[np.argsort(pts[:,0])]
    
    ## calculate for two column user specified control points [x, y]
    if np.size(pts, axis = 1) == 2:
        
        ## monotone hermite spline method by fritsch & carlson (monoH.FC)
        dx = pts[1:,[0,]] - pts[0:-1,[0,]]
        dy = pts[1:,[1,]] - pts[0:-1,[1,]]
        sx = dy / dx
        
        ## calculate constant extrapolation
        m = np.divide(sx[1:] + sx[0:-1], 2)
        m = np.array(sx).ravel().tolist()
        m.insert(0, 0)
        m.insert(len(sx), 0)
        
        ## add calculated column 'm' to user specified control points 
        ## from [x, y] to [x, y, m] and store in 'pts'
        pts = np.insert(pts, 2, m, axis = 1)

    ## store phi relevance parameter dictionary
    relevance_parameters: RelevanceParameters = {
        'method':   RELEVANCE_METHOD.MANUAL, 
        'num_pts':  np.size(pts, axis = 0), 
        'ctrl_pts': np.array(pts).ravel().tolist()
    }
    
    ## return dictionary
    return relevance_parameters
