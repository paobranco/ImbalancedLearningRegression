## Third Party Dependencies
import numpy  as np
import bisect as bs
from   pandas import Series

## Standard Library Dependencies
from typing import Any

## Internal Dependencies
from ImbalancedLearningRegression.utils.models import RelevanceParameters
    
## calculate the phi relevance function
def phi(
    
    ## arguments / inputs
    response_variable: "Series[Any]",         ## response variable
    relevance_parameters: RelevanceParameters ## params from the 'ctrl_pts()' function
    ) -> list[float]:
    
    """
    generates a monotonic piecewise cubic spline from a sorted list (ascending)
    of the response variable y in order to determine which observations exceed 
    a given threshold ('rel_thres' argument in the main 'smogn()' function)
    
    returns an array of length n (number of observations in the training set) of 
    the phi relevance values corresponding to each observation in y to determine
    whether or not an given observation in y is considered 'normal' or 'rare'
    
    the 'normal' observations get placed into a majority class subset or 'bin' 
    (normal bin) and are under-sampled, while the 'rare' observations get placed 
    into seperate minority class subset (rare bin) where they are over-sampled
    
    the original implementation was as an R foreign function call to C and later 
    adapted to Fortran 90, but was implemented here in Python for the purposes
    of consistency and maintainability
    
    ref:
    
    Branco, P., Ribeiro, R., Torgo, L. (2017). 
    Package 'UBL'. The Comprehensive R Archive Network (CRAN).
    https://cran.r-project.org/web/packages/UBL/UBL.pdf.
    
    Fritsch, F., Carlson, R. (1980).
    Monotone Piecewise Cubic Interpolation.
    SIAM Journal on Numerical Analysis, 17(2):238-246.
    https://doi.org/10.1137/0717021.
    
    Ribeiro, R. (2011). Utility-Based Regression.
    (PhD Dissertation, Dept. Computer Science, 
    Faculty of Sciences, University of Porto).
    """
    
    ## assign variables
    num_pts      = len(response_variable)             ## number of points in response_variable
    num_ctrl_pts = relevance_parameters["num_pts"]    ## number of control points
    ctrl_pts     = relevance_parameters["ctrl_pts"]   ## control points
    
    ## reindex response_variable as pts
    pts = response_variable.reset_index(drop = True)
    
    ## initialize phi relevance function
    relevances = phi_init(pts, num_pts, num_ctrl_pts, ctrl_pts)
    
    ## return phi values
    return relevances

## pre-process control points and calculate phi values
def phi_init(pts: "Series[Any]", num_pts: int, num_ctrl_pts: int, ctrl_pts: list[float]) -> list[float]:
    
    ## construct control point arrays
    x: list[float]     = []
    y_rel: list[float] = []
    m: list[float]     = []
    
    for i in range(num_ctrl_pts):
        x.append(ctrl_pts[3 * i])
        y_rel.append(ctrl_pts[3 * i + 1])
        m.append(ctrl_pts[3 * i + 2])
    
    ## calculate auxilary coefficients for 'pchip_slope_mono_fc()'
    h:     list[float] = []
    delta: list[float] = []
    
    for i in range(num_ctrl_pts - 1):
        h.append(x[i + 1] - x[i])
        delta.append((y_rel[i + 1] - y_rel[i]) / h[i])
    
    ## conduct monotone piecewise cubic interpolation
    m_adj = pchip_slope_mono_fc(m, delta, num_ctrl_pts)
    
    ## assign variables for 'pchip_val()'
    a = y_rel
    b = m_adj
    
    ## calculate auxilary coefficients for 'pchip_val()'
    c: list[float] = []
    d: list[float] = []
    
    for i in range(num_ctrl_pts - 1):
        c.append((3 * delta[i] - 2 * m_adj[i] - m_adj[i + 1]) / h[i])
        d.append((m_adj[i] - 2 * delta[i] + m_adj[i + 1]) / (h[i] * h[i]))
    
    ## calculate phi values
    relevances = [pchip_val(pts[i], x, a, b, c, d, num_ctrl_pts) for i in range(num_pts)]
    
    ## return phi values to the higher function 'phi()'
    return relevances

## calculate slopes for shape preserving hermite cubic polynomials
def pchip_slope_mono_fc(m: list[float], delta: list[float], num_ctrl_pts: int) -> list[float]:
    
    for k in range(num_ctrl_pts - 1):
        sk = delta[k]
        k1 = k + 1
        
        if abs(sk) == 0:
            m[k] = m[k1] = 0
        
        else:
            alpha = m[k] / sk
            beta = m[k1] / sk
            
            if abs(m[k]) != 0 and alpha < 0:
                m[k] = -m[k]
                alpha = m[k] / sk
            
            if abs(m[k1]) != 0 and beta < 0:
                m[k1] = -m[k1]
                beta = m[k1] / sk
            
            ## pre-process for monotoncity check
            m_2ab3 = 2 * alpha + beta - 3
            m_a2b3 = alpha + 2 * beta - 3
            
            ## check for monotoncity
            if m_2ab3 > 0 and m_a2b3 > 0 and alpha * (
                m_2ab3 + m_a2b3) < (m_2ab3 * m_2ab3):
                
                ## fix slopes if outside of monotoncity
                taus = 3 * sk / np.sqrt(alpha * alpha + beta * beta)
                m[k] = taus * alpha
                m[k1] = taus * beta
    
    ## return adjusted slopes m
    return m

## calculate phi values based on monotone piecewise cubic interpolation
def pchip_val(pt: Any, x: list[float], a: list[float], b: list[float], c: list[float], d: list[float], num_ctrl_pts: int) -> float:
    
    ## find interval that contains or is nearest to y
    i = bs.bisect(
        
        a = x,  ## array of relevance values
        x = pt   ## single observation in y
        ) - 1   ## minus 1 to match index position
    
    ## calculate phi values
    if i == num_ctrl_pts - 1:
        y_val = a[i] + b[i] * (pt - x[i])
    
    elif i < 0:
        y_val = 1
    
    else:
        s = pt - x[i]
        y_val = a[i] + s * (b[i] + s * (c[i] + s * d[i]))
    
    ## return phi values to the higher function 'phi_init()'
    return y_val
