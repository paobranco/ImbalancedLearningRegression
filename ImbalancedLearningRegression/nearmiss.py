## load dependencies - third party
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# load dependencies - internal
from ImbalancedLearningRegression.phi import phi
from ImbalancedLearningRegression.phi_ctrl_pts import phi_ctrl_pts

def nearmiss(
  
   ## main arguments / inputs
    data,                     ## training set (pandas dataframe)
    y,                        ## response variable y by name (string)
    version = 1,              ## version (1, 2, or 3), default 1
    samp_method = "balance",  ## over / under sampling ("balance" or extreme")
    drop_na_col = True,       ## auto drop columns with nan's (bool)
    drop_na_row = True,       ## auto drop rows with nan's (bool)         
  
    ## phi relevance function arguments / inputs
    rel_thres = 0.5,          ## relevance threshold considered rare (pos real)
    rel_method = "auto",      ## relevance method ("auto" or "manual")
    rel_xtrm_type = "both",   ## distribution focus ("high", "low", "both")
    rel_coef = 1.5,           ## coefficient for box plot (pos real)
    rel_ctrl_pts_rg = None,   ## input for "manual" rel method  (2d array)

    ## KNeighborsClassifier attribute
    k = 3,                    ## the number of neighbors used for K-NN
    n_jobs = 1,               ## the number of parallel jobs to run for neighbors search

    ## user-defined KNeighborsClassifier
    k_neighbors_classifier = None  ## user-defined estimator allowing more non-default attributes
                                   ## will ignore k and n_jobs values if not None

):
  
  """
  
  NearMiss-1: Majority class examples with minimum average distance to three closest minority class examples.
  NearMiss-2: Majority class examples with minimum average distance to three furthest minority class examples.
  NearMiss-3: Majority class examples with minimum distance to each minority class example.
  
  """
  
  ## pre-process missing values
    if bool(drop_na_col) == True:
        data = data.dropna(axis = 1)  ## drop columns with nan's
    
    if bool(drop_na_row) == True:
        data = data.dropna(axis = 0)  ## drop rows with nan's
    
    ## quality check for missing values in dataframe
    if data.isnull().values.any():
        raise ValueError("cannot proceed: data cannot contain NaN values")
    
    ## quality check for y
    if isinstance(y, str) is False:
        raise ValueError("cannot proceed: y must be a string")
    
    if y in data.columns.values is False:
        raise ValueError("cannot proceed: y must be an header name (string) \
               found in the dataframe")
    
    ## quality check for sampling method
    if samp_method in ["balance", "extreme"] is False:
        raise ValueError("samp_method must be either: 'balance' or 'extreme'")
    
    ## quality check for relevance threshold parameter
    if rel_thres == None:
        raise ValueError("cannot proceed: relevance threshold required")
    
    if rel_thres > 1 or rel_thres <= 0:
        raise ValueError("rel_thres must be a real number number: 0 < R < 1")

    ## quality check for k
    if type(k) != int or k <= 0:
        raise ValueError("k must be a positive integer")

    ## quality check for n_jobs
    if type(n_jobs) != int:
        raise ValueError("n_jobs must be an integer")
    
    ## store data dimensions
    n = len(data)
    d = len(data.columns)
    
    ## store original data types
    feat_dtypes_orig = [None] * d
    
    for j in range(d):
        feat_dtypes_orig[j] = data.iloc[:, j].dtype
    
    ## determine column position for response variable y
    y_col = data.columns.get_loc(y)
    
    ## move response variable y to last column
    if y_col < d - 1:
        cols = list(range(d))
        cols[y_col], cols[d - 1] = cols[d - 1], cols[y_col]
        data = data[data.columns[cols]]
    
    ## store original feature headers and
    ## encode feature headers to index position
    feat_names = list(data.columns)
    data.columns = range(d)
    
    ## sort response variable y by ascending order
    y = pd.DataFrame(data[d - 1])
    y_sort = y.sort_values(by = d - 1)
    y_sort = y_sort[d - 1]

    def _selection_dist_based(
        self, X, y, dist_vec, num_samples, key, sel_strategy="nearest"
    ):
        """Select the appropriate samples depending of the strategy selected.
        
        Taken directly from https://github.com/scikit-learn-contrib/imbalanced-learn/blob/master/imblearn/under_sampling/_prototype_selection/_nearmiss.py
        
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Original samples.
        y : array-like, shape (n_samples,)
            Associated label to X.
        dist_vec : ndarray, shape (n_samples, )
            The distance matrix to the nearest neigbour.
        num_samples: int
            The desired number of samples to select.
        key : str or int,
            The target class.
        sel_strategy : str, optional (default='nearest')
            Strategy to select the samples. Either 'nearest' or 'farthest'
        Returns
        -------
        idx_sel : ndarray, shape (num_samples,)
            The list of the indices of the selected samples.
        """

        # Compute the distance considering the farthest neighbour
        dist_avg_vec = np.sum(dist_vec[:, -self.nn_.n_neighbors :], axis=1)

        target_class_indices = np.flatnonzero(y == key)
        if dist_vec.shape[0] != _safe_indexing(X, target_class_indices).shape[0]:
            raise RuntimeError(
                "The samples to be selected do not correspond"
                " to the distance matrix given. Ensure that"
                " both `X[y == key]` and `dist_vec` are"
                " related."
            )

        # Sort the list of distance and get the index
        if sel_strategy == "nearest":
            sort_way = False
        else:  # sel_strategy == "farthest":
            sort_way = True

        sorted_idx = sorted(
            range(len(dist_avg_vec)),
            key=dist_avg_vec.__getitem__,
            reverse=sort_way,
        )

        # Throw a warning to tell the user that we did not have enough samples
        # to select and that we just select everything
        if len(sorted_idx) < num_samples:
            warnings.warn(
                "The number of the samples to be selected is larger"
                " than the number of samples available. The"
                " balancing ratio cannot be ensure and all samples"
                " will be returned."
            )

        # Select the desired number of samples
        return sorted_idx[:num_samples]
    
    
    
    ## -------------------------------- phi --------------------------------- ##
    ## calculate parameters for phi relevance function
    ## (see 'phi_ctrl_pts()' function for details)
    phi_params = phi_ctrl_pts(
        
        y = y_sort,                ## y (ascending)
        method = rel_method,       ## defaults "auto" 
        xtrm_type = rel_xtrm_type, ## defaults "both"
        coef = rel_coef,           ## defaults 1.5
        ctrl_pts = rel_ctrl_pts_rg ## user spec
    )
    
    ## calculate the phi relevance function
    ## (see 'phi()' function for details)
    y_phi = phi(
        
        y = y_sort,                ## y (ascending)
        ctrl_pts = phi_params      ## from 'phi_ctrl_pts()'
    )
    
    ## phi relevance quality check
    if all(i == 0 for i in y_phi):
        raise ValueError("redefine phi relevance function: all points are 1")
    
    if all(i == 1 for i in y_phi):
        raise ValueError("redefine phi relevance function: all points are 0")
    ## ---------------------------------------------------------------------- ##
    
    
    ## need to find minority and majority classes to use below
    

    if version == 1:
    
      dist_vec, idx_vec = self.nn_.kneighbors(
                        X_class, n_neighbors=self.nn_.n_neighbors
                    )
                    index_target_class = self._selection_dist_based(
                        X,
                        y,
                        dist_vec,
                        n_samples,
                        target_class,
                        sel_strategy="nearest",
                    )
    
    elif version == 2:
      dist_vec, idx_vec = self.nn_.kneighbors(
                        X_class, n_neighbors=target_stats[class_minority]
                    )
                    index_target_class = self._selection_dist_based(
                        X,
                        y,
                        dist_vec,
                        n_samples,
                        target_class,
                        sel_strategy="nearest",
                    )
    
    
    if version == 3:
    
      self.nn_ver3_.fit(X_class)
                    dist_vec, idx_vec = self.nn_ver3_.kneighbors(
                        _safe_indexing(X, minority_class_indices)
                    )
                    idx_vec_farthest = np.unique(idx_vec.reshape(-1))
                    X_class_selected = _safe_indexing(X_class, idx_vec_farthest)
                    y_class_selected = _safe_indexing(y_class, idx_vec_farthest)

                    dist_vec, idx_vec = self.nn_.kneighbors(
                        X_class_selected, n_neighbors=self.nn_.n_neighbors
                    )
                    index_target_class = self._selection_dist_based(
                        X_class_selected,
                        y_class_selected,
                        dist_vec,
                        n_samples,
                        target_class,
                        sel_strategy="farthest",
                    )
                    # idx_tmp is relative to the feature selected in the
                    # previous step and we need to find the indirection
                    index_target_class = idx_vec_farthest[index_target_class]
    
  
  return data_new
    
