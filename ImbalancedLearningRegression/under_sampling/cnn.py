## Third Party Dependencies
from numpy                 import random
from pandas                import DataFrame, Series, concat, Categorical, to_numeric, factorize
from sklearn.neighbors     import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

## Standard Library Dependencies
from typing import Any

## Internal Dependencies
from ImbalancedLearningRegression.under_sampling.base import BaseUnderSampler
from ImbalancedLearningRegression.utils import (
    SAMPLE_METHOD, 
    RELEVANCE_METHOD, 
    RELEVANCE_XTRM_TYPE,
    phi,
    phi_ctrl_pts
)

class CNN(BaseUnderSampler):
    """Class to perform the Random Undersampling Algorithm.
    
    Parameters
    ----------
    neighbour_classifier: KNeighborsClassifier
        KNeighborsClassifier used to determine if there is loss in model performance
        while undersampling the majority intervals. 

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
    
    n_seed: int, default = 1
        Determines the minimum number of majority set samples that are used to generate a minimum majority set
        over the course of the CNN Undersampling algorithm.
    """

    def __init__(
        self,
        neighbour_classifier: KNeighborsClassifier, 
        drop_na_row: bool = True, 
        drop_na_col: bool = True, 
        samp_method: SAMPLE_METHOD = SAMPLE_METHOD.BALANCE, 
        rel_thres: float = 0.5, 
        rel_method: RELEVANCE_METHOD = RELEVANCE_METHOD.AUTO, 
        rel_xtrm_type: RELEVANCE_XTRM_TYPE = RELEVANCE_XTRM_TYPE.BOTH, 
        rel_coef: float = 1.5, 
        rel_ctrl_pts_rg: list[list[float | int]] | None = None, 
        n_seed: int = 1
    ) -> None:
        
        super().__init__(drop_na_row = drop_na_row, drop_na_col = drop_na_col, samp_method = samp_method,
        rel_thres = rel_thres, rel_method = rel_method, rel_xtrm_type = rel_xtrm_type, rel_coef = rel_coef, rel_ctrl_pts_rg = rel_ctrl_pts_rg)

        self.n_seed               = n_seed
        self.neighbour_classifier = neighbour_classifier

    def fit_resample(self, data: DataFrame, response_variable: str) -> DataFrame:
        
        ## Validate Parameters
        self._validate_relevance_method()
        self._validate_data(data = data)
        self._validate_response_variable(data = data, response_variable = response_variable)

        ## Remove Columns with Null Values
        data = self._preprocess_nan(data = data)

        ## Create new DataFrame that will be returned and identify Minority and Majority Intervals
        new_data, response_variable_sorted = self._create_new_data(data = data, response_variable = response_variable)
        relevance_params = phi_ctrl_pts(
            response_variable = response_variable_sorted, 
            method            = self.rel_method,
            xtrm_type         = self.rel_xtrm_type,
            coef              = self.rel_coef,
            ctrl_pts          = self.rel_ctrl_pts_rg)
        relevances       = phi(response_variable = response_variable_sorted, relevance_parameters = relevance_params)
        intervals, perc  = self._identify_intervals(response_variable_sorted = response_variable_sorted, relevances = relevances)

        ## Undersample Data
        new_data = self._undersample(data = new_data, indices = intervals, perc = perc)

        ## Reformat New Data and Return
        new_data = self._format_new_data(new_data = new_data, original_data = data, response_variable = response_variable)
        return new_data

    def _undersample(self, data: DataFrame, indices: dict[int, "Series[Any]"], perc: list[float]) -> DataFrame:

        ## Determine indices of minority samples
        minority_indices = [index for idx, indexes in indices.items() if perc[idx] >= 1 for index in indexes]

        ## Create New DataFrame to hold modified DataFrame
        new_data = DataFrame()
        preprocessed_data, pre_numerical_processed_data = self._preprocess_data(data = data)

        for idx, pts in indices.items():

            ## no sampling
            if perc[idx] >= 1:
                ## simply return no sampling
                ## results to modified training set
                new_data = concat([data.loc[pts.index], new_data], ignore_index = True)

            ## under-sampling
            if perc[idx] < 1:
                
                ## undersample intervals
                ## considered 'minority'
                undersampled_data = self._cnn_undersample(data = preprocessed_data, majority_indices = pts.index.tolist(), store_indices = minority_indices)
                undersampled_data = self._format_synthetic_data(data = data, synth_data = undersampled_data, 
                                                                pre_numerical_processed_data = pre_numerical_processed_data)
                
                ## concatenate under-sampling
                ## results to modified training set
                new_data = concat([undersampled_data, new_data], ignore_index = True)

        return new_data

    def _preprocess_data(self, data: DataFrame) -> tuple[DataFrame, DataFrame]:
        """Pre-processes the entire data set before undersampling.
        
        Parameters
        ----------
        data: DataFrame
            The data set to be undersampled.

        Returns
        -------
        preprocessed_data: DataFrame
            The completely pre-processed data set ready to be undersampled.

        pre_numerical_processed_data: DataFrame
            Pre-Processed DataFrame that still has unmodified nomological columns, which will be used for
            for formatting the undersampled data set.

        """
        preprocessed_data = data.copy()

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

    def _cnn_undersample(self, data: DataFrame, majority_indices: "list[Any]", store_indices: "list[Any]") -> DataFrame:
        """Undersample a majority interval using the CNN Undersampling Algorith.

        Parameters
        ----------
        data: DataFrame
            Preprocessed entire data set ready to be undersampled.

        majority_indices: list[Any]
            List of indices for this majority interval to be undersampled.

        store_indices: list[Any]
            List of all indices that will be part of the final undersampled data set.
            At function call, initially contains only all the minority indices.

        Returns
        -------
        new_data: DataFrame
            DataFrame that contains the undersampled majority interval.

        """

        ## initialize grabbag
        grabbag_indices = list()

        ## randomly pick one or more sample(s) from majority and add it to STORE
        ## based on n_seed
        try:
            normal_seed_index = list(random.choice(a = majority_indices, size = self.n_seed, replace = False))
        except ValueError:
            print("n_seed =", self.n_seed, ">", len(majority_indices))
            print("WARNING: n_seed is greater than the number of samples avaiable in a majority bin, used n_seed = 1 instead!")
            normal_seed_index = list(random.choice(a = majority_indices, size = 1, replace = False))
        store_indices.extend(normal_seed_index)

        ## initial training
        train_X = [list(data.iloc[i, : (len(data.columns) - 1)].values) for i in store_indices]
        train_y = [(0 if i in majority_indices else 1) for i in store_indices]
        min_max_scaler = MinMaxScaler()
        train_X_minmax = min_max_scaler.fit_transform(train_X)
        self.neighbour_classifier.fit(train_X_minmax, train_y)

        ## loop through the majority set
        randomized_index = majority_indices.copy()
        random.shuffle(randomized_index)
        for i in randomized_index:
            if i in store_indices:
                continue
            predict_X = min_max_scaler.transform(data.iloc[i,:(len(data.columns) - 1)].values.reshape(1,-1))
            predict_y = self.neighbour_classifier.predict(predict_X)
            if predict_y == 0:
                grabbag_indices.append(i)
            else:
                store_indices.append(i)
                train_X = [list(data.iloc[j,:(len(data.columns) - 1)].values) for j in store_indices]
                train_y = [(0 if j in majority_indices else 1) for j in store_indices]
                min_max_scaler = MinMaxScaler()
                train_X_minmax = min_max_scaler.fit_transform(train_X)
                self.neighbour_classifier.fit(train_X_minmax, train_y)

        ## loop through the grabbag until empty or no transfer
        while True:
            if len(grabbag_indices) == 0:
                break
            has_transfer = False
            new_grabbag_indices = list()
            for i in grabbag_indices:
                if i in store_indices:
                    raise ValueError("index exists in both store and grabbag")
                predict_X = min_max_scaler.transform(data.iloc[i,:(len(data.columns) - 1)].values.reshape(1,-1))
                predict_y = self.neighbour_classifier.predict(predict_X)
                if predict_y == 0:
                    new_grabbag_indices.append(i)
                else:
                    has_transfer = True
                    store_indices.append(i)
                    train_X = [list(data.iloc[j,:(len(data.columns) - 1)].values) for j in store_indices]
                    train_y = [(0 if j in majority_indices else 1) for j in store_indices]
                    min_max_scaler = MinMaxScaler()
                    train_X_minmax = min_max_scaler.fit_transform(train_X)
                    self.neighbour_classifier.fit(train_X_minmax, train_y)
            grabbag_indices = new_grabbag_indices
            if not has_transfer:
                break

        ## conduct under sampling and store modified training set
        cond = [i in store_indices and i in majority_indices for i in range(len(data))]
        new_data = data.loc[cond,:]

        return new_data

    ## Define Setters and Getters for CNN

    @property
    def n_seed(self) -> int:
        return self._n_seed

    @n_seed.setter
    def n_seed(self, n_seed: int) -> None:
        self._validate_type(value = n_seed, dtype = (int, ), msg = f"n_seed should be an int. Passed: {n_seed}")
        self._n_seed = n_seed

    @property
    def neighbour_classifier(self) -> KNeighborsClassifier:
        return self._neighbour_classifier

    @neighbour_classifier.setter
    def neighbour_classifier(self, neighbour_classifier: KNeighborsClassifier) -> None:
        self._validate_type(value = neighbour_classifier, dtype = (KNeighborsClassifier, ), msg = f"neighbour_classifier should be an int. Passed Type: {type(neighbour_classifier)}")
        self._neighbour_classifier = neighbour_classifier