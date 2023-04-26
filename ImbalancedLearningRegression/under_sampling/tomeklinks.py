## Third Party Dependencies
from tqdm   import tqdm
from numpy  import ndarray, array, argsort, setxor1d
from pandas import DataFrame, Categorical, to_numeric, factorize

## Internal Dependencies
from ImbalancedLearningRegression.under_sampling.base import BaseUnderSampler
from ImbalancedLearningRegression.utils import (
    SAMPLE_METHOD, 
    RELEVANCE_METHOD, 
    RELEVANCE_XTRM_TYPE,
    TOMEKLINKS_OPTIONS,
    phi,
    phi_ctrl_pts,
    euclidean_dist,
    heom_dist,
    overlap_dist
)

class TomekLinks(BaseUnderSampler):
    """Class to perform the Random Undersampling Algorithm.
    
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
    
    options: TOMEKLINKS_OPTIONS, default = TOMEKLINKS_OPTIONS.MAJORITY
        Which tomeklinks will be removed, in majority intervals, in minority intervals or in both.

        Possible choices are:

            ``TOMEKLINKS_OPTIONS.MAJORITY``: Removes tomeklinks from the majority samples.
            
            ``TOMEKLINKS_OPTIONS.MINORITY``: Removes tomeklinks from the minority samples.
            
            ``TOMEKLINKS_OPTIONS.BOTH``: Removes tomeklinks from both minority and majority samples.

    """
    def __init__(
        self, 
        drop_na_row: bool = True, 
        drop_na_col: bool = True, 
        rel_thres: float = 0.5, 
        rel_method: RELEVANCE_METHOD = RELEVANCE_METHOD.AUTO, 
        rel_xtrm_type: RELEVANCE_XTRM_TYPE = RELEVANCE_XTRM_TYPE.BOTH, 
        rel_coef: float = 1.5, 
        rel_ctrl_pts_rg: list[list[float | int]] | None = None, 
        options: TOMEKLINKS_OPTIONS = TOMEKLINKS_OPTIONS.MAJORITY
    ) -> None:
        
        super().__init__(drop_na_row = drop_na_row, drop_na_col = drop_na_col, samp_method = SAMPLE_METHOD.BALANCE,
        rel_thres = rel_thres, rel_method = rel_method, rel_xtrm_type = rel_xtrm_type, rel_coef = rel_coef, rel_ctrl_pts_rg = rel_ctrl_pts_rg)

        self.options = options

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
        relevances = phi(response_variable = response_variable_sorted, relevance_parameters = relevance_params)

        ## Determine Labels
        label = [0] * len(response_variable_sorted)
        for i in range(len(response_variable_sorted)):
            if (relevances[i] >= self.rel_thres):
                label[response_variable_sorted.index[i]] = 1
            else:
                label[response_variable_sorted.index[i]] = -1

        ## Undersample Data
        new_data = self._undersample(data = new_data, label = label)

        ## Reformat New Data and Return
        new_data = self._format_new_data(new_data = new_data, original_data = data, response_variable = response_variable)
        return new_data

    def _undersample(self, data: DataFrame, label: list[int]) -> DataFrame:

        preprocessed_data, pre_numerical_processed_data = self._preprocess_data(data = data)
        new_data = self._tomeklinks_sample(data = preprocessed_data, 
                                           pre_numerical_processed_data = pre_numerical_processed_data, 
                                           label = label)
        new_data = self._format_synthetic_data(data = data, synth_data = new_data, 
                                               pre_numerical_processed_data = pre_numerical_processed_data)

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

    def _tomeklinks_sample(self, data: DataFrame, pre_numerical_processed_data: DataFrame, label: list[int]) -> DataFrame:
        """Undersamples the data set by removing the TomekLinks depending on user specifications.
        
        Paramaters
        ----------
        data: DataFrame
            Pre-Processed data set ready to be undersampled.

        pre_numerical_processed_data: DataFrame
            Pre-Processed data set, but keeping the nomological columns, used to determine
            nomological and numerical columns.

        label: list[int]
            List of int that keeps track of whether a sample is minority (1) or majority (-1).

        Returns
        -------
        new_data: DataFrame
            The undersampled data set.
        """

        ## Identify Indexes of Columns storing Nomological vs Numerical values
        columns_nom_idx = [idx for idx, column in enumerate(pre_numerical_processed_data.columns) 
                           if pre_numerical_processed_data[column].dtype in ["object", "bool", "datetime64"]]
        columns_num_idx = list(set(list(range(len(pre_numerical_processed_data.columns)))) - set(columns_nom_idx))

        ## Identify the range of values for each Numerical Columns
        columns_num_ranges = [max(data.iloc[:, idx]) - min(data.iloc[:, idx]) for idx in columns_num_idx]

        ## subset data by either numeric / continuous or nominal / categorical
        data_num = data.iloc[:, columns_num_idx]
        data_nom = data.iloc[:, columns_nom_idx]

        ## calculate distance between observations based on data types
        ## store results over null distance matrix of total num of samples x total num of samples
        dist_matrix = ndarray(shape = (len(data), len(data)))        

        for i in tqdm(range(len(data)), ascii = True, desc = "dist_matrix"):
            for j in range(len(data)):

                ## utilize euclidean distance given that 
                ## data is all numeric / continuous
                if len(columns_nom_idx) == 0:
                    dist_matrix[i][j] = euclidean_dist(
                        a = data_num.iloc[i],
                        b = data_num.iloc[j],
                        d = len(columns_num_idx)
                    )
                
                ## utilize heom distance given that 
                ## data contains both numeric / continuous 
                ## and nominal / categorical
                if len(columns_nom_idx) > 0 and len(columns_num_idx) > 0:
                    dist_matrix[i][j] = heom_dist(
                        
                        ## numeric inputs
                        a_num = data_num.iloc[i],
                        b_num = data_num.iloc[j],
                        d_num = len(columns_num_idx),
                        ranges_num = columns_num_ranges,
                        
                        ## nominal inputs
                        a_nom = data_nom.iloc[i],
                        b_nom = data_nom.iloc[j],
                        d_nom = len(columns_nom_idx)
                    )
                
                ## utilize hamming distance given that 
                ## data is all nominal / categorical
                if len(columns_num_idx) == 0:
                    dist_matrix[i][j] = overlap_dist(
                        a = data_nom.iloc[i],
                        b = data_nom.iloc[j],
                        d = len(columns_nom_idx)
                    )

        ## determine indices of k nearest neighbors (k always equal to 1)
        ## and convert knn index list to matrix
        knn_index = [None] * len(data)
        
        for i in range(len(data)):
            knn_index[i] = argsort(dist_matrix[i])[0:2]
        
        knn_matrix = array(knn_index)

        ## store indices where two observations are each other's nearest neighbor
        temp = []
        for i in range(len(data)):
            for j in range(len(data)):
                if knn_matrix[i][0] == knn_matrix[j][1] and knn_matrix[i][1] == knn_matrix[j][0]:
                    temp.append(knn_matrix[i])

        ## store indices that belong to tomeklinks AND majority class to tomeklink_majority
        ## store indices that belong to tomeklinks AND minority class to tomeklink_minority
        tomeklink_majority = []
        tomeklink_minority = []
        for i in range(len(temp)):
            if label[temp[i][0]] != label[temp[i][1]]:
                if label[temp[i][0]] == -1:
                    tomeklink_majority.append(temp[i][0])
                else:
                    tomeklink_minority.append(temp[i][0])

        ## find all index of the dataset
        all_index = []
        for i in range(len(data)):
            all_index.append(i)

        ## find the index to be undersampled according to user's choice
        remove_index = []
        if self.options == TOMEKLINKS_OPTIONS.MAJORITY:
            remove_index = tomeklink_majority
        elif self.options == TOMEKLINKS_OPTIONS.MINORITY:
            remove_index = tomeklink_minority
        else:
            remove_index = tomeklink_majority + tomeklink_minority

        ## find the non-intersecting values of all_index and remove_index  
        new_index = setxor1d(all_index, remove_index)

        ## create null matrix to store new synthetic observations
        synth_matrix = ndarray(shape = (len(new_index), len(data.columns)))
        
        ## store data in the synthetic matrix
        count = 0 
        for i in tqdm(new_index, ascii = True, desc = "new_index"):
            for attr in range(len(data.columns)):
                synth_matrix[count, attr] = (data.iloc[i, attr])
            count = count + 1

        new_data = DataFrame(synth_matrix)

        return new_data

    # Define Setters and Getters for TomekLinks

    @property
    def options(self) -> TOMEKLINKS_OPTIONS:
        return self._options

    @options.setter
    def options(self, options: TOMEKLINKS_OPTIONS) -> None:
        self._validate_type(value = options, dtype = (TOMEKLINKS_OPTIONS, ), msg = f"options must be an enum of type TOMEKLINKS_OPTIONS. Passed: '{options}'")
        self._options = options