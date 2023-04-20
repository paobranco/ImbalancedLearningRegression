## Third Party Dependencies
from tqdm   import tqdm
from numpy  import ndarray, array, argsort, setxor1d
from pandas import DataFrame, Categorical, to_numeric, factorize

## Standard Library Dependencies
from typing import Any

## Internal Dependencies
from ImbalancedLearningRegression.utils.dist_metrics  import euclidean_dist, heom_dist, overlap_dist
from ImbalancedLearningRegression.utils.phi           import phi
from ImbalancedLearningRegression.utils.phi_ctrl_pts  import phi_ctrl_pts
from ImbalancedLearningRegression.utils.models        import SAMPLE_METHOD, RELEVANCE_METHOD, RELEVANCE_XTRM_TYPE, TOMEKLINKS_OPTIONS
from ImbalancedLearningRegression.under_sampling.base import BaseUnderSampler

class TomekLinks(BaseUnderSampler):

    def __init__(self, drop_na_row: bool = True, drop_na_col: bool = True, 
                 rel_thres: float = 0.5, rel_method: RELEVANCE_METHOD = RELEVANCE_METHOD.AUTO, 
                 rel_xtrm_type: RELEVANCE_XTRM_TYPE = RELEVANCE_XTRM_TYPE.BOTH, rel_coef: float = 1.5, 
                 rel_ctrl_pts_rg: list[list[float | int]] | None = None, 
                 options: TOMEKLINKS_OPTIONS = TOMEKLINKS_OPTIONS.MAJORITY) -> None:
        
        super().__init__(drop_na_row = drop_na_row, drop_na_col = drop_na_col, samp_method = SAMPLE_METHOD.BALANCE,
        rel_thres = rel_thres, rel_method = rel_method, rel_xtrm_type = rel_xtrm_type, rel_coef = rel_coef, rel_ctrl_pts_rg = rel_ctrl_pts_rg)

        self.options = options

    def fit_resample(self, data: DataFrame, response_variable: str) -> DataFrame:
        
        # Validate Parameters
        self._validate_relevance_method()
        self._validate_data(data = data)
        self._validate_response_variable(data = data, response_variable = response_variable)

        # Remove Columns with Null Values
        data = self._preprocess_nan(data = data)

        # Create new DataFrame that will be returned and identify Minority and Majority Intervals
        new_data, response_variable_sorted = self._create_new_data(data = data, response_variable = response_variable)
        relevance_params = phi_ctrl_pts(response_variable = response_variable_sorted)
        relevances       = phi(response_variable = response_variable_sorted, relevance_parameters = relevance_params)

        # Determine Labels
        label = [0] * len(response_variable_sorted)
        for i in range(len(response_variable_sorted)):
            if (relevances[i] >= self.rel_thres):
                label[response_variable_sorted.index[i]] = 1
            else:
                label[response_variable_sorted.index[i]] = -1

        # Oversample Data
        new_data = self._undersample(data = new_data, label = label)

        # Reformat New Data and Return
        new_data = self._format_new_data(new_data = new_data, original_data = data, response_variable = response_variable)
        return new_data

    def _undersample(self, data: DataFrame, label) -> DataFrame:

        preprocessed_data, pre_numerical_processed_data = self._preprocess_data(data = data)
        new_data = self._tomeklinks_sample(data = preprocessed_data, 
                                                    pre_numerical_processed_data = pre_numerical_processed_data, 
                                                    label = label)
        new_data = self._format_synthetic_data(data = data, synth_data = new_data, 
                                                        pre_numerical_processed_data = pre_numerical_processed_data)

        return new_data

    def _preprocess_data(self, data: DataFrame) -> tuple[DataFrame, DataFrame]:

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

    def _tomeklinks_sample(self, data: DataFrame, pre_numerical_processed_data: DataFrame, label) -> DataFrame:

        # Identify Indexes of Columns storing Nomological vs Numerical values
        columns_nom_idx = [idx for idx, column in enumerate(pre_numerical_processed_data.columns) 
                           if pre_numerical_processed_data[column].dtype in ["object", "bool", "datetime64"]]
        columns_num_idx = list(set(list(range(len(pre_numerical_processed_data.columns)))) - set(columns_nom_idx))

        # Identify the range of values for each Numerical Columns
        columns_num_ranges = [max(data.iloc[:, idx]) - min(data.iloc[:, idx]) for idx in columns_num_idx]

        ## subset data by either numeric / continuous or nominal / categorical
        data_num = data.iloc[:, columns_num_idx]
        data_nom = data.iloc[:, columns_nom_idx]

        ## calculate distance between observations based on data types
        ## store results over null distance matrix of num of points in interval x total num of points
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
        
        # added
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