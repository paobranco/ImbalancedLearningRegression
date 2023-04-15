## Third Party Dependencies
from tqdm   import tqdm
from numpy  import ndarray, argsort, array, random
from pandas import DataFrame, Series, Index, concat, Categorical, factorize, to_numeric

## Standard Library Dependencies
from typing import Any

## Internal Dependencies
from ImbalancedLearningRegression.utils.phi          import phi
from ImbalancedLearningRegression.utils.phi_ctrl_pts import phi_ctrl_pts
from ImbalancedLearningRegression.over_sampling.base import BaseOverSampler
from ImbalancedLearningRegression.utils.dist_metrics import euclidean_dist, heom_dist, overlap_dist
from ImbalancedLearningRegression.utils.models       import SAMPLE_METHOD, RELEVANCE_METHOD, RELEVANCE_XTRM_TYPE

class ADASYN(BaseOverSampler):

    def __init__(self, drop_na_row: bool = True, drop_na_col: bool = True, samp_method: SAMPLE_METHOD = SAMPLE_METHOD.BALANCE, 
                 rel_thres: float = 0.5, rel_method: RELEVANCE_METHOD = RELEVANCE_METHOD.AUTO, rel_xtrm_type: RELEVANCE_XTRM_TYPE = RELEVANCE_XTRM_TYPE.BOTH, 
                 rel_coef: float | int = 1.5, rel_ctrl_pts_rg: list[list[float | int]] | None = None, neighbours: int = 5) -> None:

        super().__init__(drop_na_row = drop_na_row, drop_na_col = drop_na_col, samp_method = samp_method, rel_thres = rel_thres,
        rel_method = rel_method, rel_xtrm_type = rel_xtrm_type, rel_coef = rel_coef, rel_ctrl_pts_rg = rel_ctrl_pts_rg)

        self.neighbours = neighbours

    def _validate_neighbours(self, data: DataFrame) -> None:
        if self.neighbours > len(data):
            raise ValueError("cannot proceed: neighbours is greater than the number of observations / rows contained in the dataframe")

    def fit_resample(self, data: DataFrame, response_variable: str) -> DataFrame:

        # Validate Parameters
        self._validate_relevance_method()
        self._validate_data(data = data)
        self._validate_response_variable(data = data, response_variable = response_variable)
        self._validate_neighbours(data = data)

        # Remove Columns with Null Values
        data = self._preprocess_nan(data = data)

        # Create new DataFrame that will be returned and identify Minority and Majority Intervals
        new_data, response_variable_sorted = self._create_new_data(data = data, response_variable = response_variable)
        relevance_params = phi_ctrl_pts(response_variable = response_variable_sorted)
        relevances       = phi(response_variable = response_variable_sorted, relevance_parameters = relevance_params)
        intervals, perc  = self._identify_intervals(response_variable_sorted = response_variable_sorted, relevances = relevances)

        # Oversample Data
        new_data = self._oversample(data = new_data, indicies = intervals, perc = perc)

        # Reformat New Data and Return
        new_data = self._format_new_data(new_data = new_data, original_data = data, response_variable = response_variable)
        return new_data

    def _preprocess_synthetic_data(self, data: DataFrame, indicies: Index) -> tuple[DataFrame, DataFrame]:
        preprocessed_data: DataFrame = data.loc[indicies]
        pre_numerical_processed_data: DataFrame = data.copy()

        for data_new in [preprocessed_data, pre_numerical_processed_data]:
            ## find features without variation (constant features)
            feat_const = data_new.columns[preprocessed_data.nunique() == 1]

            ## temporarily remove constant features
            data_new = data_new.drop(feat_const, axis = 1)

            ## reindex features with variation
            for idx, column in enumerate(data_new.columns):
                data_new.rename(columns = { column : idx }, inplace = True)
        
        ## create nominal and numeric feature list and
        ## label encode nominal / categorical features
        ## (strictly label encode, not one hot encode) 
        nom_dtypes = ["object", "bool", "datetime64"]

        for idx, column in enumerate(preprocessed_data.columns):
            if preprocessed_data[column].dtype in nom_dtypes:
                preprocessed_data.isetitem(idx, Categorical(factorize(preprocessed_data.iloc[:, idx])[0]))

        preprocessed_data = preprocessed_data.apply(to_numeric)

        return preprocessed_data, pre_numerical_processed_data

    def _oversample(self, data: DataFrame, indicies: dict[int, "Series[Any]"], perc: list[float]) -> DataFrame:

        # Create New DataFrame to hold modified DataFrame
        new_data = DataFrame()

        for idx, pts in indicies.items():

            ## no sampling
            if perc[idx] <= 1:

                ## simply return no sampling
                ## results to modified training set
                new_data = concat([data.loc[pts.index], new_data], ignore_index = True)

            ## over-sampling
            if perc[idx] > 1:
                
                ## generate synthetic observations in training set
                ## considered 'minority'
                synth_data, pre_numerical_processed_data = self._preprocess_synthetic_data(data = data, indicies = pts.index)
                synth_data = self._adasyn_oversample(data = data, synth_data = synth_data, perc = perc[idx])
                synth_data = self._format_synthetic_data(data = data, synth_data = synth_data, pre_numerical_processed_data = pre_numerical_processed_data)
                
                ## concatenate over-sampling
                ## results to modified training set
                new_data = concat([synth_data, new_data], ignore_index = True)

                ## concatenate original data
                ## to modified training set
                new_data = concat([data.loc[pts.index], new_data], ignore_index = True)

        return new_data
    
    def _adasyn_oversample(self, data: DataFrame, synth_data: DataFrame, perc: float) -> DataFrame:

        # Create a copy of the original data with no preprocessing and remove constant features from it
        data_orig = data.copy()
        columns_const_idx = data_orig.columns[data_orig.nunique() == 1]
        data_orig = data_orig.drop(columns_const_idx, axis = 1)

        ## reindex features with variation
        for idx, column in enumerate(data_orig.columns):
            data_orig.rename(columns = { column : idx }, inplace = True)

        # Identify Indexes of Columns storing Nomological vs Numerical values
        columns_nom_idx = [idx for idx, column in enumerate(data_orig.columns) if data_orig[column].dtype in ["object", "bool", "datetime64"]]
        columns_num_idx = list(set(list(range(len(data_orig.columns)))) - set(columns_nom_idx))

        for idx, column in enumerate(data_orig.columns):
            if data_orig[column].dtype in ["object", "bool", "datetime64"]:
                data_orig.isetitem(idx, Categorical(factorize(data_orig.iloc[:, idx])[0]))

        data_orig = data_orig.apply(to_numeric)

        # Identify the range of values for each Numerical Columns
        columns_num_ranges = [max(data_orig.iloc[:, idx]) - min(data_orig.iloc[:, idx]) for idx in columns_num_idx]

        ## subset data by either numeric / continuous or nominal / categorical
        data_num = data_orig.iloc[:, columns_num_idx]
        data_nom = data_orig.iloc[:, columns_nom_idx]

        ## calculate distance between observations based on data types
        ## store results over null distance matrix of num of points in interval x total num of points
        dist_matrix = ndarray(shape = (len(synth_data), len(data_orig)))        

        for i in tqdm(range(len(synth_data)), ascii = True, desc = "dist_matrix"):
            for j in range(len(data_orig)):

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

        ## determine indicies of k nearest neighbors
        ## and convert knn index list to matrix
        knn_indicies      = []
        majority_ratios   = []
        majority_indicies = synth_data.index.symmetric_difference(data_orig.index)
        
        for i in range(len(synth_data)):
            knn_indicies.append(argsort(dist_matrix[i])[1:self.neighbours + 1])
            num_majority_neighbours = 0
            for idx in knn_indicies[i]:
                if idx in majority_indicies:
                    num_majority_neighbours += 1
            majority_ratios.append(num_majority_neighbours / self.neighbours)

        normalized_majority_ratios = []
        for ratio in majority_ratios:
            normalized_majority_ratios.append(ratio / sum(majority_ratios))
        assert(sum(normalized_majority_ratios) > 0.99)

        knn_matrix = array(knn_indicies)
        
        ## total number of new synthetic observations to generate
        n_synth = int(len(synth_data) * (perc - 1))

        ## calculate the number of new synthetic observations
        ## that will be generated for each observation
        num_generated_examples = []
        for ratio in normalized_majority_ratios:
            num_generated_examples.append(int(round(ratio * n_synth)))

        ## create null matrix to store new synthetic observations
        synth_matrix = ndarray(shape=((sum(num_generated_examples)), len(synth_data.columns)))

        for i in tqdm(range(len(synth_data)), ascii=True, desc="index"):
            if num_generated_examples[i] > 0:
                num = sum(num_generated_examples[:i])
                for j in range(num_generated_examples[i]):
                    if int(majority_ratios[i]) == 1:
                        synth_matrix[num + j, :] = synth_data.iloc[i, :]
                    else:
                        neigh = int(random.choice(
                            a = [idx for idx, sample in enumerate(knn_indicies[i]) if sample not in majority_indicies], # indicies of minority neighbours
                            size = 1))
                        
                        ## conduct synthetic minority over-sampling
                        ## technique for regression (adasyn)
                        diffs = data_orig.iloc[knn_matrix[i, neigh], 0:(len(synth_data.columns) - 1)] - synth_data.iloc[i, 0:(len(synth_data.columns) - 1)]
                        synth_matrix[num + j, 0:(len(synth_data.columns) - 1)] = synth_data.iloc[i, 0:(len(synth_data.columns) - 1)] + random.random() * diffs

                        ## randomly assign nominal / categorical features from
                        ## observed cases and selected neighbors
                        for x in columns_nom_idx:
                            synth_matrix[num + j, x] = [data_orig.iloc[knn_matrix[i, neigh], x],
                                                        synth_data.iloc[i, x]][round(random.random())]

                        ## generate synthetic y response variable by
                        ## inverse distance weighted
                        for idx, column in enumerate(columns_num_idx):
                            a = abs(synth_data.iloc[i, column] -
                                    synth_matrix[num + j, column]) / columns_num_ranges[idx]
                            b = abs(data_orig.iloc[knn_matrix[i, neigh], column] -
                                    synth_matrix[num + j, column]) / columns_num_ranges[idx]

                            if len(columns_nom_idx) > 0:
                                a = a + sum(synth_data.iloc[i, columns_nom_idx] !=
                                            synth_matrix[num + j, columns_nom_idx])
                                b = b + sum(data_orig.iloc[knn_matrix[i, neigh], columns_nom_idx] !=
                                            synth_matrix[num + j, columns_nom_idx])
                                            
                            if a == b:
                                synth_matrix[num + j, (len(synth_data.columns) - 1)] = synth_data.values[i, (len(synth_data.columns) - 1)] + data_orig.values[
                                    knn_matrix[i, neigh], (len(synth_data.columns) - 1)] / 2
                            else:
                                synth_matrix[num + j, (len(synth_data.columns) - 1)] = (b * synth_data.iloc[i, (len(synth_data.columns) - 1)] +
                                                                a * data_orig.iloc[knn_matrix[i, neigh], (len(synth_data.columns) - 1)]) / (a + b)

        ## convert synthetic matrix to dataframe
        synth_data = DataFrame(synth_matrix)
        
        ## synthetic data quality check
        if sum(synth_data.isnull().sum()) > 0:
            raise ValueError("synthetic data contains missing values")

        return synth_data
                
    # Define Setters and Getters for ADASYN

    @property
    def neighbours(self) -> int:
        return self._neighbours

    @neighbours.setter
    def neighbours(self, neighbours: int) -> None:
        self._validate_type(value = neighbours, dtype = (int, ), msg = f"neighbours should be an int. Passed: {neighbours}")
        self._neighbours = neighbours