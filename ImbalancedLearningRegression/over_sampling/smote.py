## Third Party Dependencies
from tqdm        import tqdm
from numpy       import ndarray, argsort, array, random
from pandas      import DataFrame, Series, concat, Categorical, factorize, to_numeric
from sklearn.svm import SVR

## Standard Library Dependencies
from abc    import abstractmethod
from typing import Any

## Internal Dependencies
from ImbalancedLearningRegression.utils.phi          import phi
from ImbalancedLearningRegression.utils.phi_ctrl_pts import phi_ctrl_pts
from ImbalancedLearningRegression.over_sampling.base import BaseOverSampler
from ImbalancedLearningRegression.utils.dist_metrics import euclidean_dist, heom_dist, overlap_dist
from ImbalancedLearningRegression.utils.models       import SAMPLE_METHOD, RELEVANCE_METHOD, RELEVANCE_XTRM_TYPE

class BaseSMOTE(BaseOverSampler):

    def __init__(self, drop_na_row: bool = True, drop_na_col: bool = True, samp_method: SAMPLE_METHOD = SAMPLE_METHOD.BALANCE, 
                 rel_thres: float = 0.5, rel_method: RELEVANCE_METHOD = RELEVANCE_METHOD.AUTO, rel_xtrm_type: RELEVANCE_XTRM_TYPE = RELEVANCE_XTRM_TYPE.BOTH, 
                 rel_coef: float | int = 1.5, rel_ctrl_pts_rg: list[list[float | int]] | None = None, neighbours: int = 5) -> None:

        super().__init__(drop_na_row = drop_na_row, drop_na_col = drop_na_col, samp_method = samp_method, rel_thres = rel_thres,
        rel_method = rel_method, rel_xtrm_type = rel_xtrm_type, rel_coef = rel_coef, rel_ctrl_pts_rg = rel_ctrl_pts_rg)

        self.neighbours = neighbours

    def _validate_neighbours(self, data: DataFrame) -> None:
        if self.neighbours > len(data):
            raise ValueError("cannot proceed: neighbours is greater than the number of observations / rows contained in the dataframe")

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
                synth_data = self._smote_oversample(pre_numerical_processed_data = pre_numerical_processed_data, 
                                                    synth_data = synth_data, perc = perc[idx])
                synth_data = self._format_synthetic_data(data = data, synth_data = synth_data, pre_numerical_processed_data = pre_numerical_processed_data)
                
                ## concatenate over-sampling
                ## results to modified training set
                new_data = concat([synth_data, new_data], ignore_index = True)

                ## concatenate original data
                ## to modified training set
                new_data = concat([data.loc[pts.index], new_data], ignore_index = True)

        return new_data
    
    def _smote_oversample(self, pre_numerical_processed_data: DataFrame, synth_data: DataFrame, perc: float) -> DataFrame:

        # Identify Indexes of Columns storing Nomological vs Numerical values
        columns_nom_idx = [idx for idx, column in enumerate(pre_numerical_processed_data.columns) 
                           if pre_numerical_processed_data[column].dtype in ["object", "bool", "datetime64"]]
        columns_num_idx = list(set(list(range(len(pre_numerical_processed_data.columns)))) - set(columns_nom_idx))

        # Identify the range of values for each Numerical Columns
        columns_num_ranges = [max(synth_data.iloc[:, idx]) - min(synth_data.iloc[:, idx]) for idx in columns_num_idx]

        ## subset data by either numeric / continuous or nominal / categorical
        data_num = synth_data.iloc[:, columns_num_idx]
        data_nom = synth_data.iloc[:, columns_nom_idx]

        ## calculate distance between observations based on data types
        ## store results over null distance matrix of n x n
        dist_matrix = ndarray(shape = (len(synth_data), len(synth_data)))

        for i in tqdm(range(len(synth_data)), ascii = True, desc = "dist_matrix"):
            for j in range(len(synth_data)):
                
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
        knn_index = [None] * len(synth_data)
        
        for i in range(len(synth_data)):
            knn_index[i] = argsort(dist_matrix[i])[1:self.neighbours + 1]
        
        knn_matrix = array(knn_index)
        
        ## number of new synthetic observations for each rare observation
        x_synth = int(perc - 1)
        
        ## total number of new synthetic observations to generate
        n_synth = int(len(synth_data) * (perc - 1 - x_synth))
        
        ## randomly index data by the number of new synthetic observations
        r_index = random.choice(
            a = tuple(range(0, len(synth_data))), 
            size = n_synth, 
            replace = False, 
            p = None
        )

        ## create null matrix to store new synthetic observations
        synth_matrix = ndarray(shape = ((x_synth * len(synth_data) + n_synth), len(synth_data.columns)))
        
        if x_synth > 0:
            for i in tqdm(range(len(synth_data)), ascii = True, desc = "synth_matrix"):
                
                for j in range(x_synth):
                    
                    ## randomly select a k nearest neighbor
                    neigh = int(random.choice(
                        a = tuple(range(self.neighbours)), 
                        size = 1))
                    
                    ## conduct synthetic minority over-sampling
                    ## technique for regression (smote)
                    diffs = synth_data.iloc[
                        knn_matrix[i, neigh], 0:(len(synth_data.columns) - 1)] - synth_data.iloc[
                        i, 0:(len(synth_data.columns) - 1)]
                    synth_matrix[i * x_synth + j, 0:(len(synth_data.columns) - 1)] = synth_data.iloc[
                        i, 0:(len(synth_data.columns) - 1)] + random.random() * diffs
                    
                    ## randomly assign nominal / categorical features from
                    ## observed cases and selected neighbors
                    for x in columns_nom_idx:
                        synth_matrix[i * x_synth + j, x] = [synth_data.iloc[
                            knn_matrix[i, neigh], x], synth_data.iloc[
                            i, x]][round(random.random())]
                    
                    ## generate synthetic y response variable by
                    ## inverse distance weighted
                    for idx, column in enumerate(columns_num_idx):
                        a = abs(synth_data.iloc[i, column] - synth_matrix[
                            i * x_synth + j, column]) / columns_num_ranges[idx]
                        b = abs(synth_data.iloc[knn_matrix[
                            i, neigh], column] - synth_matrix[
                            i * x_synth + j, column]) / columns_num_ranges[idx]
                    
                        if len(columns_nom_idx) > 0:
                            a = a + sum(synth_data.iloc[
                                i, columns_nom_idx] != synth_matrix[
                                i * x_synth + j, columns_nom_idx])
                            b = b + sum(synth_data.iloc[knn_matrix[
                                i, neigh], columns_nom_idx] != synth_matrix[
                                i * x_synth + j, columns_nom_idx])
                    
                        if a == b:
                            # Revert to using iloc once Pandas fixes the type-hint error with iloc
                            synth_matrix[i * x_synth + j, 
                                (len(synth_data.columns) - 1)] = synth_data.values[i, (len(synth_data.columns) - 1)] + synth_data.values[
                                knn_matrix[i, neigh], (len(synth_data.columns) - 1)] / 2
                        else:
                            synth_matrix[i * x_synth + j, 
                                (len(synth_data.columns) - 1)] = (b * synth_data.iloc[
                                i, (len(synth_data.columns) - 1)] + a * synth_data.iloc[
                                knn_matrix[i, neigh], (len(synth_data.columns) - 1)]) / (a + b)
                        
        if n_synth > 0:
            count = 0
            
            for i in tqdm(r_index, ascii = True, desc = "r_index"):
                
                ## randomly select a k nearest neighbor
                neigh = int(random.choice(
                    a = tuple(range(0, self.neighbours)), 
                    size = 1))
                
                ## conduct synthetic minority over-sampling 
                ## technique for regression (smote)
                diffs = synth_data.iloc[
                    knn_matrix[i, neigh], 0:(len(synth_data.columns) - 1)] - synth_data.iloc[i, 0:(len(synth_data.columns) - 1)]
                synth_matrix[x_synth * len(synth_data) + count, 0:(len(synth_data.columns) - 1)] = synth_data.iloc[
                    i, 0:(len(synth_data.columns) - 1)] + random.random() * diffs
                
                ## randomly assign nominal / categorical features from
                ## observed cases and selected neighbors
                for x in columns_nom_idx:
                    synth_matrix[x_synth * len(synth_data) + count, x] = [synth_data.iloc[
                        knn_matrix[i, neigh], x], synth_data.iloc[
                        i, x]][round(random.random())]
                
                ## generate synthetic y response variable by
                ## inverse distance weighted
                for idx, column in enumerate(columns_num_idx):
                    a = abs(synth_data.iloc[i, column] - synth_matrix[
                        x_synth * len(synth_data) + count, column]) / columns_num_ranges[idx]
                    b = abs(synth_data.iloc[knn_matrix[i, neigh], column] - synth_matrix[
                        x_synth * len(synth_data) + count, column]) / columns_num_ranges[idx]
                
                    if len(columns_nom_idx) > 0:
                        a = a + sum(synth_data.iloc[i, columns_nom_idx] != synth_matrix[
                            x_synth * len(synth_data) + count, columns_nom_idx])
                        b = b + sum(synth_data.iloc[
                            knn_matrix[i, neigh], columns_nom_idx] != synth_matrix[
                            x_synth * len(synth_data) + count, columns_nom_idx])
                    
                    if a == b:
                        synth_matrix[x_synth * len(synth_data) + count, (len(synth_data.columns) - 1)] = synth_data.values[
                            i, (len(synth_data.columns) - 1)] + synth_data.values[
                            knn_matrix[i, neigh], (len(synth_data.columns) - 1)] / 2
                    else:
                        synth_matrix[x_synth * len(synth_data) + count, (len(synth_data.columns) - 1)] = (b * synth_data.iloc[
                            i, (len(synth_data.columns) - 1)] + a * synth_data.iloc[
                            knn_matrix[i, neigh], (len(synth_data.columns) - 1)]) / (a + b)
            
                ## close loop counter
                count = count + 1
        
        ## convert synthetic matrix to dataframe
        synth_data = DataFrame(synth_matrix)
        
        ## synthetic data quality check
        if sum(synth_data.isnull().sum()) > 0:
            raise ValueError("synthetic data contains missing values")

        return synth_data

    @abstractmethod
    def fit_resample(self, data: DataFrame, response_variable: str) -> DataFrame:
        raise NotImplementedError("BaseSMOTE must never call fit_resample as it's just a base abstract class.")
                
    # Define Setters and Getters for BaseSMOTE

    @property
    def neighbours(self) -> int:
        return self._neighbours

    @neighbours.setter
    def neighbours(self, neighbours: int) -> None:
        self._validate_type(value = neighbours, dtype = (int, ), msg = f"neighbours should be an int. Passed: {neighbours}")
        self._neighbours = neighbours

class SMOTE(BaseSMOTE):

    def __init__(self, drop_na_row: bool = True, drop_na_col: bool = True, samp_method: SAMPLE_METHOD = SAMPLE_METHOD.BALANCE, 
                 rel_thres: float = 0.5, rel_method: RELEVANCE_METHOD = RELEVANCE_METHOD.AUTO, rel_xtrm_type: RELEVANCE_XTRM_TYPE = RELEVANCE_XTRM_TYPE.BOTH, 
                 rel_coef: float | int = 1.5, rel_ctrl_pts_rg: list[list[float | int]] | None = None, neighbours: int = 5) -> None:

        super().__init__(drop_na_row = drop_na_row, drop_na_col = drop_na_col, samp_method = samp_method, rel_thres = rel_thres,
        rel_method = rel_method, rel_xtrm_type = rel_xtrm_type, rel_coef = rel_coef, rel_ctrl_pts_rg = rel_ctrl_pts_rg, 
        neighbours = neighbours)

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

class SVMSMOTE(BaseSMOTE):

    def __init__(self, drop_na_row: bool = True, drop_na_col: bool = True, samp_method: SAMPLE_METHOD = SAMPLE_METHOD.BALANCE, 
                 rel_thres: float = 0.5, rel_method: RELEVANCE_METHOD = RELEVANCE_METHOD.AUTO, rel_xtrm_type: RELEVANCE_XTRM_TYPE = RELEVANCE_XTRM_TYPE.BOTH, 
                 rel_coef: float | int = 1.5, rel_ctrl_pts_rg: list[list[float | int]] | None = None, neighbours: int = 5, 
                 svm_estimator: SVR | None = None) -> None:

        super().__init__(drop_na_row = drop_na_row, drop_na_col = drop_na_col, samp_method = samp_method, rel_thres = rel_thres,
        rel_method = rel_method, rel_xtrm_type = rel_xtrm_type, rel_coef = rel_coef, rel_ctrl_pts_rg = rel_ctrl_pts_rg, 
        neighbours = neighbours)

        self.svm_estimator = svm_estimator

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

        # Train SVM Estimator
        self._train_svm_estimator(data = data)

        # Filter Indicies
        intervals = self._filter_indicies(indicies = intervals)

        # Oversample Data
        new_data = self._oversample(data = new_data, indicies = intervals, perc = perc)

        # Reformat New Data and Return
        new_data = self._format_new_data(new_data = new_data, original_data = data, response_variable = response_variable)
        return new_data

    def _train_svm_estimator(self, data: DataFrame) -> None:
        normalized_data = data.copy()

        ## find features without variation (constant features)
        feat_const = normalized_data.columns[normalized_data.nunique() == 1]

        ## temporarily remove constant features
        normalized_data = normalized_data.drop(feat_const, axis = 1)

        ## reindex features with variation
        for idx, column in enumerate(normalized_data.columns):
            normalized_data.rename(columns = { column : idx }, inplace = True)
        
        ## create nominal and numeric feature list and
        ## label encode nominal / categorical features
        ## (strictly label encode, not one hot encode) 
        nom_dtypes = ["object", "bool", "datetime64"]

        for idx, column in enumerate(normalized_data.columns):
            if normalized_data[column].dtype in nom_dtypes:
                normalized_data.isetitem(idx, Categorical(factorize(normalized_data.iloc[:, idx])[0]))

        normalized_data = normalized_data.apply(to_numeric)

        # Normalize data
        normalized_data = (normalized_data-normalized_data.mean(numeric_only = True))/normalized_data.std()

        # Create training data
        train_X = normalized_data[normalized_data.columns[:(len(normalized_data.columns) - 1)]]
        train_Y = normalized_data[len(normalized_data.columns) - 1]

        self.svm_estimator.fit(train_X, train_Y)

    def _filter_indicies(self, indicies: dict[int, "Series[Any]"]) -> dict[int, "Series[Any]"]:
        indicies_new: dict[int, "Series[Any]"] = {}

        for idx, pts in indicies.items():
            indicies_new[idx] = pts.loc[pts.index.intersection(self.svm_estimator.support_)]

        return indicies_new

    # Define Setters and Getters for SVMSMOTE

    @property
    def svm_estimator(self) -> SVR:
        return self._svm_estimator

    @svm_estimator.setter
    def svm_estimator(self, svm_estimator: SVR | None) -> None:
        if svm_estimator is None:
            svm_estimator = SVR(kernel="rbf", gamma="scale")
        self._validate_type(value = svm_estimator, dtype = (SVR, ), msg = f"svm_estimator should be an SVR. Passed Type: {type(svm_estimator)}")
        self._svm_estimator = svm_estimator