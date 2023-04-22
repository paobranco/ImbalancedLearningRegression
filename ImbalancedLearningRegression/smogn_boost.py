# load dependencies - third party
import numpy as np
import pandas as pd
import math
from ImbalancedLearningRegression.smogn import smogn
from sklearn import tree
from sklearn import preprocessing

# load dependencies - internal
from ImbalancedLearningRegression.phi_ctrl_pts import phi_ctrl_pts

## synthetic minority over-sampling technique for regression with gaussian noise with boost (based on SMOTEBoost using Adaboost)

def smogn_boost(

    ## main arguments / inputs
    data,                       ## training dataset (pandas dataframe)
    test_data,                  ## test dataset (pandas dataframe)
    y,                          ## response variable y by name (string)
    totalIterations,            ## total number of iterations (pos int)
    error_threshold = 0.2,      ## error threshold (pos real)
    pert = 0.02,                ## perturbation / noise percentage (pos real)
    replace = False,            ## sampling replacement (bool)
    k = 5,                      ## num of neighs for over-sampling (pos int)
    samp_method = "balance",     ## over / under sampling ("balance" or extreme")
    drop_na_col = True,       ## auto drop columns with nan's (bool)
    drop_na_row = True,       ## auto drop rows with nan's (bool)

    ## phi relevance function arguments / inputs
    rel_thres = 0.5,                  ## relevance threshold considered rare (pos real)
    rel_ctrl_pts_rg = None
    ):

    """
    The main function applies a boosting step to SMOGN. 
    
    The boosting algorithm was taken from Algorithm 1, shown in 
    SMOTEBoost for Regression: Improving the Prediction of Extreme Values.
    
    TO DO: Add description and references
    # Look at https://github.com/nunompmoniz/ReBoost/blob/master/R/Functions.R
    reference: https://subscription.packtpub.com/book/data/9781838552862/1/ch01lvl1sec10/train-and-test-data
    """
    
    ## storing the original training data
    original_data = data.copy()
        
    ## read the test data and split features (X) and target value (Y)
    X_test = test_data.drop(y, axis = 1)
    Y_test = test_data[y]

    ## read the training data and split features (X) and target value column (Y)
    X_data = data.drop(y, axis = 1)
    Y_data = data[y]

    ## initialize empty list of beta values, store the beta value for all iterations
    ## initialize an empty list of test predictions, which will store arrays, an array of predictions for each iteration based on test values
    total_betas = []
    dt_test_predictions = []
    
    ## Dt(i) set distribution as 1/m weights, which is length of training data -1, as one of them is the target variable y
    weights = 1/(len(data))
    
    dt_distribution = np.zeros(len(data))
    for i in range(len(data)):
        dt_distribution[i] = weights
        
    print("Weights: ", weights)
    print("Dt Distribution: ", dt_distribution)

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

    ## store original feature headers and encode feature headers to index position
    feat_names = list(data.columns)
    data.columns = range(d)

    ## sort response variable y by ascending order
    yDF = pd.DataFrame(data[d - 1])
    yDF_sort = yDF.sort_values(by = d - 1)
    yDF_sort = yDF_sort[d - 1]

    ## loop through totalIterations
    for _ in range(totalIterations):

        ## use training data provided by user to obtain oversampled dataset using SMOGN, calculating it for the bumps
        dt_over_sampled = smogn(data=original_data, y = y, k = k, rel_ctrl_pts_rg = rel_ctrl_pts_rg)

        ## split the oversampled data for subsequent training data use
        df_oversampled = pd.DataFrame(dt_over_sampled)
        x_oversampled = df_oversampled.drop(y, axis = 1)
        y_oversampled = df_oversampled[y]
        
        ## call the decision tree and use it to achieve a new model, predict regression value for y (target response variable), and return the predicted values
        dt_model = tree.DecisionTreeRegressor()

        ## train decision tree classifier
        dt_model = dt_model.fit(x_oversampled, y_oversampled)

        ## predict the features in user provided training data
        dt_data_predictions = dt_model.predict(X_data)

        ## predict the features in user provided test data and add them to the test predictions list
        dt_test_predictions.append(dt_model.predict(X_test))
        
        print("Dt Test Predictions: ", len(dt_test_predictions))

        ## initialize model error rate, calculates model error for each value
        ## initialize epsilon_t value
        model_error = np.zeros(len(dt_data_predictions))
        epsilon_t = 0

        ## calculate the model error rate of the new model achieved earlier, as the delta between original dataset and predicted oversampled dataset
        for i in range(len(dt_data_predictions)):
            model_error[i] = abs((Y_data[i] - dt_data_predictions[i])/Y_data[i])
        
            print("Model Error: ", model_error[i])
            
        ## for each y in the dataset, calculate whether it is greater/lower than threshold and update accordingly
        for i in range(len(dt_data_predictions)):
            if model_error[i] > error_threshold:
                epsilon_t = epsilon_t + dt_distribution[i]

        print("Dt Data Predictions: ", dt_data_predictions)
        print("Dt Distribution: ", dt_distribution)
        print("Epsilon T: ", epsilon_t)

        ## set curr_beta as the beta value for each iteration, it is the update parameter of weights based on the model error rate calculated
        curr_beta = round(pow(epsilon_t, 2), 10)
        total_betas.append(curr_beta)
        
        print("Current Beta: ", epsilon_t)     
        print("Total Betas: ", total_betas)
        
        ## update the distribution weights if model error for each index is lower than/equal to the error threshold
        for i in range(len(dt_distribution)):
            if model_error[i] <= error_threshold:
                dt_distribution[i] = dt_distribution[i] * curr_beta
        
        ## normalize the distribution
        dt_normalized = preprocessing.normalize(dt_distribution.reshape(1,-1), norm="max")
        
        print("Dt Normalized: ", dt_normalized)
    
    ## calculate the final result as numerator and denominator    
    ## calculating numerator, looping through index in the array of arrays, outputs an array of arrays within calculations
    calculations = []
    for idx, predictions in enumerate(dt_test_predictions):
        calculations.append([math.log(1/total_betas[idx]) * prediction for prediction in predictions])

    ## sums arrays within calculations (vector addition) and outputs to numerator
    numerator = [0] * len(calculations[0])
    for vector in calculations:
        for idx, value in enumerate(vector):
            numerator[idx] += value

    ## calculate denominator, looking at each beta in total betas for all iterations
    denominator = sum([math.log(1/beta) for beta in total_betas])

    ## calculates and returns the final result
    result = [value/denominator for value in numerator]
    print("Final Result: ", result)
    
    ## checking that the final result matches size of test predictions sub array
    print("Final Result length: ", len(result))
    for sub_arr in dt_test_predictions:
        print("Length of dt_test_prediction sub arrays: ", len(sub_arr))
        
    return result