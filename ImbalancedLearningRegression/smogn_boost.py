# load dependencies - third party
import numpy as np
import pandas as pd
import sklearn
import math
from ImbalancedLearningRegression.smogn import smogn
from sklearn import tree
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# load dependencies - internal
from ImbalancedLearningRegression.phi import phi
from ImbalancedLearningRegression.phi_ctrl_pts import phi_ctrl_pts
from ImbalancedLearningRegression.over_sampling_gn import over_sampling_gn
from ImbalancedLearningRegression.gn import gn

## synthetic minority over-sampling technique for regression with gaussian noise with boost (based on SMOTEBoost using Adaboost)

def smogn_boost(

    ## main arguments / inputs
    data,                       ## training dataset (pandas dataframe)
    test_data,                  ## test dataset (pandas dataframe)
    y,                          ## response variable y by name (string)
    TotalIterations,            ## total number of iterations (pos int)
    error_threshold = 0.2,      ## error threshold (pos real)
    pert = 0.02,                ## perturbation / noise percentage (pos real)
    replace = False,            ## sampling replacement (bool)
    k = 5,                      ## num of neighs for over-sampling (pos int)
    samp_method = "balance",     ## over / under sampling ("balance" or extreme")
    drop_na_col = True,       ## auto drop columns with nan's (bool)
    drop_na_row = True,       ## auto drop rows with nan's (bool)

    ## phi relevance function arguments / inputs
    rel_thres = 0.5,                  ## relevance threshold considered rare (pos real)

    ):

    """
    TO DO: Add description and references
    # Look at https://github.com/nunompmoniz/ReBoost/blob/master/R/Functions.R
    """
    
    ## storing the original training data
    original_data = data
        
    # read the test data and split features (X) and target value (Y), reference: https://subscription.packtpub.com/book/data/9781838552862/1/ch01lvl1sec10/train-and-test-data
    X_test = test_data.drop(y, axis = 1)
    Y_test = test_data[y]

    # read the training data and split features (X) and target value column (Y)
    X_data = data.drop(y, axis = 1)
    Y_data = data[y]

    # set an initial iteration
    iteration = 1

    ## may initialize as a list instead after more testing, initializing as a numpy array causes issues with NaN for calculations
    # set an array of results, beta values, and decision tree predictions based on x_test
    # beta = np.empty(TotalIterations)
    
    result = []
    beta = []
    dt_test_predictions = []
    # dt_test_predictions = np.ones(TotalIterations)
    
    print("Dt Test Predictions: ", dt_test_predictions)
    
    # Dt(i) set distribution as 1/m weights, which is length of training data -1, as one of them is the target variable y
    weights = 1/(len(data))
    
    print("Weights: ", weights)

    dt_distribution = np.zeros(len(data))
    for i in range(len(data)):
        dt_distribution[i] = weights

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

    ## store original feature headers and
    ## encode feature headers to index position
    feat_names = list(data.columns)
    data.columns = range(d)

    ## sort response variable y by ascending order
    yDF = pd.DataFrame(data[d - 1])
    yDF_sort = yDF.sort_values(by = d - 1)
    yDF_sort = yDF_sort[d - 1]

    # calling phi control
    pc = phi_ctrl_pts(yDF_sort)

    print("phi control: ", pc)
    
    # calling only the control points (third value) from the output
    rel_ctrl_pts_rg = pc["ctrl_pts"]
    
    print("Phi Control Points: ", rel_ctrl_pts_rg)

    # loop while iteration is less than user provided iterations
    while iteration <= TotalIterations:

        # use initial training data set provided by user to obtain oversampled dataset using SMOGN, calculating it for the bumps
        dt_over_sampled = smogn(data=original_data, y=y, k=k)

        # splitting oversampled data for subsequent training data use below
        df_oversampled = pd.DataFrame(dt_over_sampled)
        x_oversampled = df_oversampled.drop(y, axis = 1)
        y_oversampled = df_oversampled[y]

        print("y_oversampled: ", y_oversampled)
        
        # calls the decision tree and use it to achieve a new model, predict regression value for y (target response variable), and return the predicted values
        dt_model = tree.DecisionTreeRegressor()

        # train decision tree classifier
        dt_model = dt_model.fit(x_oversampled, y_oversampled)

        # predict the features in user provided data
        dt_data_predictions = dt_model.predict(X_data)

        # predict the features in user provided test data and add them to the dt_test_predictions array
        # dt_test_predictions = np.concatenate([dt_test_predictions, dt_model.predict(X_test)])
        dt_test_predictions.append(dt_model.predict(X_test))
        
        print("dt test predictions: ", dt_test_predictions)

        # initialize model error rate & epsilon t value
        model_error = np.zeros(len(dt_data_predictions))
        epsilon_t = 0

        # calculate the model error rate of the new model achieved earlier, as the delta between original dataset and predicted oversampled dataset
        # for each y in the dataset, calculate whether it is greater/lower than threshold and update accordingly
        for i in range(len(dt_data_predictions)):
            model_error[i] = abs((Y_data[i] - dt_data_predictions[i])/Y_data[i])
        
        print("Model Error: ", model_error[i])

        for i in range(len(dt_data_predictions)):
            if model_error[i] > error_threshold:
                epsilon_t = epsilon_t + dt_distribution[i]

        print("Dt Data Predictions: ", dt_data_predictions)
        print("Dt Distribution: ", dt_distribution)
        print("Epsilon T: ", epsilon_t)

        # set curr_beta as the beta value for each iteration, it is the update parameter of weights based on the model error rate calculated
        # beta = np.append(beta, curr_beta)
        curr_beta = round(pow(epsilon_t, 2), 10)
        beta.append(curr_beta)
        
        print("Current Beta: ", epsilon_t)     
        print("beta: ", beta)
        
        #print(dt_distribution)
        # update the distribution weights
        for i in range(len(dt_distribution)):
            if model_error[i] <= error_threshold:
                dt_distribution[i] = dt_distribution[i] * curr_beta
        
        # normalize the distribution
        dt_normalized = preprocessing.normalize(dt_distribution.reshape(1,-1), norm="max")
        
        print("Dt Normalized: ", dt_normalized)

        # iteration count
        iteration += 1

    # calculate result
    #numer = 0
    denom = 0  
    
    # beta 1 * all prediction values in array of first predictions, beta 2 *..... will still have as many arrays as iterations
    # add the arrays together, matching index of each array
    # divide this array by the constant calculated in denom
    # return 1 array
    
    numerator = []
    denominator = 0
    
    #print("len dt test pred: ", len(dt_test_predictions)) 
    
    ## calculating numerator, looping through the array of arrays, outputs an array of arrays
    for i in range(len(dt_test_predictions)):
        calculation = []
        for j in range(len(dt_test_predictions[i])):
            calculation.append((math.log((1/beta[i]) * dt_test_predictions[i][j])))
            result.append(calculation)
    
    print("calculation: ", calculation)
    
    ## sum all the arrays calculated in numerator

    for i in range(len(calculation)):
        for j in range(len(calculation[i])):
            numerator.append((calculation[i]) + calculation[j][i])
            result.append(numerator)
    
    print("numerator: ", numerator)
    
    
    #for i in range((calculation[0])):
     #   array_sum = 0.0
      #  for array in calculation:
       #     array_sum += array[j][i]
        #numerator.append(array_sum)
    
    ## calculate denominator
    for i in range(len(beta)):
        denominator += (math.log((1/beta[i])))
        
    ## calculate result      
    for i in range(len(numerator)):
        result.append(numerator[i]/denominator)  
    print(result)
    return result