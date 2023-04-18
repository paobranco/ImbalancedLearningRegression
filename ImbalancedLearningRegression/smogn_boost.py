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

# synthetic minority over-sampling technique for regression with gaussian noise with boost (based on SMOTEBoost using Adaboost)
# Look at https://github.com/nunompmoniz/ReBoost/blob/master/R/Functions.R

def smogn_boost(data, test_data, y, TotalIterations, pert, replace, k, error_threshold, rel_thres, samp_method = "balance"):

    # arguments/inputs
    
    # data: training set (pandas dataframe)
    # test_data: test data (pandas dataframe)
    # y: response variable y by name (string)
    # TotalIterations: user defined total number of iterations (pos int)
    # pert: perturbation / noise percentage
    # replace: sampling replacement (bool)
    # k: num of neighs for over-sampling (pos int)
    # error_threshold: user defined error threshold 
    # rel_thres: user defined relevance threshold 
    # samp_method: "balance or extreme" - sampling method is perc
    
    print("A")
    
    # read the test data and split features (X) and target value (Y), reference: https://subscription.packtpub.com/book/data/9781838552862/1/ch01lvl1sec10/train-and-test-data
    X_test = test_data.drop(y, axis = 1)
    Y_test = test_data[y]
    
    print("B")
    
    # read the training data and split features (X) and target value (Y)
    X_data = data.drop(y, axis = 1)
    Y_data = data[y]

    print("C")
    
    # set for clarity, name of target variable not data
    y_train = y
    
    # set an initial iteration
    iteration = 1
    
    print("E")
    
    # set an array of results, beta values, and decision tree predictions based on x_test
    result = np.empty(TotalIterations, dtype=int)
    beta = np.empty(TotalIterations, dtype=int)
    dt_test_predictions = np.empty(TotalIterations, dtype=int)
    
    print("F")
    
    # Dt(i) set distribution as 1/m weights, which is length of training data -1, as one of them is the target variable y 
    weights = 1/(len(data))
    print (weights)
    dt_distribution = np.zeros(len(data))
    for i in range(len(data)):
        dt_distribution[i] = weights
           
    print("G")
    print(dt_distribution)
   
    ## store data dimensions
    n = len(data)
    d = len(data.columns)
    
    print("H")
    
    ## store original data types
    feat_dtypes_orig = [None] * d
    
    print("I")
    
    for j in range(d):
        feat_dtypes_orig[j] = data.iloc[:, j].dtype
        
    print("J")
    
    ## determine column position for response variable y
    y_col = data.columns.get_loc(y)
    
    print("K")
    
    ## move response variable y to last column
    if y_col < d - 1:
        cols = list(range(d))
        cols[y_col], cols[d - 1] = cols[d - 1], cols[y_col]
        data = data[data.columns[cols]]
        
    print("L")
    
    ## store original feature headers and
    ## encode feature headers to index position
    feat_names = list(data.columns)
    data.columns = range(d)
    
    print("M")
    
    ## sort response variable y by ascending order
    y = pd.DataFrame(data[d - 1])
    y_sort = y.sort_values(by = d - 1)
    y_sort = y_sort[d - 1]
    
    print("N")


    # calling phi control
    pc = phi_ctrl_pts(y_sort)
    
    ## this is not printing, issue with PC
    print(pc)
    
    # calling only the control points (third value) from the output
    rel_ctrl_pts_rg = pc["ctrl_pts"]
    
    print("O")
    
    # loop while iteration is less than user provided iterations
    while iteration <= TotalIterations:

        # use initial training data set provided by user to obtain oversampled dataset using SMOGN, calculating it for the bumps
        dt_over_sampled = smogn(data=data, y = y_train, k = k)
        
        print("P")

        # splitting oversampled data for subsequent training data use below
        df_oversampled = dt_over_sampled, header = 0
        x_oversampled = df_oversampled.drop(y_train, axis = 1)
        y_oversampled = df_oversampled[y_train]

        print("Q")
        
        # calls the decision tree and use it to achieve a new model, predict regression value for y (target response variable), and return the predicted values
        dt_model = tree.DecisionTreeRegressor()
        
        # train decision tree classifier
        dt_model = dt_model.fit(x_oversampled, y_oversampled)
        
        # predict the features in user provided data
        dt_data_predictions = dt_model.predict(X_data)
        
        # predict the features in user provided test data
        dt_test_predictions.append(dt_model.predict(X_test))

        # initialize model error rate & epsilon t value
        model_error = np.zeros(len(dt_data_predictions))
        epsilon_t = 0

        # calculate the model error rate of the new model achieved earlier, as the delta between original dataset and predicted oversampled dataset
        # for each y in the dataset, calculate whether it is greater/lower than threshold and update accordingly
        for i in range(len(dt_data_predictions)):
            model_error[i] = abs((Y_data[i] - dt_data_predictions[i])/Y_data[i])
        
        for i in range(len(dt_data_predictions)):
            if model_error[i] > error_threshold:
                epsilon_t = epsilon_t + dt_distribution[i]
                                      
        # beta is the update parameter of weights based on the model error rate calculated
        beta.append(pow(epsilon_t, 2))

        # update the distribution weights
        for i in dt_distribution:
            if model_error[i] <= error_threshold:
                dt_distribution[i] = dt_distribution[i] * beta
            else:
                dt_distribution[i] = dt_distribution[i]

        # normalize the distribution 
        dt_normalized = preprocessing.normalize(dt_distribution, max)

        # iteration count
        iteration += 1
    
    # calculate result
    numer = 0
    denom = 0
    
    for b, i in zip(beta, dt_test_predictions):
            numer += math.log(1/b) * i
            denom += math.log(1/b)
    return numer/denom