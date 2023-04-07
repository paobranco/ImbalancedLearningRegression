# load dependencies - third party
import numpy as np
import pandas as pd
import sklearn
import math
import smogn as smogn
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

def smogn_boost(data, test_data, Y_test, TotalIterations, pert, replace, k, y, error_threshold, rel_thres, samp_method = "balance"):

    # arguments/inputs
    
    # data: training set
    # test_data: test data
    # Y_test: target variable from test data
    # TotalIterations: user defined total number of iterations (pos int)
    # pert: perturbation / noise percentage
    # replace: sampling replacement (bool)
    # k: num of neighs for over-sampling (pos int)
    # y: response variable y by name (string)
    # error_threshold: user defined error threshold 
    # rel_thres: user defined relevance threshold 
    # samp_method: "balance or extreme" - sampling method is perc

    # pre-processing the test_data, reference: https://subscription.packtpub.com/book/data/9781838552862/1/ch01lvl1sec10/train-and-test-data
    # read the test data and split features (X) and target value (Y)
    df = pd.read_csv(test_data, header = 0)
    X_test = df.drop('Y', axis = 1)
    X_test.head()
    Y_test = df['Y_test']

    # set for clarity
    y_train = y
    
    # set an initial iteration
    iteration = 1
    
    # set an initial result
    result = 0
    
    # Dt(i) set distribution as 1/m weights, which is length of data -1, as one of them is the target variable y 
    weights = 1/(len(data))
    dt_distribution = []
    for i in len(data):
        dt_distribution[i] = weights

    # calling phi control
    pc = phi_ctrl_pts (y=y, method="manual", xtrm_type = "both", coeff = 1.5, ctrl_pts=any)
    
    # calling only the control points (third value) from the output
    rel_ctrl_pts_rg = pc[2]
    
    # loop while iteration is less than user provided iterations
    while iteration <= TotalIterations:

        # use initial training data set provided by user to obtain oversampled dataset using SMOGN, calculating it for the bumps
        dt_over_sampled = smogn(data=data, y_train = y_train, k = 5, pert = pert, replace=replace, rel_thres = rel_thres, rel_method = "manual", rel_ctrl_pts_rg = rel_ctrl_pts_rg)

        # splitting oversampled data for subsequent training data use below
        df = dt_over_sampled, header = 0
        x_oversampled = df.drop('y', axis = 1)
        x_oversampled.head()
        y_oversampled = df['y_train']

        # calls the decision tree and use it to achieve a new model, predict regression value for y (target response variable), and return the predicted values
        dt_model = tree.DecisionTreeRegressor()
        
        # train decision tree classifier
        dt_model = dt_model.fit(x_oversampled, y_oversampled)
        
        # predict the response for 
        dt_data_predictions = dt_model.predict(###)

        # initialize error rate
        model_error = 0

        # calculate the error rate of the new model achieved earlier, as the delta between original dataset and predicted oversampled dataset
        # for each y in the dataset, calculate whether it is greater/lower than threshold and update accordingly
        error = abs((data[y] - dt_data_predictions[y])/data[y])
        
        for i in range(1, len[dt_data_predictions], 1):
            if model_error[i] > error_threshold:
                epsilon_t = epsilon_t + dt_distribution[i]
                                      
        # beta is the update parameter of weights based on the model error rate calculated
        beta = pow(epsilon_t, 2)

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
        
        # calculate beta log
        numer = (math.log(1/beta))*(dt_data_predictions[i]) 
        denom = (math.log(1/beta))
       
        # calculate & return result
        result += (numer/denom)
    
    return result
        
    # split oversampled data into a training and test set
    # x_train, X_test, y_train, Y_test = train_test_split(x_oversampled, X_test, y_oversampled, Y_test, test_size=0.3, random_state=0) # 70% training and 30% test
        
    # split testing data set into features and target
    # store data dimensions
    #n = len(test_data)
    #d = len(test_data.columns)
    # determine column position for response variable Y
    #Y_col = test_data.columns.get_loc(Y)
    
    # move response variable Y to last column
    #if Y_col < d - 1:
     #   cols = list(range(d))
      #  cols[Y_col], cols[d - 1] = cols[d - 1], cols[Y_col]
       # data = data[test_data.columns[cols]]
    
    # store original feature headers and
    # encode feature headers to index position
    #feat_names = list(test_data.columns)
    #data.columns = range(d)
    
    # sort response variable Y by ascending order
    #Y = pd.DataFrame(test_data[d - 1])
    #Y_sort = Y.sort_values(by = d - 1)
    #Y_sort = Y_sort[d - 1]