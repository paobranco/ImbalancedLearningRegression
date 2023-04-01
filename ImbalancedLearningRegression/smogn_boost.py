# load dependencies - third party
import numpy as np
import pandas as pd
import sklearn
import math
import smogn as smogn
from sklearn import tree
from sklearn import preprocessing

# load dependencies - internal
from ImbalancedLearningRegression.phi import phi
from ImbalancedLearningRegression.phi_ctrl_pts import phi_ctrl_pts
from ImbalancedLearningRegression.over_sampling_gn import over_sampling_gn
from ImbalancedLearningRegression.gn import gn

# synthetic minority over-sampling technique for regression with gaussian noise with boost (based on SMOTEBoost using Adaboost)

# ****need a train and a test set****
def smogn_boost(TotalIterations, data, pert, replace, k, y, error_threshold, rel_thres, samp_method = "balance"):

    # arguments / inputs
    # TotalIterations: number of iterations, user inputted integer value
    # data,  # training set
    # pert,  # perturbation / noise percentage
    # replace,  # sampling replacement (bool)
    # y, # response variable y by name (string)
    # error_threshold:  error threshold defined by the user
    # samp_method: "balance or extreme" - samp method is perc
    # rel_thres: relevance threshold defined by the user

    # set an initial iteration
    iteration = 1
    
    # set distribution as 1/m weights, which is length of data -1, as one of them is the target variable y
    
    # Look at https://github.com/nunompmoniz/ReBoost/blob/master/R/Functions.R
    # ******need to use number of rows here******
    # convert data to data frame here
    weights = 1/(len(data))
    dt_distribution = []
    for i in len(data):
        dt_distribution[i] = weights

    # calling phi control
    pc = phi_ctrl_pts (y=y, method="manual", xtrm_type = "both", coeff = 1.5, ctrl_pts=any)
    # calling the control points only from the output
    rel_ctrl_pts_rg = pc[3]
    
    # loop while iteration is less than user provided iterations
    while iteration <= TotalIterations:

        # this is the initial iteration of smogn, calculating it for the bumps, giving new data oversampled
        dt_over_sampled = smogn(data=data, y = y, k = 5, pert = pert, replace=replace, rel_thres = rel_thres, rel_method = "manual", rel_ctrl_pts_rg = rel_ctrl_pts_rg)

        # this is to call the decision tree and use it to achieve a new model, predict regression value for y (target response variable), and return the predicted values
        dt_model = tree.DecisionTreeRegressor()
        
        #check if I need to separate features and target
        dt_model = dt_model.fit(dt_over_sampled) 
        dt_data_predictions = dt_model.predict(y)

        #initialize error rate
        error = 0

        # calculate the error rate of the new model achieved earlier, as the delta between original dataset and predicted oversampled dataset
        #for each y in the dataset, calculate whether it is greater/lower than threshold and update accordingly
        error = abs((data[y] - dt_data_predictions[y])/data[y])
        
        for i in range(1, len[dt_data_predictions], 1):
            if error[i] > error_threshold:
                epsilon_t = epsilon_t + dt_distribution[i]
                                      
        # beta is the update parameter of weights based on the error rate calculated
        beta = pow(epsilon_t, 2)

        # update the distribution
        for i in dt_distribution:
            if error[i] <= error_threshold:
                dt_distribution[i] = dt_distribution[i] * beta
            else:
                dt_distribution[i] = dt_distribution[i]

        # apply normalization factor using sci kit learn
        dt_normalized = preprocessing.normalize(dt_distribution, max)

        # iteration count
        iteration = iteration + 1

    # **** need to modify original data after the whole loop ****
    
    return smogn_boost