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


def smogn_boost(TotalIterations, data, perc, pert, replace, k, y, threshold):

    # arguments / inputs
    # TotalIterations: number of iterations, user inputted integer value
    # data,  # training set
    # perc,  # over / under sampling
    # pert,  # perturbation / noise percentage
    # replace,  # sampling replacement (bool)
    # y, # response variable y by name (string)
    # threshold:  error threshold defined by the user

    # set an initial iteration
    iteration = 1
    
    # set distribution as 1/m examples, which is length of data -1, as one of them is the target variable y
    examples = 1/(len(data)-1)
    dt_distribution = []
    for i in len(data):
        dt_distribution[i] = examples

    # loop while iteration is less than user provided iterations
    while iteration <= TotalIterations:

        # this is the initial iteration of smogn, calculating it for the bumps, giving new data oversampled
        dt_over_sampled = smogn(data=data, y, k = 5, pert = 0.02, replace=replace,)

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
        
        for y in dt_data_predictions: # verify for each of the y, what is the error
            if error > threshold:
                error = dt_data_predictions[perc]
                                      
        # beta is the update parameter of weights based on the error rate calculated
        beta = pow(error, 2)

        # update the distribution
        for i in dt_distribution:
            if error <= threshold:
                dt_distribution[i] = beta
            else:
                dt_distribution[i] = 1
        
        # update distribution with new weights, weights are beta
        # dt_updated = pd.concat([perc, beta], ignore_index=True)

        # apply normalization factor using sci kit learn
        dt_normalized = preprocessing.normalize(dt_data, max)

        # iteration count
        iteration = iteration + 1
    
    return smogn_boost