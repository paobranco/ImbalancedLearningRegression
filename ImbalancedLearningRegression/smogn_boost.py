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


def smogn_boost(T, data, perc, pert, replace, y, threshold):

    # arguments / inputs
    # T: number of iterations, user inputted integer value
    # data,  # training set
    # perc,  # over / under sampling
    # pert,  # perturbation / noise percentage
    # replace,  # sampling replacement (bool)
    # y, # response variable y by name (string)
    # threshold:  error threshold defined by the user

    # set an initial iteration
    iteration = 1

    # loop while iteration is less than user provided iterations
    while iteration <= T:

        # this is the initiial iteration of smogn, calculating it for the bumps, giving new data oversampled
        dt_over_sampled = gn(data=data, index=list(), pert=pert, replace=replace)

        # this is to call the decision tree, predict regression value for y, and return the predicted values
        dt_data = tree.DecisionTreeRegressor()
        dt_data_predictions = dt_data.predict(dt_over_sampled, y)

        # calculating error rate

        #initialize error rate
        error = 0

            for each y in dt_data:
                if

        error = abs((dt_data[y] - data_new[y])/dt_data[y])

        beta = pow(error, 2)

        # update distribution with new weights, weights are beta
        dt_updated = pd.concat([perc, beta], ignore_index=True)

        # apply normalization factor using sci kit learn
        dt_normalized = preprocessing.normalize(dt_updated, max)

        # iteration count
        iteration = iteration + 1


    ):
