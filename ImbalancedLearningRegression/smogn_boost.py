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

    ## pre-process missing values
    if bool(drop_na_col) == True:
        data = data.dropna(axis = 1)  ## drop columns with nan's

    og_data = data
    # read the test data and split features (X) and target value (Y), reference: https://subscription.packtpub.com/book/data/9781838552862/1/ch01lvl1sec10/train-and-test-data
    X_test = test_data.drop(y, axis = 1)
    Y_test = test_data[y]

    # read the training data and split features (X) and target value (Y)
    X_data = data.drop(y, axis = 1)
    Y_data = data[y]

    print("B")

    # set an initial iteration
    iteration = 1

    # set an array of results, beta values, and decision tree predictions based on x_test
    result = np.empty(TotalIterations, dtype=int)
    beta = np.empty(TotalIterations, dtype=int)
    dt_test_predictions = np.empty(TotalIterations, dtype=int)

    print("C")

    # Dt(i) set distribution as 1/m weights, which is length of training data -1, as one of them is the target variable y
    weights = 1/(len(data))

    dt_distribution = np.zeros(len(data))
    for i in range(len(data)):
        dt_distribution[i] = weights

    print("D")

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

    print("E")

    ## store original feature headers and
    ## encode feature headers to index position
    feat_names = list(data.columns)
    data.columns = range(d)

    ## sort response variable y by ascending order
    yDF = pd.DataFrame(data[d - 1])
    yDF_sort = yDF.sort_values(by = d - 1)
    yDF_sort = yDF_sort[d - 1]

    print("F")

    # calling phi control
    pc = phi_ctrl_pts(yDF_sort)

    # calling only the control points (third value) from the output
    rel_ctrl_pts_rg = pc["ctrl_pts"]

    print("G")

    # loop while iteration is less than user provided iterations
    while iteration <= TotalIterations:

        print(og_data)
        # use initial training data set provided by user to obtain oversampled dataset using SMOGN, calculating it for the bumps
        dt_over_sampled = smogn(data=og_data, y="SalePrice", k=k)

        print("H")

        # splitting oversampled data for subsequent training data use below
        df_oversampled = pd.DataFrame(dt_over_sampled)
        x_oversampled = df_oversampled.drop(yDF_sort, axis = 1)
        y_oversampled = df_oversampled[yDF_sort]

        print("K")

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
