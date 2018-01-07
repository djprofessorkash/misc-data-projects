"""
NAME:               regression.py (data_projects/machine_learning_in_action/algo_ch08/)

DESCRIPTION:        Python class application of ???.

                    ???

                    All source code is available at www.manning.com/MachineLearningInAction. 

USE CASE(S):        Numeric and nominal values

ADVANTAGE(S):       Relatively easy to interpret results
                    Computationally inexpensive

DISADVANTAGE(S):    Poor model for nonlinear data

NOTE:               Original source code is in Python 2, but my code is in Python 3.

CREDIT:             Machine Learning in Action (Peter Harrington)
"""

# ====================================================================================
# ================================ IMPORT STATEMENTS =================================
# ====================================================================================


from time import time as t                  # Package for tracking modular and program runtime
import matplotlib.pyplot as plt             # Module for MATLAB-like data visualization capability
import numpy as np                          # Library for simple linear mathematical operations


# ====================================================================================
# ================================ CLASS DEFINITION ==================================
# ====================================================================================

class linear_Regression(object):

    # ======================== CLASS INITIALIZERS/DECLARATIONS =======================
    def __init__(self, TIME_I):
        self.TIME_I = TIME_I                            # Initial time measure for runtime tracker

    # ======================== METHOD TO LOAD IN SAMPLE DATASET ======================
    def load_sample_data(self, FILENAME):
        num_of_features = len(open(FILENAME).readline().split("\t")) - 1        # Produces integer to hold number of data features
        
        dataset = []                                                            # Initializes sample dataset
        labels = []                                                             # Initializes class label vector for sample data

        f = open(FILENAME)                                                      # Opens file holding sample data

        # Iterates through each line in the sample data file
        for line in f.readlines():
            line_arr = []
            current_line = line.strip().split("\t")                             # Defines the current line as formatted line string

            # Iterates through the number of data features
            for iterator in range(num_of_features):
                line_arr.append(float(current_line[iterator]))                  # Produces array of lines from dataset
            
            dataset.append(line_arr)                                            # Creates dataset from line array
            labels.append(float(current_line[-1]))                              # Creates class label vector from line array

        """ print("\nSAMPLE DATASET IS: \n{}\n\nCLASS LABEL VECTOR FOR SAMPLE DATA IS: \n{}\n".format(dataset, labels)) """
        return dataset, labels

    # ======= METHOD TO CALCULATE STANDARD LINEAR REGRESSION PREDICTION FACTORS ======
    def standard_linear_regression_calculation(self, x_arr, y_arr):
        # Creates formatted matrices for x- and y-data
        x_mat = np.mat(x_arr)
        y_mat = np.mat(y_arr).T

        # Creates linearly transformational x-data matrix 
        xTx = x_mat.T * x_mat

        # Checks that transformational x-data matrix is not singular
        if np.linalg.det(xTx) == 0.0:
            return print("\nTHIS MATRIX IS SINGULAR: INVERSE CANNOT BE CALCULATED.\n")

        # Calculates prediction factors as multiplier constants from transformational matrices
        prediction_factors = xTx.I * (x_mat.T * y_mat)
        print("\nPREDICTION FACTORS FOR THE SAMPLE DATA ARE: \n{}\n".format(prediction_factors))
        return prediction_factors

    # ================ METHOD TO BENCHMARK RUNTIME OF SPECIFIC METHOD ================
    def track_runtime(self):
        # Track ending time of program and determine overall program runtime
        TIME_F = t()
        delta = TIME_F - self.TIME_I

        if delta < 1.5:
            return print("\nReal program runtime is {0:.4g} milliseconds.\n".format(delta * 1000))
        return print("\nReal program runtime is {0:.4g} seconds.\n".format(delta))


# ====================================================================================
# ================================ MAIN RUN FUNCTION =================================
# ====================================================================================


def main():
    # Track starting time of program
    TIME_I = t()

    # Initialize class instance of linear regression model
    lin_regr = linear_Regression(TIME_I)

    """
    # Testing the calculator for prediction factors from sample data
    x_arr, y_arr = lin_regr.load_sample_data("./sample0.txt")
    prediction_factors = lin_regr.standard_linear_regression_calculation(x_arr, y_arr)
    """

    # Testing to see predictive equation calculation
    x_arr, y_arr = lin_regr.load_sample_data("./sample0.txt")
    prediction_factors = lin_regr.standard_linear_regression_calculation(x_arr, y_arr)
    x_mat = np.mat(x_arr)
    y_mat = np.mat(y_arr)
    y_hat = x_mat * prediction_factors

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_mat[:, 1].flatten().A[0], y_mat.T[:, 0].flatten().A[0])

    x_copy = x_mat.copy()
    x_copy.sort(0)
    y_hat = x_copy * prediction_factors
    ax.plot(x_copy[:, 1], y_hat)
    plt.show()

    return print("\nAdaBoost class meta-algorithm is done.\n")

if __name__ == "__main__":
    main()