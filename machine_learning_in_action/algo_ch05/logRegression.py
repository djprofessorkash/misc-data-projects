"""
NAME:               logRegression.py (data_projects/machine_learning_in_action/algo_ch05/)

DESCRIPTION:        Python class application of the logistic regression optimizer algorithm.

                    ???

                    All source code is available at www.manning.com/MachineLearningInAction. 

USE CASE(S):        Numeric and nominal values

ADVANTAGE(S):       Computationally cheap
                    Relatively easy to implement
                    Easy to contextually understand learned results

DISADVANTAGE(S):    Sensitive and prone to overfitting
                    Potential for low accuracy

NOTE:               Original source code is Python 2, but my code is Python 3.

CREDIT:             Machine Learning In Action (Peter Harrington)
"""


# ====================================================================================
# ================================ IMPORT STATEMENTS =================================
# ====================================================================================


import numpy as np                          # Library for simple linear mathematical operations
import matplotlib.pyplot as plt             # Module for MATLAB-like data visualization capability
from time import time as t                  # Package for tracking modular and program runtime


# ====================================================================================
# ================================= CLASS DEFINITION =================================
# ====================================================================================


class logistic_Regression_Optimization_Algorithm(object):

    # ======================== CLASS INITIALIZERS/DECLARATIONS =======================
    def __init__(self):
        pass

    # ======================== METHOD TO LOAD DATASET FROM FILE ======================
    def load_dataset(self):
        dataset = []
        labels = []
        FILE = open("test_set.txt")

        # Produces dataset and class label vector from formatted dataset
        for line in FILE.readlines():
            array_of_lines = line.strip().split()
            dataset.append([1.0, float(array_of_lines[0]), float(array_of_lines[1])])
            labels.append(int(array_of_lines[2]))

        # print("GIVEN DATASET IS: \n{}\n".format(dataset))
        # print("GIVEN CLASS LABELS ARE: \n{}\n".format(labels))
        return dataset, labels

    # ================ METHOD TO CALCULATE SIGMOID VALUE FROM X-INPUT ================
    def sigmoid_distribution(self, x):
        # Calculates Sigmoid Z-value (functional output) from inputted X-value
        sig = 1.0 / (1 + np.exp(-x))

        # print("SIGMOID DISTRIBUTION VALUE IS: \n{}\n".format(sig))
        return sig

    # ========== METHOD TO OPTIMIZE REGRESSION WEIGHTS USING GRADIENT ASCENT =========
    def optimize_gradient_ascent(self, input_dataset, class_labels):
        dataset = np.mat(input_dataset)                 # Input dataset is array of features (columns) and training samples (rows)
        labels = np.mat(class_labels).transpose()       # Class label vector is linear transposition of input dataset
        num_rows, num_cols = np.shape(dataset)
        ALPHA = 0.001
        MAX_REPETITIONS = 500
        reg_weights = np.ones((num_cols, 1))            # Creates array of regression weights with same size as dataset columns

        # Iterates over sigmoid distribution to optimize training data regression weights
        for _ in range(MAX_REPETITIONS):
            sig = self.sigmoid_distribution(dataset * reg_weights)      # Recursively calls sigmoid function to maximize weights
            error = labels - sig
            reg_weights += ALPHA * dataset.transpose() * error

        # print("\nTOTAL RELATIVE ERRORS ACROSS SIGMOID DISTRIBUTION IS: \n{}\n".format(error))
        print("\nRELATIVE REGRESSION WEIGHTS FROM OPTIMIZATION ARE: \n{}\n".format(reg_weights))
        return reg_weights

    # ====== METHOD TO PLOT LOGISTIC REGRESSION LINE OF BEST FIT ACROSS DATASET ======
    def plot_line_of_best_fit(self, weights):
        reg_weights = 0


# ====================================================================================
# ================================ MAIN RUN FUNCTION =================================
# ====================================================================================


def main():
    # Track starting time of program
    TIME_I = t()

    # Initialize class instance of the logistic regression optimization algorithm
    logRegres = logistic_Regression_Optimization_Algorithm()

    # Test optimize_gradient_ascent() with sigmoid_distribution calculation and loading test data
    dataset, labels = logRegres.load_dataset()
    logRegres.optimize_gradient_ascent(dataset, labels)

    # Track ending time of program and determine overall program runtime
    TIME_F = t()
    DELTA = (TIME_F - TIME_I) * 1000

    print("Real program runtime is {0:.4g} milliseconds.\n".format(DELTA))
    return

if __name__ == "__main__":
    main()