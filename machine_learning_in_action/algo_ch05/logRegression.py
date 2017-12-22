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
        f = open("test_set.txt")

        for line in f.readlines():
            array_of_lines = line.strip().split()
            dataset.append([1.0, float(array_of_lines[0]), float(array_of_lines[1])])
            labels.append(int(array_of_lines[2]))

        return dataset, labels

    # ================ METHOD TO CALCULATE SIGMOID VALUE FROM X-INPUT ================
    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    # ========== METHOD TO OPTIMIZE REGRESSION WEIGHTS USING GRADIENT ASCENT =========
    def optimize_gradient_ascent(self, input_dataset, class_labels):
        dataset = np.mat(input_dataset)
        labels = np.mat(class_labels).transpose()
        m, n = np.shape(dataset)
        alpha = 0.001
        maximum_repetitions = 500
        weights = np.ones((n, 1))

        for _ in range(maximum_repetitions):
            sig = self.sigmoid(dataset * weights)
            error = (labels - sig)
            weights += alpha * dataset.transpose() * error

        return weights


# ====================================================================================
# ================================ MAIN RUN FUNCTION =================================
# ====================================================================================


def main():
    # Track starting time of program
    t0 = t()

    # Initialize class instance of the logistic regression optimization algorithm
    logRegres = logistic_Regression_Optimization_Algorithm()

    # Track ending time of program and determine overall program runtime
    t1 = t()
    delta = (t1 - t0) * 1000

    print("Real program runtime is {0:.4g} milliseconds.\n".format(delta))
    return

if __name__ == "__main__":
    main()