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
    def plot_line_of_best_fit(self, dataset, labels, weights, TIME_I):
        reg_weights = np.array(weights)
        data_arr = np.array(dataset)
        num_rows = np.shape(data_arr)[0]

        # Predefines scatter X- and Y-coordinates
        x_coord1 = []
        y_coord1 = []
        x_coord2 = []
        y_coord2 = []

        # Creates X- and Y-coordinates based on class label vector's array of ones
        for iterator in range(num_rows):
            if int(labels[iterator]) == 1:
                x_coord1.append(data_arr[iterator, 1])
                y_coord1.append(data_arr[iterator, 2])
            else:
                x_coord2.append(data_arr[iterator, 1])
                y_coord2.append(data_arr[iterator, 2])

        # Initializes scatterplot space with given X- and Y-coordinate values
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(x_coord1, y_coord1, s = 30, c = "red", marker = "s")
        ax.scatter(x_coord2, y_coord2, s = 30, c = "green")

        # Creates domains for X and Y
        x = np.arange(-3.0, 3.0, 0.1)
        y = (-reg_weights[0] - reg_weights[1] * x) / reg_weights[2]

        # Plots X and Y domains with labels on axes to finalize scatterplot
        ax.plot(x, y)
        plt.xlabel("X1")
        plt.ylabel("Y1")

        # Runs runtime tracker for particular method
        self.track_runtime(TIME_I)

        print("DISPLAYING LOGISTIC REGRESSION BEST-FIT LINE AFTER 500 CYCLES OF GRADIENT ASCENT...\n")

        plt.show()
        return

    def track_runtime(self, TIME_I):
        # Track ending time of program and determine overall program runtime
        TIME_F = t()
        DELTA = (TIME_F - TIME_I) * 1000

        print("Real program runtime is {0:.4g} milliseconds.\n".format(DELTA))
        return


# ====================================================================================
# ================================ MAIN RUN FUNCTION =================================
# ====================================================================================


def main():
    # Track starting time of program
    TIME_I = t()

    # Initialize class instance of the logistic regression optimization algorithm
    logRegres = logistic_Regression_Optimization_Algorithm()

    # Test optimize_gradient_ascent() with sigmoid calculation and loading test data
    """
    dataset, labels = logRegres.load_dataset()
    logRegres.optimize_gradient_ascent(dataset, labels)
    logRegres.track_runtime(TIME_I)
    """

    # Test plot_line_of_best_fit() with loading test data
    dataset, labels = logRegres.load_dataset()
    weights = logRegres.optimize_gradient_ascent(dataset, labels)
    logRegres.plot_line_of_best_fit(dataset, labels, weights, TIME_I)

    return

if __name__ == "__main__":
    main()