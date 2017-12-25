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

    # ===== METHOD TO OPTIMIZE REGRESSION WEIGHTS USING GRADIENT ASCENT (BATCH) ======
    def batch_processing_gradient_ascent_optimization(self, input_dataset, class_labels):
        dataset = np.mat(input_dataset)                 # Input dataset is array of features (columns) and training samples (rows)
        labels = np.mat(class_labels).transpose()       # Class label vector is linear transposition of input dataset
        NUM_ROWS, NUM_COLS = np.shape(dataset)
        ALPHA = 0.001
        MAX_REPETITIONS = 500
        regr_weights = np.ones((NUM_COLS, 1))            # Creates array of regression weights with same size as dataset columns

        # Iterates over sigmoid distribution to optimize training data regression weights
        for _ in range(MAX_REPETITIONS):
            sig = self.sigmoid_distribution(dataset * regr_weights)      # Recursively calls sigmoid function to maximize weights
            error = labels - sig
            regr_weights += ALPHA * dataset.transpose() * error

        # # Runs runtime tracker for particular method
        # self.track_runtime(TIME_I)

        # print("\nTOTAL RELATIVE ERRORS ACROSS SIGMOID DISTRIBUTION IS: \n{}\n".format(error))
        print("\nRELATIVE REGRESSION WEIGHTS FROM OPTIMIZATION ARE: \n{}\n".format(regr_weights))
        return regr_weights

    # === METHOD TO OPTIMIZE REGRESSION WEIGHTS USING GRADIENT ASCENT (STOCHASTIC) ===
    def stochastic_gradient_ascent_optimization(self, input_dataset, class_labels):
        NUM_ROWS, NUM_COLS = np.shape(input_dataset)
        ALPHA = 0.01
        regr_weights = np.ones(NUM_COLS)                # Creates array of regression weights with same size as dataset columns

        # Iterates over sigmoid distribution to optimize training data regression weights
        for iterator in range(NUM_ROWS):
            sig = self.sigmoid_distribution(sum(input_dataset[iterator] * regr_weights))
            error = class_labels[iterator] - sig
            regr_weights += ALPHA * error * input_dataset[iterator]

        # # Runs runtime tracker for particular method
        # self.track_runtime(TIME_I)

        # print("\nTOTAL RELATIVE ERRORS ACROSS SIGMOID DISTRIBUTION IS: \n{}\n".format(error))
        print("\nRELATIVE REGRESSION WEIGHTS FROM OPTIMIZATION ARE: \n{}\n".format(regr_weights))
        return regr_weights

    # ====== METHOD TO PLOT LOGISTIC REGRESSION LINE OF BEST FIT ACROSS DATASET ======
    def plot_line_of_best_fit(self, dataset, labels, weights, TIME_I):
        regr_weights = np.array(weights)
        data_arr = np.array(dataset)
        NUM_ROWS = np.shape(data_arr)[0]

        # Predefines scatter X- and Y-coordinates
        x_coord1 = []
        y_coord1 = []
        x_coord2 = []
        y_coord2 = []

        # Creates X- and Y-coordinates based on class label vector's array of ones
        for iterator in range(NUM_ROWS):
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
        y = (-regr_weights[0] - regr_weights[1] * x) / regr_weights[2]

        # Plots X and Y domains with labels on axes to finalize scatterplot
        ax.plot(x, y)
        plt.xlabel("X1")
        plt.ylabel("Y1")

        # Runs runtime tracker for particular method
        self.track_runtime(TIME_I)

        # Displays line of best fit over scatterplot of dataset
        print("DISPLAYING LOGISTIC REGRESSION BEST-FIT LINE AFTER 500 CYCLES OF GRADIENT ASCENT...\n")
        plt.show()
        return

    # ================ METHOD TO BENCHMARK RUNTIME OF SPECIFIC METHOD ================
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

    # Test batch_processing_gradient_ascent_optimization() with sigmoid calculation and loading test data
    """
    dataset, labels = logRegres.load_dataset()
    logRegres.batch_processing_gradient_ascent_optimization(dataset, labels)
    logRegres.track_runtime(TIME_I)
    """

    # Test plot_line_of_best_fit() with loading test data
    dataset, labels = logRegres.load_dataset()
    weights = logRegres.batch_processing_gradient_ascent_optimization(dataset, labels)
    logRegres.plot_line_of_best_fit(dataset, labels, weights, TIME_I)

    return

if __name__ == "__main__":
    main()