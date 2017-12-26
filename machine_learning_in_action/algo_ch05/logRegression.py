"""
NAME:               logRegression.py (data_projects/machine_learning_in_action/algo_ch05/)

DESCRIPTION:        Python class application of the logistic regression optimizer algorithm.

                    The logistic regression algorithm is a probabilistic classifier that
                    calculates best-fit parameters against a nonlinear sigmoid function in
                    order to maximize the sample data's fitness and optimize the runtime and
                    memory complexity of the algorithm while reducing relative error. 

                    All source code is available at www.manning.com/MachineLearningInAction. 

USE CASE(S):        Numeric and nominal values

ADVANTAGE(S):       Computationally cheap
                    Relatively easy to implement
                    Easy to contextually understand learned results

DISADVANTAGE(S):    Sensitive and prone to overfitting
                    Potential for low accuracy

NOTE:               Original source code is in Python 2, but my code is in Python 3.

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


class Logistic_Regression_Optimization_Algorithm(object):

    # ======================== CLASS INITIALIZERS/DECLARATIONS =======================
    def __init__(self, TIME_I):
        self.TIME_I = TIME_I                # Initial time measure for runtime tracker

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

        """
        print("GIVEN DATASET IS: \n{}\n".format(dataset))
        print("GIVEN CLASS LABELS ARE: \n{}\n".format(labels))
        """
        return dataset, labels

    # ================ METHOD TO CALCULATE SIGMOID VALUE FROM X-INPUT ================
    def sigmoid_distribution(self, x):
        # Calculates Sigmoid Z-value (functional output) from inputted X-value
        sig = 1.0 / (1 + np.exp(-x))

        """ print("SIGMOID DISTRIBUTION VALUE IS: \n{}\n".format(sig)) """
        return sig

    # ==================== METHOD TO MAXIMIZE REGRESSION WEIGHTS =====================
    # ============= USING GRADIENT ASCENT OPTIMIZATION (BATCH PROCESSING) ============
    def batch_processing_gradient_ascent_optimization(self, input_dataset, class_labels, NUM_ITER = 500):
        dataset = np.mat(input_dataset)                 # Input dataset is array of features (columns) and training samples (rows)
        labels = np.mat(class_labels).transpose()       # Class label vector is linear transposition of input dataset
        NUM_ROWS, NUM_COLS = np.shape(dataset)
        ALPHA = 0.001
        regr_weights = np.ones((NUM_COLS, 1))            # Creates array of regression weights with same size as dataset columns

        """ print("\nTESTING BATCH PROCESSING GRADIENT ASCENT OPTIMIZER FOR {} ITERATIONS...".format(NUM_ITER)) """

        # Iterates over sigmoid distribution to optimize training data regression weights
        for _ in range(NUM_ITER):
            sig = self.sigmoid_distribution(dataset * regr_weights)      # Recursively calls sigmoid method to maximize weights
            error = labels - sig
            regr_weights += ALPHA * dataset.transpose() * error

        """
        # Runs runtime tracker for particular method
        self.track_runtime()
        """

        """
        print("\nTOTAL RELATIVE ERRORS ACROSS SIGMOID DISTRIBUTION IS: \n{}\n".format(error))
        print("\nRELATIVE REGRESSION WEIGHTS FROM OPTIMIZATION ARE: \n{}\n".format(regr_weights))
        """
        return regr_weights

    # ================= SIMPLE METHOD TO MAXIMIZE REGRESSION WEIGHTS =================
    # ================ USING GRADIENT ASCENT OPTIMIZATION (STOCHASTIC) ===============
    def simple_stochastic_gradient_ascent_optimization(self, input_dataset, class_labels):
        NUM_ROWS, NUM_COLS = np.shape(input_dataset)
        ALPHA = 0.01
        regr_weights = np.ones(NUM_COLS)                # Creates array of regression weights with same size as dataset columns

        """ print("\nTESTING SIMPLE STOCHASTIC GRADIENT ASCENT OPTIMIZER FOR ONE (1) ITERATION...") """

        # Iterates over sigmoid distribution to optimize training data regression weights
        for iterator in range(NUM_ROWS):
            sig = self.sigmoid_distribution(sum(input_dataset[iterator] * regr_weights))
            error = class_labels[iterator] - sig
            regr_weights += ALPHA * error * input_dataset[iterator]

        """
        # Runs runtime tracker for particular method
        self.track_runtime()
        """

        """
        print("\nTOTAL RELATIVE ERROR ACROSS SIGMOID DISTRIBUTION IS: \n{}\n".format(error))
        print("\nRELATIVE REGRESSION WEIGHTS FROM OPTIMIZATION ARE: \n{}\n".format(regr_weights))
        """
        return regr_weights

    # ================ ADVANCED METHOD TO MAXIMIZE REGRESSION WEIGHTS ================
    # ================ USING GRADIENT ASCENT OPTIMIZATION (STOCHASTIC) ===============
    def advanced_stochastic_gradient_ascent_optimization(self, input_dataset, class_labels, NUM_ITER = 150):
        NUM_ROWS, NUM_COLS = np.shape(input_dataset)
        regr_weights = np.ones(NUM_COLS)                # Creates array of regression weights with same size as dataset columns

        """ print("\nTESTING ADVANCED STOCHASTIC GRADIENT ASCENT OPTIMIZER FOR {} ITERATIONS...".format(NUM_ITER)) """

        # Iterates over inputted number of iterations to maximize stochastic gradient optimizer
        for iterator_outer in range(NUM_ITER):
            data_index = range(NUM_ROWS)

            # Iterates over sigmoid distribution to optimize training data regression weights
            for iterator_inner in range(NUM_ROWS):
                ALPHA = 4 / (1.0 + iterator_outer + iterator_inner) + 0.01
                random_index = int(np.random.uniform(0, len(data_index)))
                sig = self.sigmoid_distribution(sum(input_dataset[random_index] * regr_weights))
                error = class_labels[random_index] - sig
                regr_weights += ALPHA * error * input_dataset[random_index]
                del(list(input_dataset)[random_index])

        """
        # Runs runtime tracker for particular method
        self.track_runtime()
        """

        """
        print("\nTOTAL RELATIVE ERRORS ACROSS SIGMOID DISTRIBUTION IS: \n{}\n".format(error))
        print("\nRELATIVE REGRESSION WEIGHTS FROM OPTIMIZATION ARE: \n{}\n".format(regr_weights))
        """
        return regr_weights

    # ====== METHOD TO PLOT LOGISTIC REGRESSION LINE OF BEST FIT ACROSS DATASET ======
    def plot_line_of_best_fit(self, dataset, labels, weights):
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

        # Plots X- and Y-dimensions of the line of best fit
        ax.plot(x, y)

        # Plots axial labels to finalize scatterplot
        plt.xlabel("X1")
        plt.ylabel("Y1")

        """
        # Runs runtime tracker for particular method
        self.track_runtime()
        """

        # Displays line of best fit over scatterplot of dataset
        print("DISPLAYING LOGISTIC REGRESSION BEST-FIT LINE...\n")
        plt.show()
        return

    # ========= METHOD TO CLASSIFY SAMPLE VECTOR AGAINST SIGMOID DISTRIBUTION ========
    def classify_vector_with_sigmoid(self, input_data_array, regr_weights):
        sig_prob = self.sigmoid_distribution(sum(input_data_array * regr_weights))

        # Return class of 1 if sigmoid is greater than 0.5; otherwise, return class of 0
        if sig_prob > 0.5:
            return 1.0
        return 0.0

    # ======= METHOD TO APPLY SIGMOID CLASSIFIER AND GRADIENT ASCENT OPTIMIZER =======
    # ================== AGAINST SAMPLE HORSE COLIC DISEASE DATASETS =================
    def test_classifier_against_horse_data(self, current_test_iteration):
        HORSE_TRAINING_DATA = open("./horse_colic_training.txt")
        HORSE_TEST_DATA = open("./horse_colic_test.txt")
        training_set = []
        training_labels = []

        # Iterate through horse training data and produce training set and training class label vector
        for line in HORSE_TRAINING_DATA.readlines():
            current_line = line.strip().split("\t")
            array_of_lines = []

            # Produces array of lines from iterating through current line of training data
            for iterator in range(21):
                array_of_lines.append(float(current_line[iterator]))
            
            # Produces training set from line array and training label vector from 21st data entry of each current line of training data
            training_set.append(array_of_lines)
            training_labels.append(float(current_line[21]))

        # Create training regression weights using the advanced stochastic gradient ascent optimizer against the training set and training labels for 500 iterations
        training_weights = self.advanced_stochastic_gradient_ascent_optimization(np.array(training_set), training_labels)
        error_count = 0.0
        number_of_test_vectors = 0.0

        # Iterate through horse test data and produce test set and relative multi-algorithmic error count
        for line in HORSE_TEST_DATA.readlines():
            number_of_test_vectors += 1.0
            current_line = line.strip().split("\t")
            array_of_lines = []

            # Produces array of lines from iterating through current line of test data
            for iterator in range(21):
                array_of_lines.append(float(current_line[iterator]))

            # Increment error count if expected test class label does not match actual test class label
            if int(self.classify_vector_with_sigmoid(np.array(array_of_lines), training_weights)) != int(current_line[21]):
                error_count += 1

        # Calculate error rate across entire horse test data classification
        error_rate = error_count / number_of_test_vectors

        """
        # Runs runtime tracker for particular method
        self.track_runtime()
        """

        # Garbage styling simply for eye candy purposes
        if current_test_iteration == 0:
            print("\n\n")

        print("THE ERROR RATE OF THE HORSE COLIC DATA CLASSIFIER FOR TEST #{} IS: {}".format(current_test_iteration + 1, error_rate))
        return error_rate

    # ========== METHOD TO RUN k ITERATIONS OF THE HORSE DATASET CLASSIFIER ==========
    def k_series_of_test_classifications(self, k_num_series):
        error_sum = 0.0

        # Iterates k times, each time running a new instance of the test classifier and incrementing the error sum
        for k in range(k_num_series):
            error_sum += self.test_classifier_against_horse_data(k)

        # Calculates and prints average error rate of classifier across all test instances
        average_error_rate = error_sum / float(k_num_series)
        print("\n\nAFTER k={} ITERATIONS, THE AVERAGE ERROR RATE OF THE CLASSIFIER IS: {}\n".format(k_num_series, average_error_rate))
        
        # Runs runtime tracker for particular method
        self.track_runtime()
        return

    # ================ METHOD TO BENCHMARK RUNTIME OF SPECIFIC METHOD ================
    def track_runtime(self):
        # Track ending time of program and determine overall program runtime
        TIME_F = t()
        delta = TIME_F - self.TIME_I

        if delta < 1.5:
            print("\nReal program runtime is {0:.4g} milliseconds.\n".format(delta * 1000))
        else:
            print("\nReal program runtime is {0:.4g} seconds.\n".format(delta))
        return


# ====================================================================================
# ================================ MAIN RUN FUNCTION =================================
# ====================================================================================


def main():
    # Track starting time of program
    TIME_I = t()

    # Initialize class instance of the logistic regression optimization algorithm
    logRegres = Logistic_Regression_Optimization_Algorithm(TIME_I)

    # Test batch_processing_gradient_ascent_optimization() with sigmoid function calculation on sample data
    """
    dataset, labels = logRegres.load_dataset()
    logRegres.batch_processing_gradient_ascent_optimization(dataset, labels)
    """

    # Test plot_line_of_best_fit() with batch processing gradient ascent optimization on sample data
    """
    dataset, labels = logRegres.load_dataset()
    weights = logRegres.batch_processing_gradient_ascent_optimization(dataset, labels, 1000)
    logRegres.plot_line_of_best_fit(dataset, labels, weights)
    """

    # Test plot_line_of_best_fit() with simple stochastic gradient ascent optimization on sample data
    """
    dataset, labels = logRegres.load_dataset()
    weights = logRegres.simple_stochastic_gradient_ascent_optimization(np.array(dataset), labels)
    logRegres.plot_line_of_best_fit(dataset, labels, weights)
    """

    # Test plot_line_of_best_fit() with advanced stochastic gradient ascent optimization on sample data
    """
    dataset, labels = logRegres.load_dataset()
    weights = logRegres.advanced_stochastic_gradient_ascent_optimization(np.array(dataset), labels, 1000)
    logRegres.plot_line_of_best_fit(dataset, labels, weights)
    """

    # Test k_series_of_test_classifications() with modular classifier methods on horse datasets
    logRegres.k_series_of_test_classifications(10)

    return print("Logistic regression class algorithm is done.\n")

if __name__ == "__main__":
    main()