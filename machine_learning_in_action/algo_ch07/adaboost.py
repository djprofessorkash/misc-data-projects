"""
NAME:               adaboost.py (data_projects/machine_learning_in_action/algo_ch07/)

DESCRIPTION:        Python class application of the adaptive booster meta-algorithm.

                    ???

                    All source code is available at www.manning.com/MachineLearningInAction. 

USE CASE(S):        Numeric and nominal values

ADVANTAGE(S):       Relatively easy to code
                    Relatively low generalization error
                    No necessarily adjustable parameters
                    Works with most other classifier algorithms

DISADVANTAGE(S):    Sensitive to outlier values

NOTE:               Original source code is in Python 2, but my code is in Python 3.

CREDIT:             Machine Learning in Action (Peter Harrington)
"""


# ====================================================================================
# ================================ IMPORT STATEMENTS =================================
# ====================================================================================


from time import time as t                  # Package for tracking modular and program runtime
import numpy as np                          # Library for simple linear mathematical operations


# ====================================================================================
# ================================ CLASS DEFINITION ==================================
# ====================================================================================


class AdaBoost_Adaptive_Booster_Meta_Algorithm(object):

    # ======================== CLASS INITIALIZERS/DECLARATIONS =======================
    def __init__(self, TIME_I):
        self.TIME_I = TIME_I                            # Initial time measure for runtime tracker

    # ======================== METHOD TO LOAD IN SAMPLE DATASET ======================
    def load_sample_data(self):
        dataset = np.matrix([[1.0, 2.1],
                             [2.0, 1.1],
                             [1.3, 1.0],
                             [1.0, 1.0],
                             [2.0, 1.0]])
        labels = [1.0, 1.0, -1.0, -1.0, 1.0]

        """ print("\nSAMPLE DATASET IS: \n{}\n\nSAMPLE CLASS LABEL VECTOR IS: \n{}\n".format(dataset, labels)) """
        return dataset, labels

    # ====================== METHOD TO ADAPTIVELY LOAD IN DATASET ====================
    def adaptive_load_data(self, FILENAME):
        num_features = len(open(FILENAME).readline().split("\t"))
        
        # Defines dataset and class label vector as empty arrays
        dataset = []
        labels = []

        # Iterates through each line in loaded file
        for line in open(FILENAME).readlines():
            lines = []
            current_line = line.strip().split("\t")

            # Iterates through all features
            for iterator in range(num_features - 1):
                # Creates array of lines with each value from dataset
                lines.append(float(current_line[iterator]))
            
            # Creates dataset and class label vector from parsed data
            dataset.append(lines)
            labels.append(float(current_line[-1]))
        
        print("\nFIRST 20 ENTRIES IN DATASET ARE: \n{}\n\nCLASS LABEL VECTOR IS: \n{}\n".format(dataset[:20], labels))
        return dataset, labels

    # ================= METHOD TO CLASSIFY ELEMENT FROM DECISION STUMP ===============
    def classify_decision_stump(self, input_dataset, dimension, threshold_value, threshold_inequality):
        classification_array = np.ones((np.shape(input_dataset)[0], 1))

        if threshold_inequality == "lt":
            classification_array[input_dataset[:, dimension] <= threshold_value] = -1.0
        else:
            classification_array[input_dataset[:, dimension] > threshold_value] = -1.0
        
        """ print("\nCLASSIFICATION ARRAY FOR THE SAMPLE DECISION STUMP IS: \n{}\n".format(classification_array)) """
        return classification_array

    # ============ METHOD TO CONSTRUCT DECISION STUMP WITH WEIGHTED ERRORS ===========
    def construct_decision_stump(self, input_dataset, class_label_vector, data_weight_vector):
        dataset = np.mat(input_dataset)                         # Creates dataset
        labels = np.mat(class_label_vector).T                   # Creates class label vector

        NUM_ROWS, NUM_COLS = np.shape(dataset)                  # Creates dimensionality constants of dataset

        number_of_steps = 10.0                                  # Creates number of steps (constant?)
        best_stump = {}                                         # Creates dictionary to hold best stump
        best_class_estimate = np.mat(np.zeros((NUM_ROWS)))      # Creates best class estimate matrix
        minimum_error = np.inf                                  # Creates minimum error value

        # Iterates through column dimensionality of dataset
        for outer_iterator in range(NUM_COLS):
            minimum_range = np.min(dataset[:, outer_iterator])              # Creates minimum range of dataset
            maximum_range = np.max(dataset[:, outer_iterator])              # Creates maximum range of dataset

            step_size = (maximum_range - minimum_range) / number_of_steps   # Creates step size

            # Iterates through number of steps for decision stump and dual search for stump inequality
            for inner_iterator in range(-1, int(number_of_steps) + 1):
                for inequality in ["lt", "gt"]:
                    # Creates threshold value and predicted values from decision stump method
                    threshold_value = minimum_range + float(inner_iterator) * step_size
                    predicted_values = self.classify_decision_stump(dataset, outer_iterator, threshold_value, inequality)

                    # Creates array to hold error values
                    error_transform_array = np.mat(np.ones((NUM_ROWS, 1)))
                    error_transform_array[predicted_values == labels] = 0

                    # Produces array for weighted boosting errors
                    weighted_error = data_weight_vector.T * error_transform_array
                    """ print("\nDIMENSIONAL SPLIT PARAMETER IS: {}\nDECISION STUMP THRESHOLD VALUE IS: {}\nDECISION STUMP THRESHOLD INEQUALITY IS: {}\nTHE WEIGHTED ERROR FOR THE SAMPLE DATASET IS: {}\n".format(outer_iterator, threshold_value, inequality, weighted_error)) """

                    # Selects minimum error and best class label estimate from weighted errors
                    if weighted_error < minimum_error:
                        minimum_error = weighted_error
                        best_class_estimate = np.copy(predicted_values)

                        # Defines some attributes of best stump dictionary
                        best_stump["dimension"] = outer_iterator
                        best_stump["threshold"] = threshold_value
                        best_stump["inequality"] = inequality

        """ print("\nRELATIVELY BEST DECISION STUMP IS: \n{}\n\nMINIMUM ERROR IS: {}\n\nBEST CLASS ESTIMATE FOR STUMP IS: \n{}\n".format(best_stump, minimum_error, best_class_estimate)) """
        return best_stump, minimum_error, best_class_estimate

    # ========= METHOD TO TRAIN SAMPLE DATA FROM DECISION STUMP WITH BOOSTER =========
    def adaboost_training_with_decision_stump(self, input_dataset, class_label_vector, NUM_ITER = 40):
        weak_class_vector = []
        DATASET_SIZE = np.shape(input_dataset)[0]

        # Creates weight vector and continuously constructing label estimates for dataset
        data_weight_vector = np.mat(np.ones((DATASET_SIZE, 1)))
        aggregate_class_estimate = np.mat(np.zeros((DATASET_SIZE, 1)))
        """ print("\nDATA WEIGHT VECTOR IS: \n{}\n\nAGGREGATE CLASS ESTIMATE IS: \n{}\n".format(data_weight_vector.T, aggregate_class_estimate.T)) """

        # Iterates through max iteration number
        for iterator in range(NUM_ITER):
            # Defines stump structure, relative error, and holding best estimate label
            best_stump, weighted_sum_error, best_class_estimate = self.construct_decision_stump(input_dataset, class_label_vector, data_weight_vector)

            # Defines epsilon as smallest float value
            epsilon = np.finfo(float).eps

            # Creates alpha value and sets in best stump structure
            alpha = float(0.5 * np.log((1.0 - weighted_sum_error + epsilon) / (weighted_sum_error + epsilon)))            
            """ print("\nRELATIVE WEIGHTED SUM OF ERRORS: \n{}\n\nALPHA: \n{}\n\nITERATIVE CLASS ESTIMATE IS: \n{}\n".format(weighted_sum_error, alpha, best_class_estimate.T)) """

            best_stump["alpha"] = alpha
            weak_class_vector.append(best_stump)
            
            # Creates weighing factors based on class label estimate accuracy and apply to data weighing vectors
            weighing_exponential_factor = np.multiply(-1 * alpha * np.mat(class_label_vector).T, best_class_estimate)
            data_weight_vector = np.multiply(data_weight_vector, np.exp(weighing_exponential_factor))
            data_weight_vector = data_weight_vector / np.sum(data_weight_vector)

            # Creates aggregate class label estimate from previous best estimate label
            aggregate_class_estimate += alpha * best_class_estimate
            """ print("\nAGGREGATING CLASS ESTIMATE IS: \n{}\n".format(aggregate_class_estimate.T)) """

            # Creates aggregate relative errors for class labeling and condenses into error rate across method
            aggregate_errors = np.multiply(np.sign(aggregate_class_estimate) != np.mat(class_label_vector).T, np.ones((DATASET_SIZE, 1)))
            error_rate = np.sum(aggregate_errors) / DATASET_SIZE
            """ print("\nTOTAL ERROR RATE ACROSS TRAINER IS: \n{}\n".format(error_rate)) """
            
            # Breaks loop if error rate limit tends to zero
            if error_rate == 0.0:
                break

        print("\nWEAK CLASS VECTOR IS: \n{}\n".format(weak_class_vector))
        return weak_class_vector

    # ========= METHOD TO TEST SAMPLE DATA FROM DECISION STUMP WITH BOOSTER ==========
    def adaboost_testing_with_decision_stump(self, input_dataset, weak_classifiers):
        dataset = np.mat(input_dataset)
        NUM_ROWS = np.shape(input_dataset)[0]

        # Creates continuously constructing label estimates for dataset
        aggregate_class_estimate = np.mat(np.zeros((NUM_ROWS, 1)))

        # Iterates through range of weak classifier array size
        for iterator in range(len(weak_classifiers)):
            # Repeatedly classifies decision stump, producing finer class estimates with each iteration
            iterative_class_estimate = self.classify_decision_stump(dataset, weak_classifiers[iterator]["dimension"], weak_classifiers[iterator]["threshold"], weak_classifiers[iterator]["inequality"])
            aggregate_class_estimate += weak_classifiers[iterator]["alpha"] * iterative_class_estimate
            """ print("\nAGGREGATING CLASS ESTIMATE IS: \n{}\n".format(aggregate_class_estimate)) """

        # Returns sign (+/-) of completely aggregated class label estimates for test dataset
        print("\nPREDICTED LABEL VECTOR (SIGNS) OF FINAL AGGREGATE CLASS ESTIMATE IS: \n{}\n".format(np.sign(aggregate_class_estimate)))
        return np.sign(aggregate_class_estimate), self.track_runtime()

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

    # Initialize class instance of the AdaBoost ensemble method algorithm
    ada = AdaBoost_Adaptive_Booster_Meta_Algorithm(TIME_I)

    """
    # Run a sample dataset and vector of labels in the class instance
    dataset, labels = ada.load_sample_data()
    """

    """
    # Run the decision-stump constructor and classifier
    dataset, labels = ada.load_sample_data()
    data_weight_vector = np.mat(np.ones((5, 1)) / 5)
    ada.construct_decision_stump(dataset, labels, data_weight_vector)
    """

    """
    # Run the decision-stump-based trainer for 9 iterations
    dataset, labels = ada.load_sample_data()
    ada.adaboost_training_with_decision_stump(dataset, labels, 9)
    """

    # Run the decision-stump-based tester with 30 training iterations and user-inputted test data
    dataset, labels = ada.load_sample_data()
    weak_classifiers = ada.adaboost_training_with_decision_stump(dataset, labels, 30)
    ada.adaboost_testing_with_decision_stump([[5, 5], [0, 0], [1.5, 1.2], [-1, -1]], weak_classifiers)

    return print("\nAdaBoost class meta-algorithm is done.\n")

if __name__ == "__main__":
    main()