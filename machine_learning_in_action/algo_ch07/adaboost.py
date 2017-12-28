"""
NAME:               adaboost.py (data_projects/machine_learning_in_action/algo_ch07/)

DESCRIPTION:        Python class application of ???.

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
    def load_dataset(self):
        dataset = np.matrix([[1.0, 2.1],
                             [2.0, 1.1],
                             [1.3, 1.0],
                             [1.0, 1.0],
                             [2.0, 1.0]])
        labels = [1.0, 1.0, -1.0, -1.0, 1.0]

        """ print("\nSAMPLE DATASET IS: \n{}\n\nSAMPLE CLASS LABEL VECTOR IS: \n{}\n".format(dataset, labels)) """
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
        dataset = np.mat(input_dataset)
        labels = np.mat(class_label_vector).T

        NUM_ROWS, NUM_COLS = np.shape(dataset)

        number_of_steps = 10.0
        best_stump = {}
        best_class_estimate = np.mat(np.zeros((NUM_ROWS)))
        minimum_error = np.inf

        for outer_iterator in range(NUM_COLS):
            minimum_range = np.min(dataset[:, outer_iterator])
            maximum_range = np.max(dataset[:, outer_iterator])

            step_size = (maximum_range - minimum_range) / number_of_steps

            for inner_iterator in range(-1, int(number_of_steps) + 1):
                for inequality in ["lt", "gt"]:
                    threshold_value = minimum_range + float(inner_iterator) * step_size
                    predicted_values = self.classify_decision_stump(dataset, outer_iterator, threshold_value, inequality)

                    error_array = np.mat(np.ones((NUM_ROWS, 1)))
                    error_array[predicted_values == labels] = 0

                    weighted_error = data_weight_vector.T * error_array
                    """ print("\nDIMENSIONAL SPLIT PARAMETER IS: {}\nDECISION STUMP THRESHOLD VALUE IS: {}\nDECISION STUMP THRESHOLD INEQUALITY IS: {}\nTHE WEIGHTED ERROR FOR THE SAMPLE DATASET IS: {}\n".format(outer_iterator, threshold_value, inequality, weighted_error)) """

                    if weighted_error < minimum_error:
                        minimum_error = weighted_error
                        best_class_estimate = np.copy(predicted_values)

                        best_stump["dimension"] = outer_iterator
                        best_stump["threshold"] = threshold_value
                        best_stump["inequality"] = inequality

        print("\nRELATIVELY BEST DECISION STUMP IS: \n{}\n\nMINIMUM ERROR IS: {}\n\nBEST CLASS ESTIMATE FOR STUMP IS: {}\n".format(best_stump, minimum_error, best_class_estimate))

        # Performs runtime tracker for particular method
        self.track_runtime()

        return best_stump, minimum_error, best_class_estimate

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

    # Initialize class instance of the AdaBoost ensemble method algorithm
    ada = AdaBoost_Adaptive_Booster_Meta_Algorithm(TIME_I)

    """
    # Test loading a sample dataset in the class instance
    dataset, labels = ada.load_dataset()
    """

    # Test the decision-stump constructor and classifier
    dataset, labels = ada.load_dataset()
    data_weight_vector = np.mat(np.ones((5, 1)) / 5)
    ada.construct_decision_stump(dataset, labels, data_weight_vector)

    return print("\nAdaBoost class meta-algorithm is done.\n")

if __name__ == "__main__":
    main()
