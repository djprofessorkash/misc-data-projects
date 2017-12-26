"""
NAME:               svm_MLiA.py (data_projects/machine_learning_in_action/algo_ch06/)

DESCRIPTION:        Python class application of the support vector machine algorithm.

                    ??? 

                    All source code is available at www.manning.com/MachineLearningInAction. 

USE CASE(S):        Numeric and nominal values

ADVANTAGE(S):       Computationally cheap
                    Relatively low generalization error
                    Easy to contextually understand learned results

DISADVANTAGE(S):    Natively only handles binary classification
                    Sensitive to tuning parameters and choice of kernel

NOTE:               Original source code is in Python 2, but my code is in Python 3.

CREDIT:             Machine Learning in Action (Peter Harrington)
"""


# ====================================================================================
# ================================ IMPORT STATEMENTS =================================
# ====================================================================================


import numpy as np                          # Library for simple linear mathematical operations
from time import time as t                  # Package for tracking modular and program runtime


# ====================================================================================
# ================================= CLASS DEFINITION =================================
# ====================================================================================


class support_Vector_Machine_Algorithm(object):

    # ======================== CLASS INITIALIZERS/DECLARATIONS =======================
    def __init__(self, TIME_I):
        self.TIME_I = TIME_I                    # Initial time measure for runtime tracker
        self.FILE = open("test_set.txt")        # Open filename as read-only for test dataset

    # ======================== METHOD TO LOAD IN SAMPLE DATASET ======================
    def load_dataset(self):
        dataset = []
        labels = []

        # Iterates through sample file to produce sample dataset and class label vector
        for line in self.FILE.readlines():
            array_of_lines = line.strip().split("\t")
            dataset.append([float(array_of_lines[0]), float(array_of_lines[1])])
            labels.append(float(array_of_lines[2]))

        """ print("\nSAMPLE DATASET IS: \n{}\n\nSAMPLE LABELS ARE: \n{}\n".format(dataset, labels)) """
        return dataset, labels

    # ========== METHOD TO SELECT POTENTIAL ALPHA FROM UNIFORM DISTRIBUTION ==========
    def select_random_potential_alpha(self, alpha_index, alpha_total):
        potential_alpha = alpha_index

        while (potential_alpha == alpha_index):
            potential_alpha = int(np.random.uniform(0, alpha_total))
        
        print("\nJ-VALUE IS: {}\n".format(potential_alpha))
        return potential_alpha

    # ========= METHOD TO POTENTIAL ALPHA VALUE AGAINST BOUNDARY CONSTRAINTS =========
    def process_alpha_against_constraints(alpha_from_potential, alpha_ceiling, alpha_floor):
        # Processes alpha value against ceiling constraint (cannot be greater than)
        if alpha_from_potential > alpha_ceiling:
            alpha_from_potential = alpha_ceiling

        # Processes alpha value against floor constraint (cannot be less than)
        if alpha_floor > alpha_from_potential:
            alpha_from_potential = alpha_floor

        print("\nALPHA VALUE PROCESSED AGAINST CONSTRAINTS IS: {}\n".format(alpha_from_potential))
        return alpha_from_potential

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

    # Initialize class instance of the support vector machine algorithm
    svm = support_Vector_Machine_Algorithm(TIME_I)

    # Test load_dataset() method on SVM
    dataset, labels = svm.load_dataset()

    return print("\nSupport vector machine class algorithm is done.\n")

if __name__ == "__main__":
    main()