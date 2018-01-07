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
import numpy as np                          # Library for simple linear mathematical operations


# ====================================================================================
# ================================ CLASS DEFINITION ==================================
# ====================================================================================

class linear_Regression(object):

    # ======================== CLASS INITIALIZERS/DECLARATIONS =======================
    def __init__(self, TIME_I):
        self.TIME_I = TIME_I                            # Initial time measure for runtime tracker

    def load_sample_data(self, FILENAME):
        num_of_features = len(open(FILENAME).readline().split("\t")) - 1
        
        dataset = []
        labels = []

        f = open(FILENAME)

        for line in f.readlines():
            line_arr = []
            current_line = line.strip().split("\t")

            for iterator in range(num_of_features):
                line_arr.append(float(current_line[iterator]))
            
            dataset.append(line_arr)
            labels.append(float(current_line[-1]))

        print("\nSAMPLE DATASET IS: \n{}\n\nCLASS LABEL VECTOR FOR SAMPLE DATA IS: \n{}\n".format(dataset, labels))
        return dataset, labels

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

    return print("\nAdaBoost class meta-algorithm is done.\n")

if __name__ == "__main__":
    main()