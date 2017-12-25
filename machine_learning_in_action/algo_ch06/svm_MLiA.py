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


from time import time as t                  # Package for tracking modular and program runtime


# ====================================================================================
# ================================= CLASS DEFINITION =================================
# ====================================================================================


class support_Vector_Machine_Algorithm(object):

    # ======================== CLASS INITIALIZERS/DECLARATIONS =======================
    def __init__(self, TIME_I):
        self.TIME_I = TIME_I                # Initial time measure for runtime tracker

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

    return print("Support vector machine class algorithm is done.\n")

if __name__ == "__main__":
    main()