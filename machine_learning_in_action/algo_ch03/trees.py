"""
NAME:               trees.py (data_projects/machine_learning_in_action/algo_ch03/)

DESCRIPTION:        Python class application of decision tree ML algorithms. 

                    ???

                    All source code is available at www.manning.com/MachineLearningInAction. 

USE CASE(S):        Numeric and nominal values

ADVANTAGE(S):       Computationally cheap
                    Easy to contextually understand learned results
                    Missing values do not significantly affect results
                    Innately deals with irrelevant features

DISADVANTAGE(S):    Sensitive and prone to overfitting

NOTE:               Original source code is Python 2, but my code is Python 3.

CREDIT:             Machine Learning In Action (Peter Harrington)
"""


# ====================================================================================
# ================================ IMPORT STATEMENTS =================================
# ====================================================================================


from math import log                        # Package for performing logarithmic operations
from time import time as t                  # Package for tracking modular and program runtime


# ====================================================================================
# ================================= CLASS DEFINITION =================================
# ====================================================================================


class k_Decision_Tree_Algorithm(object):
    pass
 
 
# ====================================================================================
# ================================ MAIN RUN FUNCTION =================================
# ====================================================================================


def main():
    # Track starting time of running program
    t0 = t()

    # Initialize class instance of the decision tree algorithm
    kD = k_Decision_Tree_Algorithm()
    return

if __name__ == "__main__":
    main()