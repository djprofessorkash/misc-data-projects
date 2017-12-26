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

    # ============= METHOD TO CALCULATE POTENTIAL ALPHA RANGE VALUES BY A ============
    # ==================== SIMPLE SEQUENTIAL MINIMAL OPTIMIZATION ====================
    def simple_sequential_minimal_optimization(self, input_dataset, class_labels, absolute_ceiling_constant, alpha_tolerance, MAX_ITER):
        dataset = np.mat(input_dataset)                     # Produces formatted dataset
        labels = np.mat(class_labels).transpose()           # Produces transposed class label vector
        b = 0                                               # Initializes value of b to increment later
        NUM_ROWS, NUM_COLS = np.shape(dataset)              # Produces constants of dataset's dimensionality
        
        # Initializes alpha matrix of zeros by number of rows in dataset
        alphas = np.mat(np.zeros((NUM_ROWS, 1)))
        iteration_constant = 0

        # Iterates until predefined iteration constant and iteration ceiling are equal
        while (iteration_constant < MAX_ITER):
            changed_alpha_pairs = 0                         # Initializes dynamic alpha pair values to change later

            # Iterates based on number of rows in dataset to optimize alpha pairs
            for iterator in range(NUM_ROWS):
                # Creates temporary constants for alpha ranges against dataset and labels by the method's parent iterator
                fX_iterator = float(np.multiply(alphas, labels).T * (dataset * dataset[iterator, :].T)) + b
                E_iterator = fX_iterator - float(labels[iterator])

                # Checks if iteration constants abide by absolute and relative boundary conditions defined by the ceiling and tolerance levels
                if ((labels[iterator] * E_iterator < -alpha_tolerance) and (alphas[iterator] < absolute_ceiling_constant)) or ((labels[iterator] * E_iterator > alpha_tolerance) and (alphas[iterator] > 0)):
                    # Creates potential alpha value from randomizer method
                    potential_alpha = self.select_random_potential_alpha(iterator, NUM_ROWS)
                    
                    # Creates temporary constants for alpha ranges against dataset and labels by the method's potential alpha ranges
                    fX_potential = float(np.multiply(alphas, labels).T * (dataset * dataset[potential_alpha, :].T)) + b
                    E_potential = fX_potential - float(labels[potential_alpha])

                    # Creates dummy constants to hold old alpha values from method's parent iterator and potential alpha values
                    old_alpha_iterator = np.copy(alphas[iterator])
                    old_alpha_potential = np.copy(alphas[potential_alpha])

                    # Checks if iterated labels match the expected potential alpha label values
                    if (labels[iterator] != labels[potential_alpha]):
                        # Defines the alpha's ceiling and floor if there is a mismatch
                        alpha_ceiling = min(absolute_ceiling_constant, absolute_ceiling_constant + alphas[potential_alpha] - alphas[iterator])
                        alpha_floor = max(0, alphas[potential_alpha] - alphas[iterator])
                    else:
                        # Defines the alpha's ceiling and floor if there is a match
                        alpha_ceiling = min(absolute_ceiling_constant, alphas[potential_alpha] + alphas[iterator])
                        alpha_floor = max(0, alphas[potential_alpha] + alphas[iterator] - absolute_ceiling_constant)

                    # Checks if floor and ceiling are equivalent and if so, prints for convenience
                    if alpha_ceiling == alpha_floor:
                        print("\nFOR ALPHA'S BOUNDARY CONSTRAINTS, THE CEILING AND FLOOR ARE FOUND TO BE EQUAL.\n")
                        continue

                    # Defines marker value for altering the alpha value for optimization
                    optimal_alpha_change_marker = 2.0 * dataset[iterator, :] * dataset[potential_alpha, :].T - dataset[iterator, :] * dataset[iterator, :].T - dataset[potential_alpha, :] * dataset[potential_alpha, :].T

                    # Checks if optimal alpha marker is zero and if so, prints for convenience
                    if optimal_alpha_change_marker >= 0:
                        print("\nFOR ALPHA'S OPTIMIZATION, THE VALUE OF THE OPTIMAL ALPHA CHANGE MARKER IS EQUAL TO OR GREATER THAN ZERO.\n")
                        continue

                    # Optimizes alpha values based on optimal marker and constraint processing method
                    alphas[potential_alpha] -= labels[potential_alpha] * (E_iterator - E_potential) / optimal_alpha_change_marker
                    alphas[potential_alpha] = self.process_alpha_against_constraints(alphas[potential_alpha], alpha_ceiling, alpha_floor)

                    # Checks if margin between new and old alphas are too small and if so, prints for convenience
                    if (abs(alphas[potential_alpha] - old_alpha_potential) < 0.00001):
                        print("\nTHE POTENTIAL ALPHA VALUE IS NOT MOVING ENOUGH.\n")
                        continue

                    alphas[iterator] += labels[potential_alpha] * labels[iterator] * (old_alpha_potential - alphas[potential_alpha])
                    b1 = b - E_iterator - labels[iterator] * (alphas[iterator] - old_alpha_iterator) * dataset[iterator, :] * dataset[iterator, :].T - labels[potential_alpha] * (alphas[potential_alpha] - old_alpha_potential) * dataset[iterator, :] * dataset[potential_alpha, :].T
                    b2 = b - E_potential - labels[iterator] * (alphas[iterator] - old_alpha_iterator) * dataset[iterator, :] * dataset[potential_alpha, :].T - labels[potential_alpha] * (alphas[potential_alpha] - old_alpha_potential) * dataset[potential_alpha, :] * dataset[potential_alpha, :].T

                    if (0 < alphas[iterator]) and (absolute_ceiling_constant > alphas[iterator]):
                        b = b1
                    elif (0 < alphas[potential_alpha]) and (absolute_ceiling_constant > alphas[potential_alpha]):
                        b = b2
                    else:
                        b = (b1 + b2) / 2.0

                    changed_alpha_pairs += 1
                    print("\nITERATION CONSTANT IS: {}\n\nFUNCTIONAL ITERATOR IS: {}\n\nCHANGED ALPHA PAIRS ARE: \n{}\n".format(iteration_constant, iterator, changed_alpha_pairs))

            if (changed_alpha_pairs == 0):
                iteration_constant += 1
            else:
                iteration_constant = 0
            
            print("\nTOTAL ITERATION NUMBER IS: {}\n".format(iteration_constant))
        
        print("\nB-VALUE IS: {}\n\nALPHAS ARE: \n{}\n".format(b, alphas))
        return b, alphas

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