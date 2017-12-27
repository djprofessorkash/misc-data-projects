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
# ===================== CLASS DEFINITION: SUPPORT VECTOR MACHINE =====================
# ====================================================================================


class Support_Vector_Machine_Algorithm(object):

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
        
        """ print("\nPOTENTIAL ALPHA VALUE IS: {}\n".format(potential_alpha)) """
        return potential_alpha

    # ===== METHOD TO PROCESS POTENTIAL ALPHA VALUE AGAINST BOUNDARY CONSTRAINTS =====
    def process_alpha_against_constraints(self, alpha_from_potential, alpha_ceiling, alpha_floor):
        # Processes alpha value against ceiling constraint (cannot be greater than)
        if alpha_from_potential > alpha_ceiling:
            alpha_from_potential = alpha_ceiling

        # Processes alpha value against floor constraint (cannot be less than)
        if alpha_floor > alpha_from_potential:
            alpha_from_potential = alpha_floor

        """ print("\nALPHA VALUE PROCESSED AGAINST CONSTRAINTS IS: {}\n".format(alpha_from_potential)) """
        return alpha_from_potential

    # ============= METHOD TO CALCULATE POTENTIAL ALPHA RANGE VALUES BY A ============
    # ============== SIMPLE PLATT SEQUENTIAL MINIMAL OPTIMIZATION (SMO) ==============
    def simple_sequential_minimal_optimization(self, input_dataset, class_labels, absolute_ceiling_constant, alpha_tolerance, MAX_ITER):
        dataset = np.mat(input_dataset)                     # Produces formatted dataset
        labels = np.mat(class_labels).transpose()           # Produces transposed class label vector
        NUM_ROWS, NUM_COLS = np.shape(dataset)              # Produces constants of dataset's dimensionality
        beta = 0                                            # Initializes value of beta to increment later

        # Initializes alpha matrix of zeros by number of rows in dataset
        alphas = np.mat(np.zeros((NUM_ROWS, 1)))
        iteration_constant = 0

        # Iterates until predefined iteration constant and iteration ceiling are equal
        while (iteration_constant < MAX_ITER):
            changed_alpha_pairs = 0                         # Initializes dynamic alpha pair values to change later

            # Iterates based on number of rows in dataset to optimize alpha pairs
            for iterator in range(NUM_ROWS):
                # Creates temporary constants for alpha ranges against dataset and labels by the method's parent iterator
                fX_iterator = float(np.multiply(alphas, labels).T * (dataset * dataset[iterator, :].T)) + beta
                E_iterator = fX_iterator - float(labels[iterator])

                # Checks if iteration constants abide by absolute and relative boundary conditions defined by the ceiling and tolerance levels
                if ((labels[iterator] * E_iterator < -alpha_tolerance) and (alphas[iterator] < absolute_ceiling_constant)) or ((labels[iterator] * E_iterator > alpha_tolerance) and (alphas[iterator] > 0)):
                    # Creates potential alpha value from randomizer method
                    potential_alpha = self.select_random_potential_alpha(iterator, NUM_ROWS)
                    
                    # Creates temporary constants for alpha ranges against dataset and labels by the method's potential alpha ranges
                    fX_potential = float(np.multiply(alphas, labels).T * (dataset * dataset[potential_alpha, :].T)) + beta
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
                        """ print("\nFOR ALPHA'S BOUNDARY CONSTRAINTS, THE CEILING AND FLOOR ARE FOUND TO BE EQUAL.\n") """
                        continue

                    # Defines marker value for altering the alpha value for optimization
                    optimal_alpha_change_marker = 2.0 * dataset[iterator, :] * dataset[potential_alpha, :].T - dataset[iterator, :] * dataset[iterator, :].T - dataset[potential_alpha, :] * dataset[potential_alpha, :].T

                    # Checks if optimal alpha marker is zero and if so, prints for convenience
                    if optimal_alpha_change_marker >= 0:
                        """ print("\nFOR ALPHA'S OPTIMIZATION, THE VALUE OF THE OPTIMAL ALPHA CHANGE MARKER IS EQUAL TO OR GREATER THAN ZERO.\n") """
                        continue

                    # Optimizes alpha values based on optimal marker and constraint processing method
                    alphas[potential_alpha] -= labels[potential_alpha] * (E_iterator - E_potential) / optimal_alpha_change_marker
                    alphas[potential_alpha] = self.process_alpha_against_constraints(alphas[potential_alpha], alpha_ceiling, alpha_floor)

                    # Checks if margin between new and old alphas are too small and if so, prints for convenience
                    if (abs(alphas[potential_alpha] - old_alpha_potential) < 0.00001):
                        """ print("\nTHE POTENTIAL ALPHA VALUE IS NOT MOVING ENOUGH.\n") """
                        continue

                    # Increments new alpha values and produces temporary beta-values to track differential alpha changes
                    alphas[iterator] += labels[potential_alpha] * labels[iterator] * (old_alpha_potential - alphas[potential_alpha])
                    beta1 = beta - E_iterator - labels[iterator] * (alphas[iterator] - old_alpha_iterator) * dataset[iterator, :] * dataset[iterator, :].T - labels[potential_alpha] * (alphas[potential_alpha] - old_alpha_potential) * dataset[iterator, :] * dataset[potential_alpha, :].T
                    beta2 = beta - E_potential - labels[iterator] * (alphas[iterator] - old_alpha_iterator) * dataset[iterator, :] * dataset[potential_alpha, :].T - labels[potential_alpha] * (alphas[potential_alpha] - old_alpha_potential) * dataset[potential_alpha, :] * dataset[potential_alpha, :].T

                    # Checks if new alpha values fall within beta-dependent boundary conditions for beta-value reinitialization
                    if (0 < alphas[iterator]) and (absolute_ceiling_constant > alphas[iterator]):
                        beta = beta1
                    elif (0 < alphas[potential_alpha]) and (absolute_ceiling_constant > alphas[potential_alpha]):
                        beta = beta2
                    else:
                        beta = (beta1 + beta2) / 2.0

                    # Iterate dynamic alpha pair value for loop functionality
                    changed_alpha_pairs += 1
                    """ print("\nITERATION CONSTANT IS: {}\n\nFUNCTIONAL ITERATOR IS: {}\n\nCHANGED ALPHA PAIRS ARE: \n{}\n".format(iteration_constant, iterator, changed_alpha_pairs)) """

            # Checks value of dynamic alpha pair value to iterate the method's parent iteration constant
            if (changed_alpha_pairs == 0):
                iteration_constant += 1
            else:
                iteration_constant = 0
            
            # Prints formatted iteration number for method
            """ print("\nTOTAL ITERATION NUMBER IS: {}\n".format(iteration_constant)) """

        # Get number of support vectors across SVM-SMO
        print("SUPPORT VECTORS ALONG THE SAMPLE DATASET ARE:")
        [print(dataset[iterator], labels[iterator]) for iterator in range(100) if alphas[iterator] > 0.0]

        # Prints beta-values and formatted alphas greater than zero
        print("\nBETA-VALUE IS: {}\n\nALPHAS (GREATER THAN ZERO) ARE: \n{}\n".format(beta, alphas[alphas > 0]))

        """
        # Performs runtime tracker for particular method
        self.track_runtime()
        """

        return beta, alphas

    # =========== METHOD TO CALCULATE E PARAMETER FOR SVM SMO OPTIMIZATION ===========
    def calculate_E_parameter(self, smo_support_optimizer, alpha_param):
        # Produces holding SMO optimization parameters
        fX_param = float(np.multiply(smo_support_optimizer.alphas, smo_support_optimizer.labels).T * (smo_support_optimizer.dataset * smo_support_optimizer.dataset[alpha_param, :].T)) + smo_support_optimizer.beta
        E_param = fX_param - float(smo_support_optimizer.labels[alpha_param])
        
        """ print("FIRST HOLDING SMO OPTIMIZATION PARAMETER fX IS: {}\n\nSECOND HOLDING SMO OPTIMIZATION PARAMETER E IS: {}\n".format(fX_param, E_param)) """
        return E_param

    # ====== METHOD TO SELECT OPTIMIZED ALPHA FROM SMO OPTIMIZER AND PARAMETERS ======
    # ==================== VIA AN INNER-LOOP ITERATION HEURISTIC =====================
    def inner_loop_heuristic_smo_optimization(self, iterator, smo_support_optimizer, E_iterator):
        # Predefines maximum values for change in E and alpha
        maximum_alpha_param = -1
        maximum_delta_E = 0
        E_potential_alpha = 0

        # Define error cache from SMO optimization method
        smo_support_optimizer.error_cache[iterator] = [1, E_iterator]
        valid_error_cache_list = np.nonzero(smo_support_optimizer.error_cache[:, 0].A)[0]

        # Check if error cache list length is significant
        if (len(valid_error_cache_list)) > 1:

            # Iterates through all alpha values in the error cache
            for alpha_param in valid_error_cache_list:

                # Checks if the iterator value matches the alpha value
                if alpha_param == iterator:
                    continue

                # Defines the E holding parameter from the SMO optimizer
                E_param = self.calculate_E_parameter(smo_support_optimizer, alpha_param)
                delta_E = abs(E_iterator - E_param)

                # Checks if change in E holding parameter is differentially larger than the maximum change and if so, redefines the maxes
                if (delta_E > maximum_delta_E):
                    maximum_alpha_param = alpha_param
                    maximum_delta_E = delta_E
                    E_potential_alpha = E_param
            
            return maximum_alpha_param, E_potential_alpha
        else:
            # If the error cache is not significant, defines the alpha value and holding parameter using the helper methods
            potential_alpha = self.select_random_potential_alpha(iterator, smo_support_optimizer.NUM_ROWS)
            E_potential_alpha = self.calculate_E_parameter(smo_support_optimizer, potential_alpha)
        
        """ print("POTENTIAL ALPHA VALUE IS: {}\n\nSMO OPTIMIZATION PARAMETER FOR POTENTIAL ALPHA IS: {}\n".format(potential_alpha, E_potential_alpha)) """
        return potential_alpha, E_potential_alpha

    # ====== METHOD TO SELECT OPTIMIZED ALPHA FROM SMO OPTIMIZER AND PARAMETERS ======
    # ==================== VIA AN OUTER-LOOP ITERATION HEURISTIC =====================
    def outer_loop_heuristic_smo_optimization(self, input_dataset, class_labels, absolute_ceiling_constant, alpha_tolerance, MAX_ITER, param_tuple=("lin", 0)):
        # Call the SVM-SMO Support Optimizer object
        smo_support_optimizer = Platt_SMO_Support_Optimization_Structure(np.mat(input_dataset), np.mat(class_labels).transpose(), absolute_ceiling_constant, alpha_tolerance)

        # Predefine iteration constant and boolean to track full set progress
        iteration_constant = 0
        entire_set_checked = True
        changed_alpha_pairs = 0

        # Iterates while iteration constant falls within method's boundary conditions
        while (iteration_constant < MAX_ITER) and ((changed_alpha_pairs > 0) or (entire_set_checked)):
            changed_alpha_pairs = 0

            # Checks if the entire set has been iterated through already
            if entire_set_checked:
                # Iterates through the input dataset size
                for iterator in range(smo_support_optimizer.NUM_ROWS):
                    # Increments changed alpha pairs from multilevel choice heuristic method
                    changed_alpha_pairs += self.multilevel_choice_heuristic_smo_optimization(iterator, smo_support_optimizer)
                    """ print("FOR THE FULL SET...\n\nITERATION CONSTANT IS: {}\nLOOP ITERATOR IS: {}\nCHANGED ALPHA VALUE PAIRS ARE: \n{}\n".format(iteration_constant, iterator, changed_alpha_pairs)) """
                iteration_constant += 1                     # Increments the iteration constant
            
            else:
                # Creates unbound values from nonzero alpha entries when entire set has not been checked
                unbound_values = np.nonzero((smo_support_optimizer.alphas.A > 0) * (smo_support_optimizer.alphas.A < absolute_ceiling_constant))[0]

                # Iterates through size of unbound values
                for iterator in unbound_values:
                    # Increments changed alpha pairs from multilevel choice heuristic method 
                    changed_alpha_pairs += self.multilevel_choice_heuristic_smo_optimization(iterator, smo_support_optimizer)
                    """ print("FOR THE UNBOUND VALUES...\n\nITERATION CONSTANT IS: {}\nLOOP ITERATOR IS: {}\nCHANGED ALPHA VALUE PAIRS ARE: \n{}\n".format(iteration_constant, iterator, changed_alpha_pairs)) """
                iteration_constant += 1                     # Increments the iteration constant
            
            # Alters checked set boolean based on whether set has been checked (pretty self-explanatory)
            if entire_set_checked:
                entire_set_checked = False
            elif (changed_alpha_pairs == 0):
                entire_set_checked = True
            """ print("FINAL ITERATION NUMBER IS: {}\n".format(iteration_constant)) """
        
        # Get number of support vectors across SVM-SMO
        print("\nSUPPORT VECTORS ALONG THE SAMPLE DATASET FOR THE ADVANCED SVM-SMO ARE:")
        [print(input_dataset[iterator], class_labels[iterator], sep = " --> ") for iterator in range(100) if smo_support_optimizer.alphas[iterator] > 0.0]

        # Prints SVM-SMO beta-values and formatted alphas greater than zero
        print("\nSAVED SVM-SMO BETA VALUE IS: {}\n\nSAVED SVM-SMO ALPHA (GREATER THAN ZERO) VALUES ARE: \n{}\n".format(smo_support_optimizer.beta, smo_support_optimizer.alphas[smo_support_optimizer.alphas > 0]))

        """
        # Performs runtime tracker for particular method
        self.track_runtime()
        """

        return smo_support_optimizer.beta, smo_support_optimizer.alphas

    # ====== METHOD TO SELECT OPTIMIZED ALPHA FROM SMO OPTIMIZER AND PARAMETERS ======
    # ========== VIA A MULTILEVEL SECOND-CHOICE HEURISTIC SELECTION ROUTINE ==========
    def multilevel_choice_heuristic_smo_optimization(self, iterator, smo_support_optimizer):
        E_iterator = self.calculate_E_parameter(smo_support_optimizer, iterator)

        # Checks if iteration constants abide by absolute and relative boundary conditions defined by the ceiling and tolerance levels
        if ((smo_support_optimizer.labels[iterator] * E_iterator < -smo_support_optimizer.alpha_tolerance) and (smo_support_optimizer.alphas[iterator] < smo_support_optimizer.absolute_ceiling_constant)) or ((smo_support_optimizer.labels[iterator] * E_iterator > smo_support_optimizer.alpha_tolerance) and (smo_support_optimizer.alphas[iterator] > 0)):
            potential_alpha, E_potential_alpha = self.inner_loop_heuristic_smo_optimization(iterator, smo_support_optimizer, E_iterator)
            
            # Creates dummy constants to hold old alpha values from method's parent iterator and potential alpha values
            old_alpha_iterator = np.copy(smo_support_optimizer.alphas[iterator])
            old_alpha_potential = np.copy(smo_support_optimizer.alphas[potential_alpha])

            # Checks if iterated labels match the expected potential alpha label values
            if (smo_support_optimizer.labels[iterator] != smo_support_optimizer.labels[potential_alpha]):
                # Defines the alpha's ceiling and floor if there is a mismatch
                alpha_ceiling = min(smo_support_optimizer.absolute_ceiling_constant, smo_support_optimizer.absolute_ceiling_constant + smo_support_optimizer.alphas[potential_alpha] - smo_support_optimizer.alphas[iterator])
                alpha_floor = max(0, smo_support_optimizer.alphas[potential_alpha] - smo_support_optimizer.alphas[iterator])
            else:
                # Defines the alpha's ceiling and floor if there is a match
                alpha_ceiling = min(smo_support_optimizer.absolute_ceiling_constant, smo_support_optimizer.alphas[potential_alpha] + smo_support_optimizer.alphas[iterator])
                alpha_floor = max(0, smo_support_optimizer.alphas[potential_alpha] + smo_support_optimizer.alphas[iterator] - smo_support_optimizer.absolute_ceiling_constant)

            # Checks if floor and ceiling are equivalent and if so, prints for convenience
            if (alpha_ceiling == alpha_floor):
                """ print("\nFOR ALPHA'S BOUNDARY CONSTRAINTS, THE CEILING AND FLOOR ARE FOUND TO BE EQUAL.\n") """
                return 0

            # Defines marker value for altering the alpha value for optimization
            optimal_alpha_change_marker = 2.0 * smo_support_optimizer.dataset[iterator, :] * smo_support_optimizer.dataset[potential_alpha, :].T - smo_support_optimizer.dataset[iterator, :] * smo_support_optimizer.dataset[iterator, :].T - smo_support_optimizer.dataset[potential_alpha, :] * smo_support_optimizer.dataset[potential_alpha, :].T

            # Checks if optimal alpha marker is zero and if so, prints for convenience
            if optimal_alpha_change_marker >= 0:
                """ print("\nFOR ALPHA'S OPTIMIZATION, THE VALUE OF THE OPTIMAL ALPHA CHANGE MARKER IS EQUAL TO OR GREATER THAN ZERO.\n") """
                return 0

            # Optimizes alpha values based on optimal marker and constraint processing method
            smo_support_optimizer.alphas[potential_alpha] -= smo_support_optimizer.labels[potential_alpha] * (E_iterator - E_potential_alpha) / optimal_alpha_change_marker
            smo_support_optimizer.alphas[potential_alpha] = self.process_alpha_against_constraints(smo_support_optimizer.alphas[potential_alpha], alpha_ceiling, alpha_floor)
            self.update_E_parameter(smo_support_optimizer, potential_alpha)

            # Checks if margin between new and old alphas are too small and if so, prints for convenience
            if (abs(smo_support_optimizer.alphas[potential_alpha] - old_alpha_potential) < 0.00001):
                """ print("\nTHE POTENTIAL ALPHA VALUE IS NOT MOVING ENOUGH.\n") """
                return 0

            # Increments alpha values by SMO optimizer and updates E parameter
            smo_support_optimizer.alphas[iterator] += smo_support_optimizer.labels[potential_alpha] * smo_support_optimizer.labels[iterator] * (old_alpha_potential - smo_support_optimizer.alphas[potential_alpha])
            self.update_E_parameter(smo_support_optimizer, iterator)

            # Produces temporary beta-values to track differential alpha changes
            beta1 = smo_support_optimizer.beta - E_iterator - smo_support_optimizer.labels[iterator] * (smo_support_optimizer.alphas[iterator] - old_alpha_iterator) * smo_support_optimizer.dataset[iterator, :] * smo_support_optimizer.dataset[iterator, :].T - smo_support_optimizer.labels[potential_alpha] * (smo_support_optimizer.alphas[potential_alpha] - old_alpha_potential) * smo_support_optimizer.dataset[iterator, :] * smo_support_optimizer.dataset[potential_alpha, :].T
            beta2 = smo_support_optimizer.beta - E_potential_alpha - smo_support_optimizer.labels[iterator] * (smo_support_optimizer.alphas[iterator] - old_alpha_iterator) * smo_support_optimizer.dataset[iterator, :] * smo_support_optimizer.dataset[potential_alpha, :].T - smo_support_optimizer.labels[potential_alpha] * (smo_support_optimizer.alphas[potential_alpha] - old_alpha_potential) * smo_support_optimizer.dataset[potential_alpha, :] * smo_support_optimizer.dataset[potential_alpha, :].T

            # Checks if new alpha values fall within beta-dependent boundary conditions for beta-value reinitialization
            if (0 < smo_support_optimizer.alphas[iterator]) and (smo_support_optimizer.absolute_ceiling_constant > smo_support_optimizer.alphas[iterator]):
                smo_support_optimizer.beta = beta1
            elif (0 < smo_support_optimizer.alphas[potential_alpha]) and (smo_support_optimizer.absolute_ceiling_constant > smo_support_optimizer.alphas[potential_alpha]):
                smo_support_optimizer.beta = beta2
            else:
                smo_support_optimizer.beta = (beta1 + beta2) / 2.0
            return 1
        else:
            return 0

    # ============ METHOD TO REFRESH E PARAMETER FOR SVM SMO OPTIMIZATION ============
    def update_E_parameter(self, smo_support_optimizer, alpha_param):
        # Defines the E holding parameter using the SMO optimizer helper methods
        E_param = self.calculate_E_parameter(smo_support_optimizer, alpha_param)
        smo_support_optimizer.error_cache[alpha_param] = [1, E_param]
        return

    # ======== METHOD TO RETRIEVE HYPERPLANE DISTRIBUTIVE VALUES FROM ALPHAS =========
    def get_hyperplane_from_alphas(self, alphas, input_dataset, class_labels):
        dataset = np.mat(input_dataset)
        labels = np.mat(class_labels).transpose()
        NUM_ROWS, NUM_COLS = np.shape(dataset)          # Grab dataset dimensionalities
        hyperplane = np.zeros((NUM_COLS, 1))            # Predefine hyperplane as array of zeros

        # Iterates through dataset dimensionality to produce hyperplane from alpha value spread
        for iterator in range(NUM_ROWS):
            hyperplane += np.multiply(alphas[iterator] * labels[iterator], dataset[iterator, :].T)

        print("\nHYPERPLANE DISTRIBUTIVE VALUES FROM ALPHA VALUE SPREAD IS: \n{}\n".format(hyperplane))
        return hyperplane

    # ================== METHOD TO CLASSIFY SELECT DATA AGAINST SVM ==================
    def classify_data_with_machine(self, dataset, labels, alphas, beta):
        # Calculate formatted data matrix and hyperplane distributive values from data and alpha spread
        hyperplane = self.get_hyperplane_from_alphas(alphas, dataset, labels)
        datamat = np.mat(dataset)

        # Calculate SVM data projection from data matrix, hyperplane distributive values, and beta value
        svm_data_projection = datamat * np.mat(hyperplane) + beta

        # Performs runtime tracker for particular method
        self.track_runtime()
        
        return print("PROJECTION OF NEW DATA FROM LABEL CLASSIFIER IS: \n{}\n\nEXACT LABEL OF DATA IS: \n{}\n".format(svm_data_projection[4], labels[4]))

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
# ===================== CLASS DEFINITION: SMO SUPPORT OPTIMIZER ======================
# ====================================================================================


class Platt_SMO_Support_Optimization_Structure(object):

    # ======================== CLASS INITIALIZERS/DECLARATIONS =======================
    def __init__(self, input_dataset, class_labels, absolute_ceiling_constant, alpha_tolerance):
        self.dataset = input_dataset                                        # Formatted dataset from sample data
        self.labels = class_labels                                          # Class label vector from sample data
        self.absolute_ceiling_constant = absolute_ceiling_constant          # Alpha ceiling constant for SMO boundary parametrization
        self.alpha_tolerance = alpha_tolerance                              # Alpha tolerance for SMO boundary parametrization
        self.NUM_ROWS = np.shape(input_dataset)[0]                          # Constant to hold number of rows of dataset
        self.alphas = np.mat(np.zeros((self.NUM_ROWS, 1)))                  # Alpha value range initialized as array of zeros
        self.beta = 0                                                       # SVM-SMO beta value
        self.error_cache = np.mat(np.zeros((self.NUM_ROWS, 2)))             # Caching value for tracking compounding errors


# ====================================================================================
# ================================ MAIN RUN FUNCTION =================================
# ====================================================================================


def main():
    # Track starting time of program
    TIME_I = t()

    # Initialize class instance of the support vector machine algorithm
    svm = Support_Vector_Machine_Algorithm(TIME_I)

    """
    # Test load_dataset() method on SVM
    dataset, labels = svm.load_dataset()
    """

    """
    # Test basic Platt SMO in SVM with helper methods
    dataset, labels = svm.load_dataset()
    beta, alphas = svm.simple_sequential_minimal_optimization(dataset, labels, 0.6, 0.001, 40)
    """

    """
    # Test advanced Platt SMO in SVM with helper methods, multilevel looping heuristics, and object-oriented storage
    dataset, labels = svm.load_dataset()
    beta, alphas = svm.outer_loop_heuristic_smo_optimization(dataset, labels, 0.6, 0.001, 40)
    hyperplane = svm.get_hyperplane_from_alphas(alphas, dataset, labels)
    """

    # Classify new data using advanced Platt SMO in SVM
    dataset, labels = svm.load_dataset()
    beta, alphas = svm.outer_loop_heuristic_smo_optimization(dataset, labels, 0.6, 0.001, 40)
    svm.classify_data_with_machine(dataset, labels, alphas, beta)

    return print("\nSupport vector machine class algorithm is done.\n")

if __name__ == "__main__":
    main()