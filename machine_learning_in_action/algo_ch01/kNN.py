"""
NAME: kNN.py (data_projects/machine_learning_in_action/algo_ch01/)
DESCRIPTION:    Tutorial program to apply the k-Nearest Neighbor ML Algorithm.

                All source code is available at www.manning.com/MachineLearningInAction. 

WARNING: Original source code is written in Python 2. 
CREDIT: Machine Learning In Action (Peter Harrington)
"""


# ====================================================================================
# ================================ IMPORT STATEMENTS =================================
# ====================================================================================


import numpy as np                          # Library for simple linear mathematical operations
import operator as op                       # Library for intrinsic Pythonic mathematical operations
from os import listdir as ld                # Module for returning list of directory filenames
import matplotlib.pyplot as plt             # Module for MATLAB-like data visualization capability
from time import time as t                  # Module for tracking modular and program runtime


# ====================================================================================
# ================================= CLASS DEFINITION =================================
# ====================================================================================


class k_Nearest_Neighbors_Algorithm(object):

    # ======================== CLASS INITIALIZERS/DECLARATIONS =======================
    def __init__(self):
        self.f = open("dating_test_set.txt")        # Open dating test set as active file
        self.sampling_ratio = 0.10                  # Ratio to hold some testing data

    # ================= METHOD THAT CLASSIFIES DATASET AGAINST LABELS ================
    def classify0(self, inX, dataset, labels, k):
        dataset_size = dataset.shape[0]

        # Distance(s) calculation
        diff_mat = np.tile(inX, (dataset_size, 1)) - dataset
        sq_diff_mat = diff_mat ** 2
        sq_distances = sq_diff_mat.sum(axis = 1)
        distances = sq_distances ** 0.5

        sorted_dist_indices = distances.argsort()
        class_count = {}

        # Voting with lowest k distances
        for iterator in range(k):
            voting_label = labels[sorted_dist_indices[iterator]]
            class_count[voting_label] = class_count.get(voting_label, 0) + 1

        # Sort dictionary
        sorted_class_count = sorted(class_count.items(), key = op.itemgetter(1), reverse = True)
        return sorted_class_count[0][0]

    # =========== METHOD THAT CONVERTS FILE TO DATASET AND VECTOR OF LABELS ==========
    def file_to_matrix(self):
        classifier_dictionary = {"largeDoses": 3, "smallDoses": 2, "didntLike": 1}

        array_of_lines = self.f.readlines()             # Get array of lines in file
        num_of_lines = len(array_of_lines)              # Get number of lines in file

        # Create NumPy matrix and labels to return
        return_mat = np.zeros((num_of_lines, 3))
        class_label_vector = []

        # Parse line to a list
        index = 0
        for line in array_of_lines:
            line = line.strip()
            list_from_line = line.split("\t")
            return_mat[index, :] = list_from_line[0:3]

            # If previous list from line is a number, append it to the class label vector
            if(list_from_line[-1].isdigit()):
                class_label_vector.append(int(list_from_line[-1]))
            else:
                class_label_vector.append(classifier_dictionary.get(list_from_line[-1]))
            index += 1
        return return_mat, class_label_vector

    # =================== METHOD THAT LINEARLY NORMALIZES DATASETS ===================
    def auto_norm(self, dataset):
        # Calculate minimum values, maximum values, and ranges across dating data set
        min_vals = dataset.min(0)
        max_vals = dataset.max(0)
        ranges = max_vals - min_vals

        # Normalize data set using linear algebraic transformations
        sample_data_mat = dataset.shape[0]
        norm_mat = np.zeros(np.shape(dataset))
        norm_mat = dataset - np.tile(min_vals, (sample_data_mat, 1))
        norm_mat = norm_mat / np.tile(ranges, (sample_data_mat, 1))

        # print("\nDATING DATA MATRIX: \n{}\n\nNORMALIZED MATRIX: \n{}\n\nVALUE RANGES: \n{}\n\nMINIMUM VALUES: \n{}\n\nMAXIMUM VALUES: \n{}".format(dataset, norm_mat, ranges, min_vals, max_vals))
        return norm_mat, ranges, min_vals, max_vals

    # ================== METHOD THAT CREATES DATING DATA SCATTERPLOT =================
    def create_scatterplot(self, t0):
        fig = plt.figure()
        ax = fig.add_subplot(111)               # Creates visual subplot using MatPlotLib

        dating_data_mat, dating_labels = self.file_to_matrix()
        # print("\nDATING DATA MATRIX: \n{}\n\nFIRST TWENTY DATING DATA LABELS: \n{}".format(dating_data_mat, dating_labels[:20]))

        ax.scatter(dating_data_mat[:, 1], dating_data_mat[:, 2], 15.0 * np.array(dating_labels), 15.0 * np.array(dating_labels))
        plt.show()
        return

    # ================ METHOD THAT USES CLASSIFIER AGAINST DATING DATA ===============
    def dating_class_set(self, t0):
        dating_data_mat, dating_labels = self.file_to_matrix()
        norm_mat, ranges, min_vals, max_vals = self.auto_norm(dating_data_mat)
        sample_data_mat = norm_mat.shape[0]
        # print("\nDATING DATA MATRIX: \n{}\n\nFIRST TWENTY DATING DATA LABELS: \n{}".format(dating_data_mat, dating_labels[:20]))
        # print("\nNORMALIZED MATRIX: \n{}\n\nVALUE RANGES: \n{}\n\nMINIMUM VALUES: \n{}\n\nMAXIMUM VALUES: \n{}".format(norm_mat, ranges, min_vals, max_vals))

        # Creates error count and test vectors from 10% of dating data set
        error_count = 0.0
        num_test_vectors = int(sample_data_mat * self.sampling_ratio)

        # Tests sample data in classifier function and assigns labels relatively
        for iterator in range(num_test_vectors):
            result_from_classifier = self.classify0(norm_mat[iterator, :], norm_mat[num_test_vectors: sample_data_mat, :], dating_labels[num_test_vectors: sample_data_mat], 3)
            print("The classifier came back with: {}. \nThe real answer is: {}.".format(result_from_classifier, dating_labels[iterator]))

            if (result_from_classifier != dating_labels[iterator]):
                error_count += 1.0
            
        # Assigns error rate indicative of failures in classifier accuracy
        print("The total error rate is: {}.".format(error_count / float(num_test_vectors)))
        return

    # =========== METHOD THAT CLASSIFIES NEW USER ENTRY AGAINST DATING DATA ==========
    def classify_person(self, t0):
        # Define resultant labels and dating attributes for data set
        result_list = ["not at all", "in small doses", "in large doses"]
        t_user_start = t()
        attribute_percent_gaming = float(input("\nPercentage of time spent playing video games?  > "))
        attribute_ff_miles = float(input("Frequent flier miles earned per year?  > "))
        attribute_ice_cream = float(input("Liters of ice cream consumed per year?  > "))
        t_user_end = t()
        
        dating_data_mat, dating_labels = self.file_to_matrix()
        norm_mat, ranges, min_vals, max_vals = self.auto_norm(dating_data_mat)

        # Create array with user-entered attributes and use classifier to test attributes against training data
        attr_arr = np.array([attribute_ff_miles, attribute_percent_gaming, attribute_ice_cream])
        result_from_classifier = self.classify0((attr_arr - min_vals) / ranges, norm_mat, dating_labels, 3)

        print("\nYou will probably like this person... {}.\n".format(result_list[result_from_classifier - 1]))
        self.calculate_runtime(t0, t_user_start, t_user_end)
        return

    # ============ METHOD THAT CALCULATES METHOD-DEPENDENT PROGRAM RUNTIME ===========
    def calculate_runtime(self, t0, t_user_start = 0, t_user_end = 0):
        t1 = t()
        delta = t1 - (t_user_end - t_user_start) - t0

        # Print final statement on program runtime
        print("Total program runtime is {0:.4g} seconds.\n".format(delta))
        return


# ====================================================================================
# ================================= MAIN RUN FUNCTION ================================
# ====================================================================================


def main():
    # Track starting time of running program
    t0 = t()

    # Initialize class instance of the kNN algorithm and test kNN classifier methods
    # NOTE: Run one kNN classifier method at a time for runtime accuracy
    kNN = k_Nearest_Neighbors_Algorithm()

    # kNN.create_scatterplot(t0)
    # kNN.dating_class_set(t0)
    kNN.classify_person(t0)
    return

if __name__ == "__main__":
    main()
