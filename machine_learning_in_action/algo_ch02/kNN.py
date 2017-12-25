"""
NAME:               kNN.py (data_projects/machine_learning_in_action/algo_ch02/)

DESCRIPTION:        Python class application of the k-Nearest Neighbor ML algorithm. 

                    The kNN algorithm is a basic machine learning algorithm that compares new 
                    data based on comparative similarities to available training data and looks
                    for up to k nearest neighbors to the new data to assign contextual labels.

                    All source code is available at www.manning.com/MachineLearningInAction. 

USE CASE(S):        Numeric and nominal values

ADVANTAGE(S):       High accuracy
                    Outlier insensitivity
                    No assumptions about data 

DISADVANTAGE(S):    Computationally expensive
                    Requires considerable memory 

NOTE:               The handwriting application of kNN requires a subdirectory of images called 
                    'digits' that is too large to upload to GitHub. Instead, download and unpack
                    the file titled 'digits.zip'.

                    Original source code is Python 2, but my code is Python 3.

CREDIT:             Machine Learning In Action (Peter Harrington)
"""


# ====================================================================================
# ================================ IMPORT STATEMENTS =================================
# ====================================================================================


import numpy as np                          # Library for simple linear mathematical operations
import operator as op                       # Library for intrinsic Pythonic mathematical operations
import matplotlib.pyplot as plt             # Module for MATLAB-like data visualization capability
from os import listdir as ld                # Package for returning list of directory filenames
from time import time as t                  # Package for tracking modular and program runtime


# ====================================================================================
# ================================= CLASS DEFINITION =================================
# ====================================================================================


class k_Nearest_Neighbors_Algorithm(object):

    # ======================== CLASS INITIALIZERS/DECLARATIONS =======================
    def __init__(self):
        self.FILE = open("dating_test_set.txt")                 # Open dating test set as active file
        self.TRAINING_DIGITS = "./digits/training_digits"       # Directory reference to handwritten training digits
        self.TEST_DIGITS = "./digits/test_digits"               # Directory reference to handwritten test digits
        self.SAMPLING_RATIO = 0.10                              # Ratio to hold some testing data

    # ================= METHOD THAT CLASSIFIES DATASET AGAINST LABELS ================
    def basic_label_classifier(self, inX, dataset, labels, k):
        dataset_size = dataset.shape[0]

        # Distance(s) calculation
        diff_mat = np.tile(inX, (dataset_size, 1)) - dataset
        sq_diff_mat = diff_mat ** 2
        sq_distances = sq_diff_mat.sum(axis = 1)
        distances = sq_distances ** 0.5

        sorted_dist_indices = distances.argsort()
        class_count = {}

        # Iterate through k neighbors and select voting labels with lowest k distances
        for iterator in range(k):
            voting_label = labels[sorted_dist_indices[iterator]]
            class_count[voting_label] = class_count.get(voting_label, 0) + 1

        # Sort dictionary
        sorted_class_count = sorted(class_count.items(), key = op.itemgetter(1), reverse = True)

        """ print("FIRST ENTRY OF SORTED CLASS COUNT IS: \n{}\n".format(sorted_class_count[0][0])) """
        return sorted_class_count[0][0]

    # =========== METHOD THAT CONVERTS FILE TO DATASET AND VECTOR OF LABELS ==========
    def convert_file_to_matrix(self):
        classifier_dictionary = {"largeDoses": 3, "smallDoses": 2, "didntLike": 1}

        array_of_lines = self.FILE.readlines()              # Get array of lines in file
        num_of_lines = len(array_of_lines)                  # Get number of lines in file

        # Create NumPy matrix and labels to return
        returning_dataset = np.zeros((num_of_lines, 3))
        class_label_vector = []
        index = 0

        # Parse line to a list
        for line in array_of_lines:
            line = line.strip()
            list_from_line = line.split("\t")
            returning_dataset[index, :] = list_from_line[0:3]

            # If previous list from line is a number, append it to the class label vector
            if(list_from_line[-1].isdigit()):
                class_label_vector.append(int(list_from_line[-1]))
            else:
                class_label_vector.append(classifier_dictionary.get(list_from_line[-1]))

            index += 1

        """ print("CONVERTED DATASET IS: \n{}\nCLASS LABEL VECTOR IS: \n{}\n".format(returning_dataset, class_label_vector)) """
        return returning_dataset, class_label_vector

    # ===================== METHOD THAT CONVERTS IMAGE TO VECTOR =====================
    def convert_image_to_vector(self, file):
        image_vector = np.zeros((1, 1024))
        IMAGE = open(file)

        # Converts 32x32 image to 1x1024 vector
        for iterator_outer in range(32):
            line = IMAGE.readline()

            for iterator_inner in range(32):
                image_vector[0, 32 * iterator_outer + iterator_inner] = int(line[iterator_inner])

        """ print("SAMPLE IMAGE VECTOR, FIRST 32 DIGITS: \n{}.\nSAMPLE IMAGE VECTOR, SECOND 32 DIGITS: \n{}.\n".format(image_vector[0, 0:31], image_vector[0, 32:63])) """
        return image_vector

    # =================== METHOD THAT LINEARLY NORMALIZES DATASETS ===================
    def auto_linear_normalization(self, dataset):
        # Calculate minimum values, maximum values, and ranges across dating data set
        min_vals = dataset.min(0)
        max_vals = dataset.max(0)
        ranges = max_vals - min_vals

        # Normalize data set using linear algebraic transformations
        sample_dataset = dataset.shape[0]
        norm_dataset = np.zeros(np.shape(dataset))
        norm_dataset = dataset - np.tile(min_vals, (sample_dataset, 1))
        norm_dataset = norm_dataset / np.tile(ranges, (sample_dataset, 1))

        """ print("\nDATING DATASET: \n{}\n\nNORMALIZED DATASET: \n{}\n\nVALUE RANGES: \n{}\n\nMINIMUM VALUES: \n{}\n\nMAXIMUM VALUES: \n{}".format(dataset, norm_dataset, ranges, min_vals, max_vals)) """
        return norm_dataset, ranges, min_vals, max_vals

    # ================== METHOD THAT CREATES DATING DATA SCATTERPLOT =================
    def create_scatterplot(self, t0):
        fig = plt.figure()
        ax = fig.add_subplot(111)               # Creates visual subplot using MatPlotLib

        dating_dataset, dating_labels = self.convert_file_to_matrix()
        """ print("\nDATING DATASET: \n{}\n\nFIRST TWENTY DATING DATA LABELS: \n{}".format(dating_dataset, dating_labels[:20])) """

        ax.scatter(dating_dataset[:, 1], dating_dataset[:, 2], 15.0 * np.array(dating_labels), 15.0 * np.array(dating_labels))
        self.calculate_runtime(t0)
        plt.show()
        return

    # ================ METHOD THAT USES CLASSIFIER AGAINST DATING DATA ===============
    def dating_class_set(self, t0):
        dating_dataset, dating_labels = self.convert_file_to_matrix()
        norm_dataset, ranges, min_vals, max_vals = self.auto_linear_normalization(dating_dataset)
        sample_dataset = norm_dataset.shape[0]
        # print("\nDATING DATASET: \n{}\n\nFIRST TWENTY DATING DATA LABELS: \n{}".format(dating_dataset, dating_labels[:20]))
        # print("\nNORMALIZED DATASET: \n{}\n\nVALUE RANGES: \n{}\n\nMINIMUM VALUES: \n{}\n\nMAXIMUM VALUES: \n{}".format(norm_dataset, ranges, min_vals, max_vals))

        # Creates error count and test vectors from 10% of dating data set
        error_count = 0.0
        num_test_vectors = int(sample_dataset * self.SAMPLING_RATIO)

        # Tests sample data in classifier function and assigns labels relatively
        for iterator in range(num_test_vectors):
            classifier_response = self.basic_label_classifier(norm_dataset[iterator, :], norm_dataset[num_test_vectors: sample_dataset, :], dating_labels[num_test_vectors: sample_dataset], 3)
            print("The classifier came back with: {}. \nThe real answer is: {}.".format(classifier_response, dating_labels[iterator]))

            if (classifier_response != dating_labels[iterator]):
                error_count += 1.0
            
        # Assigns error rate indicative of failures in classifier accuracy
        print("\nThe total error rate is: {}.".format(error_count / float(num_test_vectors)))
        self.calculate_runtime(t0)
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
        
        dating_dataset, dating_labels = self.convert_file_to_matrix()
        norm_dataset, ranges, min_vals, max_vals = self.auto_linear_normalization(dating_dataset)

        # Create array with user-entered attributes and use classifier to test attributes against training data
        attr_arr = np.array([attribute_ff_miles, attribute_percent_gaming, attribute_ice_cream])
        classifier_response = self.basic_label_classifier((attr_arr - min_vals) / ranges, norm_dataset, dating_labels, 3)

        print("\nYou will probably like this person... {}.".format(result_list[classifier_response - 1]))
        self.calculate_runtime(t0, t_user_start, t_user_end)
        return

    # ========= METHOD THAT APPLIES CLASSIFIER AGAINST HANDWRITTEN IMAGE DATA ========
    def handwriting_class_test(self, t0):
        handwriting_labels = []
        training_file_list = ld(self.TRAINING_DIGITS)
        dir_length_training = len(training_file_list)
        training_mat = np.zeros((dir_length_training, 1024))

        # Create dataset and label vectors from training image data
        for iterator in range(dir_length_training):
            file_name_str = training_file_list[iterator]
            file_str = file_name_str.split(".")[0]
            class_num_str = int(file_str.split("_")[0])

            handwriting_labels.append(class_num_str)
            training_mat[iterator, :] = self.convert_image_to_vector("{}/{}".format(self.TRAINING_DIGITS, file_name_str))
        
        error_count = 0.0
        test_file_list = ld(self.TEST_DIGITS)
        dir_length_test = len(test_file_list)

        # Create dataset and label vectors from test image data, then use with classifier against training data
        for iterator in range(dir_length_test):
            file_name_str = test_file_list[iterator]
            file_str = file_name_str.split(".")[0]
            class_num_str = int(file_str.split("_")[0])

            vector_under_test = self.convert_image_to_vector("{}/{}".format(self.TEST_DIGITS, file_name_str))
            classifier_response = self.basic_label_classifier(vector_under_test, training_mat, handwriting_labels, 3)

            print("The classifier came back with: {}.\nThe real answer is: {}.\n".format(classifier_response, class_num_str))

            if (classifier_response != class_num_str):
                error_count += 1.0

        print("\nThe total number of errors is: {}.\nThe total error rate is: {}.\n".format(error_count, error_count / float(dir_length_test)))
        self.calculate_runtime(t0)
        return

    # ============ METHOD THAT CALCULATES METHOD-DEPENDENT PROGRAM RUNTIME ===========
    def calculate_runtime(self, t0, t_user_start=0, t_user_end=0):
        t1 = t()
        delta = t1 - (t_user_end - t_user_start) - t0

        # Print final statement on program runtime
        print("\nReal program runtime is {0:.4g} seconds.\n".format(delta))
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
    # kNN.classify_person(t0)
    kNN.handwriting_class_test(t0)
    return

if __name__ == "__main__":
    main()
