"""
NAME: kNN.py (data_projects/machine_learning_in_action/algo_ch01/)
DESCRIPTION:    Tutorial program to apply the k-Nearest Neighbor ML Algorithm.

                All source code is available at www.manning.com/MachineLearningInAction. 

WARNING: Original source code is written in Python 2. 
CREDIT: Machine Learning In Action (Peter Harrington)
"""


import numpy as np
import operator as op
from os import listdir as ld
from matplotlib import pyplot as plt
# from array import array

# Function that creates data set from given arrays and labels
def example_data_set():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ["A", "A", "B", "B"]
    return group, labels


# Function that classifies inputted array based on control dataset
def classify0(inX, data_set, labels, k):
    data_set_size = data_set.shape[0]

    # Distance(s) calculation
    diff_mat = np.tile(inX, (data_set_size, 1)) - data_set
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


# Function that converts data from text file into dataset and relative labels 
def file_to_matrix(file):
    love_dictionary = {"largeDoses": 3, "smallDoses": 2, "didntLike": 1}
    f = open(file)

    array_of_lines = f.readlines()                  # Get array of lines in file
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
            class_label_vector.append(love_dictionary.get(list_from_line[-1]))
        index += 1

    return return_mat, class_label_vector

# Function that automatically normalizes datasets using linear algebra
def auto_norm(data_set):
    # Calculate minimum values, maximum values, and ranges across dating data set
    min_vals = data_set.min(0)
    max_vals = data_set.max(0)
    ranges = max_vals - min_vals

    # Normalize data set using linear transformations
    normal_data_set = np.zeros(np.shape(data_set))
    sample_data_mat = data_set.shape[0]
    normal_data_set = data_set - np.tile(min_vals, (sample_data_mat, 1))
    normal_data_set = normal_data_set / np.tile(ranges, (sample_data_mat, 1))

    return normal_data_set, ranges, min_vals, max_vals

# Function that tests our classifier against the dating data set
def dating_class_set():
    hold_back_ratio = 0.10          # Ratio to hold some testing data

    dating_data_mat, dating_labels = file_to_matrix("dating_test_set.txt")
    norm_mat, ranges, min_vals, max_vals = auto_norm(dating_data_mat)
    sample_data_mat = norm_mat.shape[0]

    # Creates error count and test vectors from 10% of dating data set
    error_count = 0.0
    num_test_vectors = int(sample_data_mat * hold_back_ratio)

    # Tests sample data in classifier function and assigns labels relatively
    for iterator in range(num_test_vectors):
        result_from_classifier = classify0(norm_mat[iterator, :], norm_mat[num_test_vectors: sample_data_mat, :], dating_labels[num_test_vectors: sample_data_mat], 3)
        print("The classifier came back with: {}. \nThe real answer is: {}.".format(result_from_classifier, dating_labels[iterator]))

        if (result_from_classifier != dating_labels[iterator]):
            error_count += 1.0
        
    # Assigns error rate indicative of failures in classifier accuracy
    print("The total error rate is: {}.".format(error_count / float(num_test_vectors)))

# Function that classifies new data entries against supervised data
def classify_person():
    # Define resultant labels and dating attributes for data set
    result_list = ["not at all", "in small doses", "in large doses"]
    attribute_percent_gaming = float(input("\nPercentage of time spent playing video games? "))
    attribute_ff_miles = float(input("Frequent flier miles earned per year? "))
    attribute_ice_cream = float(input("Liters of ice cream consumed per year? "))
    
    dating_data_mat, dating_labels = file_to_matrix("dating_test_set.txt")
    norm_mat, ranges, min_vals, max_vals = auto_norm(dating_data_mat)

    # Create array with user-entered attributes and use classifier to test attributes against training data
    attr_arr = np.array([attribute_ff_miles, attribute_percent_gaming, attribute_ice_cream])
    result_from_classifier = classify0((attr_arr - min_vals) / ranges, norm_mat, dating_labels, 3)

    print("\nYou will probably like this person... {}.".format(result_list[result_from_classifier - 1]))

def main():
    # group, labels = example_data_set()
    # print("DATA GROUP \n{}".format(group))
    # print("DATA LABELS: \n{}".format(labels))

    # classifier = classify0([0, 0], group, labels, 3)
    # print("CLASSIFIER: \n{}".format(classifier))

    # dating_data_mat, dating_labels = file_to_matrix("dating_test_set.txt")
    # print("DATING DATA MATRIX: \n{}".format(dating_data_mat))
    # print("FIRST TEN DATING DATA LABELS: \n{}".format(dating_labels[:10]))

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(dating_data_mat[:, 1], dating_data_mat[:, 2], 15.0 * np.array(dating_labels), 15.0 * np.array(dating_labels))
    # plt.show()

    # norm_mat, ranges, min_vals, max_vals = auto_norm(dating_data_mat)
    # print("DATING DATA MATRIX: \n{}".format(dating_data_mat))
    # print("NORMALIZED MATRIX: \n{}".format(norm_mat))
    # print("VALUE RANGES: \n{}".format(ranges))
    # print("MINIMUM VALUES: \n{}".format(min_vals))
    # print("MAXIMUM VALUES: \n{}".format(max_vals))

    # dating_class_set()

    classify_person()

if __name__ == "__main__":
    main()
