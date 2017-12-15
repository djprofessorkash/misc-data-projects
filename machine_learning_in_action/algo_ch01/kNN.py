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
def create_data_set():
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

def auto_norm(data_set):
    # Calculate minimum values, maximum values, and ranges across dating data set
    min_vals = data_set.min(0)
    max_vals = data_set.max(0)
    ranges = max_vals - min_vals

    # Normalize data set using linear transformations
    normal_data_set = np.zeros(np.shape(data_set))
    transform_mat = data_set.shape[0]
    normal_data_set = data_set - np.tile(min_vals, (transform_mat, 1))
    normal_data_set = normal_data_set / np.tile(ranges, (transform_mat, 1))

    return normal_data_set, ranges, min_vals, max_vals

def dating_class_set():
    hold_back_ratio = 0.10

    dating_data_mat, dating_labels = file_to_matrix("dating_test_set.txt")

def main():
    group, labels = create_data_set()
    # print("DATA GROUP \n{}".format(group))
    # print("DATA LABELS: \n{}".format(labels))

    classifier = classify0([0, 0], group, labels, 3)
    # print("CLASSIFIER: \n{}".format(classifier))

    dating_data_mat, dating_labels = file_to_matrix("dating_test_set.txt")
    # print("DATING DATA MATRIX: \n{}".format(dating_data_mat))
    # print("FIRST TEN DATING DATA LABELS: \n{}".format(dating_labels[:10]))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dating_data_mat[:, 1], dating_data_mat[:, 2], 15.0 * np.array(dating_labels), 15.0 * np.array(dating_labels))
    # plt.show()

    norm_mat, ranges, min_vals, max_vals = auto_norm(dating_data_mat)
    print("DATING DATA MATRIX: \n{}".format(dating_data_mat))
    print("NORMALIZED MATRIX: \n{}".format(norm_mat))
    print("VALUE RANGES: \n{}".format(ranges))
    print("MINIMUM VALUES: \n{}".format(min_vals))
    print("MAXIMUM VALUES: \n{}".format(max_vals))

if __name__ == "__main__":
    main()
