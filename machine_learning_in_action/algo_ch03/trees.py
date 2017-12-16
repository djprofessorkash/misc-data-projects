"""
NAME:               trees.py (data_projects/machine_learning_in_action/algo_ch03/)

DESCRIPTION:        Python class application of decision tree ML algorithms. 

                    Decision trees are advanced classification algorithms that incrementally
                    categorize by a log(n) search until all k elements are classified. Decision
                    trees work by splitting data into data subsets based on all possible
                    attribute types and dynamically splitting and testing subsets against all
                    attribute values until all data has been classified. 

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


import operator as op                       # Library for intrinsic Pythonic mathematical operations
from math import log                        # Package for performing logarithmic operations
from time import time as t                  # Package for tracking modular and program runtime


# ====================================================================================
# ================================= CLASS DEFINITION =================================
# ====================================================================================


class Decision_Tree_Algorithm(object):
    
    def __init__(self):
        pass


def create_dataset():
    dataset =  [[1, 1, "yes"],
                [1, 1, "yes"],
                [1, 0, "no"],
                [0, 1, "no"],
                [0, 1, "no"]]
    labels = ["no surfacing", "flippers"]

    print("\nSAMPLE DATA ARE: {}\nSAMPLE LABELS ARE: {}\n".format(dataset, labels))
    return dataset, labels

def calculate_Shannon_entropy(dataset):
    num_of_entries = len(dataset)
    label_counts = dict()

    # Creates dictionary of all possible class types
    for feature_vector in dataset:
        current_label = feature_vector[-1]

        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        
        label_counts[current_label] += 1
    
    # Initialize Shannon entropy, the measure of information in a dataset
    Shannon_entropy = 0.0

    # Calculates Shannon entropy based on product of information types and probabilities
    for key in label_counts:
        info_probability = float(label_counts[key] / num_of_entries)
        Shannon_entropy -= info_probability * log(info_probability, 2)

    # print("SHANNON ENTROPY OF SAMPLE DATASET IS: {}\n".format(Shannon_entropy))
    return Shannon_entropy

def split_dataset(dataset, axis, value):
    split_data = []

    # Iterate through dataset and split into subsets based on unique features
    for feature_vector in dataset:
        if feature_vector[axis] == value:
            reduced_feature_vector = feature_vector[:axis]
            reduced_feature_vector.extend(feature_vector[axis + 1:])
            split_data.append(reduced_feature_vector)

    # print("SPLITTED DATA SUBSETS ARE: {}\n".format(split_data))
    return split_data

def choose_best_feature_to_split_on(dataset):
    num_of_features = len(dataset[0]) - 1
    base_entropy = calculate_Shannon_entropy(dataset)
    best_information_gain = 0.0
    best_feature = 1

    # Create unique set of data labels to identify best feature to split on
    for feature in range(num_of_features):
        feature_list = [sample[feature] for sample in dataset]
        unique_values = set(feature_list)
        new_entropy = 0.0

        # Calculate entropy for each feature in dataset
        for value in unique_values:
            subset = split_dataset(dataset, feature, value)
            info_probability = len(subset) / float(len(dataset))
            new_entropy += info_probability * calculate_Shannon_entropy(subset)
        
        # Calculate relative information gain for particular feature
        information_gain = base_entropy - new_entropy

        # Find best information gain and best feature across all features
        if (information_gain > best_information_gain):
            best_information_gain = information_gain
            best_feature = feature
    
    print("BEST FEATURE TO SPLIT ON: {}\nRESPECTIVE BEST INFORMATION GAIN: {}\n".format(best_feature, best_information_gain))
    return best_feature
 
 
# ====================================================================================
# ================================ MAIN RUN FUNCTION =================================
# ====================================================================================


def main():
    # Track starting time of running program
    t0 = t()

    # Initialize class instance of the decision tree algorithm
    dt = Decision_Tree_Algorithm()

    """ Testing Shannon entropy function """
    # dataset, labels = create_dataset()
    # calculate_Shannon_entropy(dataset)
    
    """ Testing manipulating Shannon entropy levels """
    # dataset[0][-1] = "maybe"
    # print(dataset)
    # calculate_Shannon_entropy(dataset)

    """ Testing split_dataset() function on sample data """
    # dataset, labels = create_dataset()
    # split_dataset(dataset, 0, 1)

    """ Testing choose_best_feature_to_split_on() against sample data """
    dataset, labels = create_dataset()
    choose_best_feature_to_split_on(dataset)
    return

if __name__ == "__main__":
    main()