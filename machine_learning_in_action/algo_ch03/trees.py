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

    print("SHANNON ENTROPY OF SAMPLE DATASET IS: {}\n".format(Shannon_entropy))
    return Shannon_entropy
 
 
# ====================================================================================
# ================================ MAIN RUN FUNCTION =================================
# ====================================================================================


def main():
    # Track starting time of running program
    t0 = t()

    # Initialize class instance of the decision tree algorithm
    dt = Decision_Tree_Algorithm()

    dataset, labels = create_dataset()
    calculate_Shannon_entropy(dataset)
    
    # dataset[0][-1] = "maybe"
    # print(dataset)
    # calculate_Shannon_entropy(dataset)
    return

if __name__ == "__main__":
    main()