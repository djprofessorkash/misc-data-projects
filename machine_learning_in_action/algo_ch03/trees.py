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
    
    # ======================== CLASS INITIALIZERS/DECLARATIONS =======================
    def __init__(self):
        self.dataset = [[1, 1, "yes"],
                        [1, 1, "yes"],
                        [1, 0, "no"],
                        [0, 1, "no"],
                        [0, 1, "no"]]
        self.labels = ["no surfacing", "flippers"]

    # ================== METHOD TO CREATE SMALL DATASET FOR TESTING ==================
    def create_dataset(self):
        dataset =  [[1, 1, "yes"],
                    [1, 1, "yes"],
                    [1, 0, "no"],
                    [0, 1, "no"],
                    [0, 1, "no"]]
        labels = ["no surfacing", "flippers"]

        print("\nSAMPLE DATA ARE: {}\nSAMPLE LABELS ARE: {}\n".format(dataset, labels))
        return dataset, labels

    # ================ METHOD TO CALCULATE SHANNON ENTROPY OF DATASET ================
    def calculate_Shannon_entropy(self, dataset):
        num_of_entries = len(dataset)
        label_counts = dict()

        # Creates dictionary of all possible class types
        for feature_vector in dataset:
            current_label = feature_vector[-1]

            # Creates histogram of label counts based on unique class types
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

    # ================ METHOD TO SPLIT DATASET BASED ON UNIQUE FEATURE ===============
    def split_dataset(self, dataset, axis, value):
        split_data = []

        # Iterate through dataset and split into subsets based on unique features
        for feature_vector in dataset:
            if feature_vector[axis] == value:
                reduced_feature_vector = feature_vector[:axis]
                reduced_feature_vector.extend(feature_vector[axis + 1:])
                split_data.append(reduced_feature_vector)

        # print("SPLITTED DATA SUBSETS ARE: {}\n".format(split_data))
        return split_data

    # ============ METHOD TO CHOOSE BEST FEATURE ON WHICH TO SPLIT DATASET ===========
    def choose_best_feature_to_split_on(self, dataset):
        num_of_features = len(dataset[0]) - 1
        base_entropy = self.calculate_Shannon_entropy(dataset)
        best_information_gain = 0.0
        best_feature = 1

        # Create unique set of data labels to identify best feature to split on
        for feature in range(num_of_features):
            feature_list = [sample[feature] for sample in dataset]
            unique_values = set(feature_list)
            new_entropy = 0.0

            # Calculate entropy for each feature in dataset
            for value in unique_values:
                subset = self.split_dataset(dataset, feature, value)
                info_probability = len(subset) / float(len(dataset))
                new_entropy += info_probability * self.calculate_Shannon_entropy(subset)
            
            # Calculate relative information gain for particular feature
            information_gain = base_entropy - new_entropy

            # Find best information gain and best feature across all features
            if (information_gain > best_information_gain):
                best_information_gain = information_gain
                best_feature = feature
        
        # print("BEST FEATURE TO SPLIT ON: {}\nRESPECTIVE BEST INFORMATION GAIN: {}\n".format(best_feature, best_information_gain))
        return best_feature

    # ================ METHOD TO CREATE HISTOGRAM OF SORTED DECISIONS ================
    def majority_histogram(self, class_list):
        histogram = dict()

        # Creates histogram distribution of class_list element occurrences
        for vote in class_list:
            if vote not in histogram.keys(): 
                histogram[vote] = 0
            histogram[vote] += 1

        # Sorts histogram by keys
        sorted_histogram = sorted(histogram.items(), key = op.itemgetter(1), reverse = True)
        
        print("SORTED HISTOGRAM IS: {}\n".format(sorted_histogram))
        return sorted_histogram[0][0]

    # ================== METHOD TO CREATE DECISION TREE FROM DATASET =================
    def create_tree(self, dataset, labels):
        class_list = [sample[-1] for sample in dataset]

        # Stops iteration through decision tree when all classes are equal
        if class_list.count(class_list[0]) == len(class_list):
            return class_list[0]

        # Returns majority histogram when there are no more features left to iterate through
        if len(dataset[0]) == 1:
            return self.majority_histogram(class_list)

        # Define best feature, best feature, and create decision tree object
        best_feature = self.choose_best_feature_to_split_on(dataset)
        best_feature_label = labels[best_feature]
        decision_tree = {best_feature_label: {}}

        # Get set of unique values from features of dataset
        del(labels[best_feature])
        feature_values = [sample[best_feature] for sample in dataset]
        unique_values = set(feature_values)

        # Recursively iterate through decision trees to sort data by sublabels of dataset
        for value in unique_values:
            sublabels = labels[:]
            decision_tree[best_feature_label][value] = self.create_tree(self.split_dataset(dataset, best_feature, value), sublabels)

        # print("DECISION TREE: {}\n".format(decision_tree))
        return decision_tree

 
# ====================================================================================
# ================================ MAIN RUN FUNCTION =================================
# ====================================================================================


def main():
    # Track starting time of running program
    t0 = t()

    # Initialize class instance of the decision tree algorithm
    dt = Decision_Tree_Algorithm()

    dataset, labels = dt.create_dataset()
    decision_tree = dt.create_tree(dataset, labels)
    print("COMPLETE DECISION TREE: {}\n".format(decision_tree))
    return

if __name__ == "__main__":
    main()