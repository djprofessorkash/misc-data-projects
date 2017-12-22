"""
NAME:               trees.py (data_projects/machine_learning_in_action/algo_ch03/)

DESCRIPTION:        Python class application of the ID3 decision tree ML algorithms. 

                    Decision trees are advanced classification algorithms that incrementally
                    categorize by a log(n) search until all k elements are classified. Decision
                    trees work by splitting data into data subsets based on all possible
                    attribute types and dynamically splitting and testing subsets against all
                    attribute values until all data has been classified. The specific algorithm
                    used in this code is the Iterative Dichotomiser 3 Decision Tree Algorithm.

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


import tree_plotter as dt_plt               # Modular program for visualizing decision trees as plots
import pickle as rick                       # Library for serializing Python objects (http://tiny.cc/picklerick)
import operator as op                       # Library for intrinsic Pythonic mathematical operations
from math import log                        # Package for performing logarithmic operations
from time import time as t                  # Package for tracking modular and program runtime


# ====================================================================================
# ================================= CLASS DEFINITION =================================
# ====================================================================================


class ID3_Decision_Tree_Algorithm(object):
    
    # ======================== CLASS INITIALIZERS/DECLARATIONS =======================
    # TODO: Initialize dataset and labels here, then reference throughout methods
    def __init__(self):
        """
        self.dataset = [[1, 1, "yes"],
                        [1, 1, "yes"],
                        [1, 0, "no"],
                        [0, 1, "no"],
                        [0, 1, "no"]]
        self.labels = ["no surfacing", "flippers"]
        """
        pass

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

    # =============== METHOD TO CLASSIFY DATA WITH CLASS LABEL VECTOR ================
    def classify(self, decision_tree, feature_labels, test_vector):
        tree_string = list(decision_tree)[0]
        tree_dictionary = decision_tree[tree_string]
        feature_index = feature_labels.index(tree_string)

        # Recursively loop through tree keys to create class label at leaf node of best fit
        for key in tree_dictionary.keys():
            if test_vector[feature_index] == key:
                if type(tree_dictionary[key]).__name__ == "dict":
                    class_label = self.classify(tree_dictionary[key], feature_labels, test_vector)
                else:
                    class_label = tree_dictionary[key]
        
        # print("CLASS LABEL IS: {}\n".format(class_label))
        return class_label

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

    # ===================== METHOD TO STORE DECISION TREE IN FILE ====================
    def store_tree(self, decision_tree, file):
        f = open(file, "wb")
        rick.dump(decision_tree, f)
        f.close()
        return

    # ===================== METHOD TO GRAB DECISION TREE IN FILE =====================
    def grab_tree(self, file):
        f = open(file, "rb")
        return rick.load(f)

 
# ====================================================================================
# ================================ MAIN RUN FUNCTION =================================
# ====================================================================================


def main():
    # Track starting time of program
    t0 = t()

    # Initialize class instance of the decision tree algorithm
    dt = ID3_Decision_Tree_Algorithm()

    # Playing with the Lenses dataset
    f = open("lenses.txt")
    lenses = [line.strip().split("\t") for line in f.readlines()]
    lenses_labels = ["age", "prescript", "astigmatic", "tear_rate"]
    lenses_tree = dt.create_tree(lenses, lenses_labels)
    print("\nDECISION TREE FOR THE LENSES DATASET IS: {}\n".format(lenses_tree))
    dt_plt.create_plot(t0, lenses_tree)

    # Run testing methods on decision tree algorithm
    """
    dataset, labels = dt.create_dataset()
    tree = dt_plt.retrieve_tree(0)
    dt.store_tree(tree, "classifier_storage.txt")
    grabbed_tree = dt.grab_tree("classifier_storage.txt")
    print("GRABBED DECISION TREE IS: {}\n".format(grabbed_tree))
    """

    # Classify new test vector against decision tree
    """
    dataset, labels = dt.create_dataset()
    tree = dt_plt.retrieve_tree(0)
    class_label = dt.classify(tree, labels, [1, 1]).upper()
    print("CLASS LABEL RESULT IS: {}\n".format(class_label))
    """
    

    # Create decision tree from dataset and labels
    """
    dataset, labels = dt.create_dataset()
    decision_tree = dt.create_tree(dataset, labels)
    print("COMPLETE DECISION TREE: {}\n".format(decision_tree))
    """

    # Track ending time of program and determine overall program runtime
    t1 = t()
    delta = (t1 - t0) * 1000

    print("Real program runtime is {0:.4g} milliseconds.\n".format(delta))
    return

if __name__ == "__main__":
    main()