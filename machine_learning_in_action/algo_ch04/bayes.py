"""
NAME:               bayes.py (data_projects/machine_learning_in_action/algo_ch03/)

DESCRIPTION:        Python class application of the Naïve Bayes classifier algorithm.  

                    ???

                    All source code is available at www.manning.com/MachineLearningInAction. 

USE CASE(S):        Nominal values

ADVANTAGE(S):       Works with small amount of data
                    Handles multiple classes

DISADVANTAGE(S):    Sensitive to how input data is prepared

NOTE:               Original source code is Python 2, but my code is Python 3.

CREDIT:             Machine Learning In Action (Peter Harrington)
"""


# ====================================================================================
# ================================ IMPORT STATEMENTS =================================
# ====================================================================================


from time import time as t                  # Package for tracking modular and program runtime


# ====================================================================================
# ================================= CLASS DEFINITION =================================
# ====================================================================================


class Naïve_Bayes_Classifier_Algorithm(object):

    def __init__(self):
        pass

    def load_data_set(self):
        posting_list = [["my", "dog", "has", "flea", "problems", "help", "please"],
                        ["maybe", "not", "take", "him", "to", "dog", "park", "stupid"],
                        ["my", "dalmation", "is", "so", "cute", "I", "love", "him"],
                        ["stop", "posting", "stupid", "worthless", "garbage"],
                        ["mr", "licks", "ate", "my", "steak", "how", "to", "stop", "him"],
                        ["quit", "buying", "worthless", "dog", "food", "stupid"]]
        class_vector = [0, 1, 0, 1, 0, 1]

        print("\nPOSTING LIST IS: {}\nCLASS VECTOR IS: {}\n".format(posting_list, class_vector))
        return posting_list, class_vector

    def create_vocab_list(self, dataset):
        vocab_set = set([])

        for document in dataset:
            vocab_set = vocab_set | set(document)

        print("LIST OF VOCABULARY WORDS IS: {}\n".format(list(vocab_set)))
        return list(vocab_set)

    def convert_word_set_to_vector(self, vocab_list, input_set):
        return_vector = [0] * len(vocab_list)

        for word in input_set:
            if word in vocab_list:
                return_vector[vocab_list.index(word)] = 1
            else:
                print("The word '{}' is not in my vocabulary. ".format(word))
        
        print("RETURN VECTOR IS: {}\n".format(return_vector))
        return return_vector

def main():
    # Track starting time of program
    t0 = t()

    # Initialize class instance of the naïve Bayes classifier algorithm
    bayes = Naïve_Bayes_Classifier_Algorithm()

    list_of_posts, list_of_classes = bayes.load_data_set()
    list_of_vocab_words = bayes.create_vocab_list(list_of_posts)

    bayes.convert_word_set_to_vector(list_of_vocab_words, list_of_posts[3])

    # Track ending time of program and determine overall program runtime
    t1 = t()
    delta = (t1 - t0) * 1000

    print("Real program runtime is {0:.4g} milliseconds.\n".format(delta))
    return

if __name__ == "__main__":
    main()