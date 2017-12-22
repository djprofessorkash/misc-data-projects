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

                    In order for dependencies to work, user must install the 'FEEDPARSER'
                    module using their preferred Python package manager.

                    For Python 2.x or older Python versions, use PIP to install:
                        usr$  pip install feedparser

                    For Python 3.x, use PIP3 to install:
                        usr$  pip3 install feedparser

                    For Anaconda 3.x, use Anaconda's built-in package manger Conda to install:
                        usr$  conda install -c anaconda feedparser
                    
                    For other package managers, please read your package manager documentation. 

CREDIT:             Machine Learning In Action (Peter Harrington)
"""


# ====================================================================================
# ================================ IMPORT STATEMENTS =================================
# ====================================================================================


import re                                   # Library for regular expression support
import numpy as np                          # Library for simple linear mathematical operations
import operator as op                       # Library for intrinsic Pythonic mathematical operations
import feedparser as fp                     # Library for universal parsing of web information feeds
from time import time as t                  # Package for tracking modular and program runtime


# ====================================================================================
# ================================= CLASS DEFINITION =================================
# ====================================================================================


class Naïve_Bayes_Classifier_Algorithm(object):

    # ======================== CLASS INITIALIZERS/DECLARATIONS =======================
    def __init__(self):
        pass

    # ========================= METHOD TO LOAD SAMPLE DATASET ========================
    def load_data_set(self):
        posting_list = [["my", "dog", "has", "flea", "problems", "help", "please"],
                        ["maybe", "not", "take", "him", "to", "dog", "park", "stupid"],
                        ["my", "dalmation", "is", "so", "cute", "I", "love", "him"],
                        ["stop", "posting", "stupid", "worthless", "garbage"],
                        ["mr", "licks", "ate", "my", "steak", "how", "to", "stop", "him"],
                        ["quit", "buying", "worthless", "dog", "food", "stupid"]]
        class_vector = [0, 1, 0, 1, 0, 1]       # 0: not abusive, 1: abusive

        print("\nPOSTING LIST IS: {}\nCLASS VECTOR IS: {}\n".format(posting_list, class_vector))
        return posting_list, class_vector

    # ========================== METHOD TO TRAIN BAYES MODEL =========================
    def naïve_bayes_trainer(self, training_matrix, training_category):
        number_of_training_documents = len(training_matrix)
        number_of_words = len(training_matrix[0])

        # Initialize relative conditional and partial probabilities
        p_abusive = sum(training_category) / float(number_of_training_documents)
        p0_numerator = np.ones(number_of_words)
        p1_numerator = np.ones(number_of_words)
        p0_denominator = 2.0
        p1_denominator = 2.0

        # Iterate through training word documents and add vectors together for conditional probability equation
        for iterator in range(number_of_training_documents):
            if training_category[iterator] == 1:
                p1_numerator += training_matrix[iterator]
                p1_denominator += sum(training_matrix[iterator])
            else:
                p0_numerator += training_matrix[iterator]
                p0_denominator += sum(training_matrix[iterator])
        
        # Calculate naïve conditional probability vectors
        p0_vector = np.log(p0_numerator / p0_denominator)
        p1_vector = np.log(p1_numerator / p1_denominator)

        # print("PROBABILITY VECTOR FOR NORMAL WORDS IS: \n\n{}\n\nPROBABILITY VECTOR FOR ABUSIVE WORDS IS: \n\n{}\n\nPROBABILITY OF ANY DOCUMENT BEING ABUSIVE IS: {}\n".format(p0_vector, p1_vector, p_abusive))
        return p0_vector, p1_vector, p_abusive

    # ==================== METHOD TO CLASSIFY DATA IN BAYES MODEL ====================
    def classify_naïve_bayes(self, vector_to_classify, p0_vector, p1_vector, p_test_class):
        # Multiply element vectors together for summative accuracy
        p0 = sum(vector_to_classify * p0_vector) + np.log(1.0 - p_test_class)
        p1 = sum(vector_to_classify * p1_vector) + np.log(p_test_class)

        # Returns classifier if entry fits in summative vector
        if p1 > p0:
            return 1
        else:
            return 0

    # ================== METHOD TO TEST BAYES MODEL AGAINST NEW DATA =================
    def test_naïve_bayes(self):
        list_of_posts, list_of_classes = self.load_data_set()
        list_of_vocab_words = self.create_vocab_list(list_of_posts)
        training_matrix = []

        # Creates training matrix with word set vectors with which to produce conditional probabilities
        for post_in_document in list_of_posts:
            training_matrix.append(self.convert_bag_of_words_to_vector(list_of_vocab_words, post_in_document))
        
        # Produces conditional and relative probabilities from training data
        p0_vector, p1_vector, p_abusive = self.naïve_bayes_trainer(np.array(training_matrix), np.array(list_of_classes))
        
        # First test entry: expected resultant value is 0 indicating non-abusive terminology
        test_entry = ["love", "my", "dalmation"]
        current_document = np.array(self.convert_bag_of_words_to_vector(list_of_vocab_words, test_entry))
        print("{} CLASSIFIED AS {}".format(test_entry, self.classify_naïve_bayes(current_document, p0_vector, p1_vector, p_abusive)))

        # Second test entry: expected resultant value is 1 indicating abusive terminology
        test_entry = ["stupid", "garbage"]
        current_document = np.array(self.convert_bag_of_words_to_vector(list_of_vocab_words, test_entry))
        print("{} CLASSIFIED AS {}".format(test_entry, self.classify_naïve_bayes(current_document, p0_vector, p1_vector, p_abusive)))
        return

    # ==================== METHOD TO CREATE WORD SET FROM DATASET ====================
    def create_vocab_list(self, dataset):
        vocab_set = set([])

        # Creates set union from dataset
        for document in dataset:
            vocab_set = vocab_set | set(document)

        # print("LIST OF VOCABULARY WORDS IS: {}\n".format(list(vocab_set)))
        return list(vocab_set)

    # =================== METHOD TO CONVERT WORD SET TO WORD VECTOR ==================
    def convert_bag_of_words_to_vector(self, vocab_list, input_set):
        return_vector = [0] * len(vocab_list)

        # Creates vector of unique words from list of vocabulary data
        for word in input_set:
            if word in vocab_list:
                return_vector[vocab_list.index(word)] += 1
            else:
                print("The word '{}' is not in my vocabulary. ".format(word))
        
        # print("RETURN VECTOR IS: {}\n".format(return_vector))
        return return_vector

    # ================ METHOD TO PARSE TEXT FROM STRING USING REGEXES ================
    def text_parser(self, long_string):
        list_of_tokens = re.split(r"\W+", long_string)
        return [token.lower() for token in list_of_tokens if len(token) > 2]

    # ===================== METHOD TO CHECK FOR SPAM-MATCHING DATA ===================
    # TODO: FIX THIS; IT'S BROKEN!!!!!!
    # EXPECTED: Error is minimal, but non-zero
    # ACTUAL: Error is zero for every test
    def check_for_spam(self):
        document_list = []
        class_list = []
        full_text = []

        # Iterates through each email bank and add each parsed email text to all lists (document, full-text, class)
        for iterator in range(1, 26):
            # Adds parsed email text from spam emails
            word_list = self.text_parser(open("email/spam/{}.txt".format(iterator), encoding="ISO-8859-1").read())
            document_list.append(word_list)
            full_text.extend(word_list)
            class_list.append(1)
            
            # Adds parsed email text from ham emails
            word_list = self.text_parser(open("email/ham/{}.txt".format(iterator), encoding="ISO-8859-1").read())
            document_list.append(word_list)
            full_text.extend(word_list)
            class_list.append(0)
        
        # Creates local vocabulary list from email tokens
        vocab_list = self.create_vocab_list(document_list)
        training_set = range(50)
        test_set = []

        # Creates test set of ten random emails from uniform distribution
        for _ in range(10):
            random_index = int(np.random.uniform(0, len(training_set)))
            test_set.append(training_set[random_index])
            del(list(training_set)[random_index])
        
        training_matrix = []
        training_classes = []

        # Creates training dataset and class labels from document list
        for document_index in training_set:
            training_matrix.append(self.convert_bag_of_words_to_vector(vocab_list, document_list[document_index]))
            training_classes.append(class_list[document_index])

        # Creates vectors for initial conditional probabilities and spam probability from training data
        p0_vector, p1_vector, p_spam = self.naïve_bayes_trainer(np.array(training_matrix), np.array(training_classes))
        # print("P0 VECTOR IS: {}\n".format(p0_vector))
        # print("P1 VECTOR IS: {}\n".format(p1_vector))
        # print("P SPAM IS: {}\n".format(p_spam))
        error_count = 0.0

        # Produces vector from test word data, then tests word vector against Bayes classifier and tracks error
        for document_index in test_set:
            word_vector = self.convert_bag_of_words_to_vector(vocab_list, document_list[document_index])
            # print("WORD VECTOR IS: {}\n".format(word_vector))
            # print(self.classify_naïve_bayes(np.array(word_vector), p0_vector, p1_vector, p_spam))
            # print(class_list[document_index])

            if self.classify_naïve_bayes(np.array(word_vector), p0_vector, p1_vector, p_spam) != class_list[document_index]:
                error_count += 1.0
                print("\nCLASSIFICATION ERROR: {}\n".format(document_list[document_index]))

        # print("ERROR COUNT IS: {}\n".format(error_count))
        # print("TEST SET IS: {}\n".format(test_set))
        print("\nTHE ERROR RATE IS: {}\n".format(float(error_count) / len(test_set)))
        return

    # =============== METHOD TO CREATE SORTED DICTOGRAM FROM VOCAB DATA ==============
    def calculate_probability_distribution(self, vocab_list, full_text):
        histogram = {}

        for token in vocab_list:
            histogram[token] = full_text.count(token)

        sorted_histogram = sorted(histogram.items(), key = op.itemgetter(1), reverse = True)
        
        print("SORTED PROBABILITY HISTOGRAM DISTRIBUTION: {}\n".format(sorted_histogram[:30]))
        return sorted_histogram[:30]

    # ========= METHOD TO TEST LOCAL WORD FREQUENCIES FROM RANDOMIZED EMAILS =========
    def test_local_words(self, feed1, feed0):
        document_list = []
        class_list = []
        full_text = []
        minimum_feed_length = min(len(feed1["entries"]), len(feed0["entries"]))

        # Creates word lists from two RSS feeds and adds word data to each list (document, full-text, class)
        for iterator in range(minimum_feed_length):
            word_list = self.text_parser(feed1["entries"][iterator]["summary"])
            document_list.append(word_list)
            full_text.extend(word_list)
            class_list.append(1)

            word_list = self.text_parser(feed0["entries"][iterator]["summary"])
            document_list.append(word_list)
            full_text.extend(word_list)
            class_list.append(0)
        
        # Creates top 30 words and vocab list data from document list
        vocab_list = self.create_vocab_list(document_list)
        top_thirty_words = self.calculate_probability_distribution(vocab_list, full_text)

        # Checks existence of top thirty words in vocab list and if so, decrements from vocab list
        for word_pair in top_thirty_words:
            if word_pair[0] in vocab_list:
                vocab_list.remove(word_pair[0])
    
        # Creates training set from input feeds
        training_set = range(2 * minimum_feed_length)
        test_set = []

        # Creates test set of ten random emails from uniform distribution
        for _ in range(20):
            random_index = int(np.random.uniform(0, len(training_set)))
            test_set.append(training_set[random_index])
            del(list(training_set)[random_index])

        training_matrix = []
        training_classes = []

        # Creates training matrix from document word data and class vectors
        for document_index in training_set:
            training_matrix.append(self.convert_bag_of_words_to_vector(vocab_list, document_list[document_index]))
            training_classes.append(class_list[document_index])

        # Creates vectors for initial conditional probabilities and spam probability from training data
        p0_vector, p1_vector, p_spam = self.naïve_bayes_trainer(np.array(training_matrix), np.array(training_classes))
        error_count = 0.0

        # Produces vector from test word data, then tests word vector against Bayes classifier and tracks error
        for document_index in test_set:
            word_vector = self.convert_bag_of_words_to_vector(vocab_list, document_list[document_index])

            if self.classify_naïve_bayes(np.array(word_vector), p0_vector, p1_vector, p_spam) != class_list[document_index]:
                error_count += 1.0

        # Returns local vocabulary list, local error rate, and initial conditional vector probabilities
        print("THE ERROR RATE IS: {}\n".format(float(error_count) / len(test_set)))
        # print("LOCAL VOCABULARY LIST IS: {}\nPROBABILITY VECTOR FOR NORMAL WORDS IS: {}\nPROBABILITY VECTOR FOR TARGET WORDS IS: {}\n".format(vocab_list, p0_vector, p1_vector))
        return vocab_list, p0_vector, p1_vector


# ====================================================================================
# ================================= MAIN RUN FUNCTION ================================
# ====================================================================================


def main():
    # Track starting time of program
    t0 = t()

    # Initialize class instance of the naïve Bayes classifier algorithm
    bayes = Naïve_Bayes_Classifier_Algorithm()

    # Testing Bayes classifier against training data
    """
    bayes.test_naïve_bayes()
    """

    # Testing spam test method
    # TODO: Is currently broken; error is consistently zero. Must fix!
    """ 
    bayes.check_for_spam()
    """

    # Testing RSS parsing Bayesian classifier
    ny = fp.parse("https://newyork.craigslist.org/stp/index.rss")
    sf = fp.parse("https://sfbay.craigslist.org/stp/index.rss")
    vocab_list, p_sf, p_ny = bayes.test_local_words(ny, sf)

    # Side example for testing regex flexibility for returning token count
    """
    sentence = "This book is the best book on Python or M.L. that I have ever laid my eyes upon."
    split_sentence = sentence.split()
    regex = re.compile("\\W*")
    list_of_tokens = regex.split(sentence)
    non_empty_tokens = [token.lower() for token in list_of_tokens if len(token) > 0]
    print(non_empty_tokens)
    """

    # Track ending time of program and determine overall program runtime
    t1 = t()
    delta = (t1 - t0) * 1000

    print("Real program runtime is {0:.4g} milliseconds.\n".format(delta))
    return

if __name__ == "__main__":
    main()