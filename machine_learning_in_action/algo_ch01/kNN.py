"""
kNN.py (data_projects/machine_learning_in_action/algo_ch01/)
Tutorial program to apply the k-Nearest Neighbor ML Algorithm.

All source code is available at www.manning.com/MachineLearningInAction. 
Credit: MACHINE LEARNING IN ACTION (PETER HARRINGTON)
"""


from numpy import *
from matplotlib import pyplot as plt
import operator

# Function that creates data set from given arrays and labels
def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ["A", "A", "B", "B"]
    return group, labels

"""
To run the createDataSet() function in the Python interpreter, write the following:

>>> import kNN                              # NOTE: Imports main file in the interpreter
>>> group, labels = kNN.createDataSet()     # NOTE: Creates data sets from given data
>>> group                                   # NOTE: Displays created data set
>>> labels                                  # NOTE: Displays created labels
"""

# Function that classifies inputted array based on control dataset
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]

    # Distance calculation
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances ** 0.5

    sortedDistIndices = distances.argsort()
    classCount = {}

    # Voting with lowest k distances
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    # Sort dictionary
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

"""
To run the classify0() function in the Python interpreter, write the following:

>>> import kNN                                  # NOTE: Imports main file in the interpreter
>>> group, labels = kNN.createDataSet()         # NOTE: Creates data set from given data
>>> kNN.classify0(inX, dataSet, labels, k)      # NOTE: Runs classifier function on dataset
        - inX:      [0, 0] or other 2x1 array between [0, 0] and [1, 1].
        - dataSet:  group
        - labels:   labels
        - k:        3 
"""

# Function that converts data from text file into dataset and relative labels 
def file2matrix(filename):
    fr = open(filename)

    # Get number of lines in file
    numberOfLines = len(fr.readlines())

    # Create NumPy matrix to return
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    fr = open(filename)

    # Parse line to a list
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split("\t")
        returnMat[index, :] = listFromLine[0:3]

        classLabelVector.append(str(listFromLine[-1]))
        index += 1

    return returnMat, classLabelVector

"""
To run the file2matrix() function in the Python interpreter, write the following:

>>> import kNN                                                          # NOTE: Imports main file in the interpreter
>>> datingDataMat, datingLabels = kNN.file2matrix("datingTestSet.txt")  # NOTE: Runs file2matrix conversion 
>>> datingDataMat                                                       # NOTE: Displays created data set
>>> datingLabels                                                        # NOTE: Displays created labels

>>> from matplotlib import pyplot as plt                    # NOTE: Installs plotting capabilities
>>> from array import array                                 # NOTE: Installs array constructors
>>> fig = plt.figure()                                      # NOTE: Initializes plot
>>> ax = fig.add_subplot(111)                               # NOTE: Creates unit plot in first quadrant
>>> ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2])    # NOTE: Generates scatter data across plot
>>> plt.show()                                              # NOTE: Renders plot
"""