"""
kNN.py (data_projects/machine_learning_in_action/algo_ch01/)
Tutorial program to apply the k-Nearest Neighbor ML Algorithm.

All source code is available at www.manning.com/MachineLearningInAction. 
Credit: MACHINE LEARNING IN ACTION (PETER HARRINGTON)
"""


import numpy as np
import operator as op
from os import listdir as ld
# from matplotlib import pyplot as plt
# from array import array


# Function that creates data set from given arrays and labels
def createDataSet():
    group = np.array([1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1])
    labels = ["A", "A", "B", "B"]
    return group, labels


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


def main():
    group, labels = createDataSet()
    # print(group)
    # print(labels)

    classifier = classify0([0, 0], group, labels, 3)
    # print(classifier)

    datingDataMat, datingLabels = file2matrix("datingTestSet.txt")
    # print(datingDataMat)
    # print(datingLabels[:10])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2])
    # ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
    plt.show()

if __name__ == "__main__":
    main()
