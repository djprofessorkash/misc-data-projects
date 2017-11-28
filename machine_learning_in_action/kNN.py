# Tutorial program to apply the k-Nearest Neighbor ML Algorithm
# All source code is available at www.manning.com/MachineLearningInAction 
#
# Credit: MACHINE LEARNING IN ACTION (PETER HARRINGTON)


from numpy import *
import operator

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ["A", "A", "B", "B"]
    return group, labels

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

        classLabelVector.append(int(listFromLine[-1]))
        index += 1

    return returnMat, classLabelVector