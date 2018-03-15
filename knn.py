"""
created on 2018/03/04

KNN: K-nearest neighbors classifier


output: the most popular class label
input: data vector

@author: woshahua
"""

from numpy import *
import operator


def file2matrix(filename):
    fr = open(filename)
    # first read all the lines in a file
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)
    # 3, cause this data only has 3 attributes
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []

    index = 0
    for line in arrayOfLines:
        # clean space
        line = line.strip()
        listFromLine = line.split("\t")
        # how this work ?
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(listFromLine[-1])

        index +=1

    return returnMat, classLabelVector

def createDataSet():

    group = array([[1.0, 1.1],[1.0,1.0], [0,0], [0, 0.1]])
    labels = ["A", "A", "B", "B"]
    return group, labels

def classify0(inX, dataSet, labels, k):

    dataSetSize = dataSet.shape[0]
    # 将单个属性值转化为list, 计算差值list
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5

    # @problem: this argsort may have problem, cause it sort like [2 3 1 0], but it should be [2 3 0 1]
    sortedDistIndicies = distances.argsort()

    classCount = {}
    for i in range(k):
        voteIlabel =labels[sortedDistIndicies[i]]
        # @@@ how this work ?
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
        # @@@ good code to learn
        # default sort is small -> big, reverse big -> small
        sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse=True)

    # return the best class
    return sortedClassCount[0][0]

# normalization: to make the importance of each attribute to be fairly computed
def autoNorm(dataSet):

    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))

    m = dataSet.shape[0]
    # normalized to [0, 1]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    """
    compute is like:

    first here is a 4 x 3 vector, we want each vector to minus a value A_value.
    then we generate a 4 x 1 vector with A_value in each element.

    then calculate 4 x 3 vector - 4 x 1 vector
    """
    return normDataSet, ranges, minVals

def datingClassTest():
    hoRatio = 0.10
    # 1. read data
    datingDataMat, datingLabels = file2matrix("datingTestSet.txt")
    # 2. normalize data
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # m: number of data
    m = normMat.shape[0]
    # numTestVecs: number of test patterns
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:], datingLabels[numTestVecs:m],3)
        print ("the classifier answer is: {}, the real answer is: {}".format(classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0

    print ("the error rate is: {}".format(errorCount/float(numTestVecs)))
    print (errorCount)

# execution
datingClassTest()
