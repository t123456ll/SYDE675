

# -*- coding: utf-8 -*-

import numpy as np
from math import log
import matplotlib.pyplot as plt
import random
import operator
import urllib
import math
import pprint

def shuffleDataSet(dataSet):
    indices = np.arange(dataSet.shape[0])
    np.random.shuffle(indices)
    dataSet = dataSet[indices]
    return dataSet

def loadDataSet(url, labels):
    raw_data = urllib.urlopen(url)
    data_set = np.loadtxt(raw_data, delimiter=",")
    dataSet = np.delete(data_set, 0, axis=1) # del the id
    dataSet = shuffleDataSet(dataSet) # shuffle the dataSet
    len_data = len(dataSet)
    len_test = int(math.ceil(len_data / 10))
    len_train = int(math.ceil(len_data / 10 * 9 * 0.8))
    len_valid = int(math.ceil(len_data / 10 * 9 * 0.2))
    test_data = dataSet[:len_test].tolist()
    train_data = dataSet[len_test: len_test+len_train].tolist()
    valid_data = dataSet[len_test+len_train: len_test+len_valid+len_train].tolist()
    dataSet = dataSet.tolist()

    return dataSet, train_data, valid_data, test_data, labels

def ent(data):
    feat = {}
    for feature in data:
        curlabel = feature[-1]
        if curlabel not in feat:
            feat[curlabel] = 0
        feat[curlabel] += 1
    s = 0.0
    num = len(data)
    for it in feat:
        p = feat[it] * 1.0 / num
        s -= p * log(p, 2)
    return s


def remove_feature(data, i, value, flag):
    newdata = []
    for row in data:
        if flag == True:
            if row[i] < value:
                temp = row[:i]
                temp.extend(row[i + 1:])
                newdata.append(temp)
        else:
            if row[i] >= value:
                temp = row[:i]
                temp.extend(row[i + 1:])
                newdata.append(temp)
    #    print('newdata = ',newdata)
    return newdata


def choosebest(data):
    m = len(data)
    maxgain = 0.0
    bestfeature = -1
    bestpoint = -1.0
    n = len(data[0]) - 1
    S = ent(data)
    for i in range(n):
        curfeature = []
        for j in range(m):
            curfeature.append(data[j][i])
        curfeature.sort()
        maxgain = 0.0
        point_id = -1
        for j in range(m - 1):
            point = float(curfeature[j + 1] + curfeature[j]) / 2
            Set = [[it for it in curfeature if it < point], [it for it in curfeature if it > point]]
            p1 = float(len(Set[0])) / m
            p2 = float(len(Set[1])) / m
            split = 0
            if p1 != 0:
                split -= p1 * log(p1, 2)
            if p2 != 0:
                split -= p2 * log(p2, 2)
            if split == 0:
                continue
            gain = (S - p1 * ent(remove_feature(data, i, point, True)) - p2 * ent(
                remove_feature(data, i, point, False))) / split
            if gain > maxgain:
                maxgain = gain
                bestfeature = i
                bestpoint = point
    return bestfeature, bestpoint


def classify(tree, feature, value):
    if type(tree).__name__ != 'dict':
        return tree
    root = list(tree.keys())[0]
    sons = tree[root]
    i = feature.index(root)
    if value[i] >= list(sons.keys())[1]:
        return classify(sons[list(sons.keys())[1]], feature, value)
    else:
        return classify(sons[list(sons.keys())[0]], feature, value)


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(data, feature, validData, mode=False):
    curlabel = [it[-1] for it in data]
    if curlabel.count(curlabel[0]) == len(curlabel):
        return curlabel[0]

    if len(curlabel) <= int(log(n,2)):
        return majorityCnt(curlabel)
    if len(data[0]) == 1:
        return majorityCnt(curlabel)
    i, point = choosebest(data)
    bestfeature = feature[i]
    tree = {bestfeature: {}}
    del feature[i]
    newfeature = feature[:]
    newdata = remove_feature(data, i, point, True)
    tree[bestfeature][0] = createTree(newdata, newfeature, validData, mode=False)
    newdata = remove_feature(data, i, point, False)
    newfeature = feature[:]
    tree[bestfeature][point] = createTree(newdata, newfeature, validData, mode=False)
    return tree


def dfs(tree, deep, sample):
    if (type(tree) != sample):
        return deep
    cnt = 0
    for key in tree.keys():
        cnt = max(cnt, dfs(tree[key], deep + 1, sample))
    return cnt



def testing(myTree,data_test,labels):
    error=0.0
    for i in range(len(data_test)):
        testVec = data_test[i]
        predLabel = classify(myTree,labels,testVec)
        tureLabel = data_test[i][-1]
        if predLabel != tureLabel:
            error+=1
    accuracy = 1-(error/len(data_test))
    print ('myTree error', error)
    print ('myTree accuracy', accuracy)
    return float(error)

def main():
    dataSet, trainData, validData, testData, labels = loadDataSet("glass.data",
                                                                  ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe",
                                                                   "Type"])
    data_full = trainData[:]
    labels_full = labels[:]
    # minErrorNum = 1000
    tree = createTree(trainData, labels, validData, mode=False)  # , data_full, labels_full, validData, mode="post"
    pprint.pprint(tree)
    testing(tree, testData, labels)
    tree = createTree(trainData, labels, validData, mode=True)
    pprint.pprint(tree)
    testing(tree, testData, labels)


if __name__ == '__main__':
    main()
def classify(inputTree,featLabels,testVec):
    newlist = list()
    for i in inputTree.keys():
        newlist.append(i)
    firstStr = newlist[0]
    secondDict=inputTree[firstStr]
    featIndex=featLabels.index(firstStr)
    secondlist = list()
    for i in secondDict.keys():
        secondlist.append(i)
    for key in secondlist:
        if (("<=" in key) and (testVec[featIndex] <= (float(re.findall(r"\d+\.?\d*",key)[0])))) or\
             ((">" in key) and (testVec[featIndex] > (float(re.findall(r"\d+\.?\d*",key)[0]))))     :
            if type(secondDict[key]).__name__=='dict':
                classLabel=classify(secondDict[key],featLabels,testVec)
            else:classLabel=secondDict[key]
    return classLabel

def testing(myTree,data_test,labels):
    error=0.0
    for i in range(len(data_test)):
        if classify(myTree,labels,data_test[i])!=data_test[i][-1]:
            error+=1
    print ('myTree %d' %error)
    return float(error)