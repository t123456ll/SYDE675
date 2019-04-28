import math
import numpy as np
import operator
import urllib
import re
import copy

# download dataset
url1 = "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"
url2 = "https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data"
minErrorNum = 1000


def ShuffleDataSet(dataSet):
    indices = np.arange(dataSet.shape[0])
    np.random.shuffle(indices)
    dataSet = dataSet[indices]
    return dataSet

def LoadDataSet(mode, url):
    if mode == 1: # designed for discrete
        recordList = []
        fp = open(url, "rb")
        content = fp.read()
        fp.close()
        rowList = content.splitlines()
        recordList = [row.split(",") for row in rowList if row.strip()]
        dataSet = np.array(recordList)

    if mode == 0: # designed for continue
        raw_data = urllib.urlopen(url)
        data_set = np.loadtxt(raw_data, delimiter=",")
        dataSet = np.delete(data_set, 0, axis=1) # del the id

    dataSet = ShuffleDataSet(dataSet)  # shuffle the dataSet
    return dataSet

def GetTest(dataSet, it):
    len_data = len(dataSet)
    len_test = int(math.ceil(len_data / 10))
    start = it * len_test
    finish = (it+1) * len_test
    test_data = dataSet[start: finish].tolist()
    rest1 = dataSet[:start]
    rest2 = dataSet[finish:]
    rest_data = np.vstack((rest1,rest2))
    return rest_data, test_data

def GetValid(dataSet):
    dataSet = ShuffleDataSet(dataSet)  # shuffle the dataSet
    len_data = len(dataSet)
    len_valid = int(round(len_data * 0.2))
    valid_data = dataSet[:len_valid].tolist()
    train_data = dataSet[len_valid:].tolist()

    return train_data, valid_data


def ClassCount(dataSet): # record the appear time of each class
    label_count={}
    for one in dataSet:
        if one[-1] not in label_count.keys():
            label_count[one[-1]]=0
        label_count[one[-1]]+=1
    return label_count

def CalcShannonEntropy(data): # calculate the shannon entropy
    label_count=ClassCount(data)
    num_entries=len(data)
    Entropy=0.0
    for i in label_count:
        prob=float(label_count[i])/num_entries
        Entropy-=prob*math.log(prob,2)
    return Entropy

def MajorityClass(dataSet):  # return the label which appear most
    label_count=ClassCount(dataSet)
    sorted_label_count=sorted(label_count.items(),key=operator.itemgetter(1),reverse=True)
    return sorted_label_count[0][0]

def SplitDiscreteDataSet(dataSet,i,value):
    sub_data_set=[]
    for item in dataSet:
        if item[i]==value:
            reduce_data=item[:i]
            reduce_data.extend(item[i+1:])
            sub_data_set.append(reduce_data)
    return sub_data_set

def SplitContinuousDataSet(dataSet,i,value,direction):
    subDataSet=[]
    for one in dataSet:
        if direction==0:
            if one[i]>value:
                reduceData=one[:i]
                reduceData.extend(one[i+1:])
                subDataSet.append(reduceData)
        if direction==1:
            if one[i]<=value:
                reduceData=one[:i]
                reduceData.extend(one[i+1:])
                subDataSet.append(reduceData)
    return subDataSet

def ChooseSplitList(featureVList, dataSet):
    typeList = [data[-1] for data in dataSet]
    sortedIndx = np.array(featureVList).argsort()[::-1]
    sortedFeatV = np.array(featureVList)[sortedIndx]
    sortedType = np.array(typeList)[sortedIndx]

    cur = sortedType[0]
    count = 0
    splitNumList = []
    splitValList = []
    for i in sortedType:
        if (i == cur):
            count += 1
            cur = i
        elif (i != cur):
            left = sortedFeatV[count-1]
            right = sortedFeatV[count]
            splitValList.append((left+right)/2)
            splitNumList.append(count)
            count += 1
            cur = i

    return splitValList


def ChooseBestFeat(dataSet,labels):
    baseEntropy=CalcShannonEntropy(dataSet)
    bestFeat=0
    baseGainRatio=-1
    numFeats=len(dataSet[0])-1
    bestSplit=-1000
    bestSplitDic={}
    # print('dataSet[0]:' + str(dataSet[0]))
    for i in range(numFeats):
        featVals=[example[i] for example in dataSet]
        # print('chooseBestFeat:'+str(i))
        featType = type(featVals[0]).__name__
        if featType =='float' or featType =='int':
            splitList = ChooseSplitList(featVals, dataSet)
            for j in splitList:
            # for j in range(len(splitList)):
                newEntropy=0.0
                gainRatio=0.0
                splitInfo=0.0
                value = j
                subDataSet0 = SplitContinuousDataSet(dataSet,i,value,0)
                subDataSet1 = SplitContinuousDataSet(dataSet,i,value,1)
                # print ("dataset : ", dataSet)
                prob0 = float(len(subDataSet0))/len(dataSet)
                newEntropy += prob0*CalcShannonEntropy(subDataSet0)
                prob1 = float(len(subDataSet1))/len(dataSet)
                newEntropy += prob1*CalcShannonEntropy(subDataSet1)
                if (prob0 != 0):
                    splitInfo -= prob0*math.log(prob0,2)
                if (prob1 != 0):
                    splitInfo -= prob1*math.log(prob1,2)
                if (splitInfo == 0):
                    continue
                gainRatio=float(baseEntropy-newEntropy)/splitInfo
                # print('IVa '+str(j)+':'+str(splitInfo))
                if gainRatio>baseGainRatio:
                    baseGainRatio=gainRatio
                    bestSplit=j
                    bestFeat=i
            bestSplitDic[labels[i]]=bestSplit
        else:
            uniqueFeatVals=set(featVals)
            GainRatio=0.0
            splitInfo=0.00001
            newEntropy=0.0
            for value in uniqueFeatVals:
                subDataSet=SplitDiscreteDataSet(dataSet,i,value)
                prob=float(len(subDataSet))/len(dataSet)
                splitInfo-=prob*math.log(prob,2)
                newEntropy+=prob*CalcShannonEntropy(subDataSet)
            gainRatio=float(baseEntropy-newEntropy)/splitInfo
            if gainRatio > baseGainRatio:
                bestFeat = i
                baseGainRatio = gainRatio

    dataType = type(dataSet[0][bestFeat]).__name__
    if dataType=='float' or dataType=='int':
        bestFeatValue=bestSplitDic[labels[bestFeat]]
        # print('baseEntropy ' + str(j) + ':' + str(baseEntropy))
        # print('splitinfo ' + str(j) + ':' + str(splitInfo))
        # print('gainratio ' + str(j) + ':' + str(gainRatio))
        # print('infogain ' + str(j) + ':' + str(baseEntropy - newEntropy))
    else :  # dataType=='str':
        bestFeatValue=labels[bestFeat]
        # print('baseEntropy ' + str(bestFeat) + ':' + str(baseEntropy))
        # print('splitinfo ' + str(bestFeat) + ':' + str(splitInfo))
        # print('gainratio ' + str(bestFeat) + ':' + str(gainRatio))
        # print('infogain ' + str(bestFeat) + ':' + str(baseEntropy - newEntropy))
    return bestFeat,bestFeatValue

def MajorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
    return max(classCount, key=classCount.get)

def testing_feat(feat, train_data, test_data, labels):
    class_list = [example[-1] for example in train_data]
    bestFeatIndex = labels.index(feat)
    train_data = [example[bestFeatIndex] for example in train_data]
    test_data = [(example[bestFeatIndex], example[-1]) for example in test_data]
    all_feat = set(train_data)
    error = 0.0
    for value in all_feat:
        class_feat = [class_list[i] for i in range(len(class_list)) if train_data[i] == value]
        major = MajorityCnt(class_feat)
        for data in test_data:
            if data[0] == value and data[1] != major:
                error += 1.0
    # print 'myTree %d' % error
    return error

def testingMajor(major, data_test):
    error = 0.0
    for i in range(len(data_test)):
        if major != data_test[i][-1]:
            error += 1
    # print 'major %d' % error
    return float(error)



def CreateTree(dataSet, labels, validData, mode):   # using mode to choose whether using pruning, using minErrorNum to record the best error num
    global minErrorNum
    lenData = len(dataSet[0])
    classList=[example[-1] for example in dataSet]
    if len(set(classList)) == 1: # have only one type
        return classList[0]
    if lenData == 1 : # no feature can be used to classify
        return MajorityClass(dataSet)
    if lenData == 2: # in order to avoid the same value lead to [] (empty subset)
        checkFeat = [example[0] for example in dataSet]
        if len(set(checkFeat)) == 1:
            return MajorityClass(dataSet)
    Entropy = CalcShannonEntropy(dataSet)
    labels_copy = copy.deepcopy(labels)  # using the deep copy to avoid the effect on original data
    bestFeat,bestFeatValue=ChooseBestFeat(dataSet,labels)
    # print('bestFeat:'+str(bestFeat)+'--'+str(labels[bestFeat])+', bestFeatVal:'+str(bestFeatValue))
    myTree={labels[bestFeat]:{}}
    subLabels = labels[:bestFeat]
    subLabels.extend(labels[bestFeat+1:]) # del the used feature
    # print('subLabels:'+str(subLabels))

    dataType = type(dataSet[0][bestFeat]).__name__
    if dataType=='int' or dataType=='float':
        value=bestFeatValue
        greaterDataSet=SplitContinuousDataSet(dataSet,bestFeat,value,0)
        smallerDataSet=SplitContinuousDataSet(dataSet,bestFeat,value,1)
        # print('greaterDataset:' + str(greaterDataSet))
        # print('smallerDataSet:' + str(smallerDataSet))
        # print('== ' * len(dataSet[0]))
        value = round(value, 5)
        if len(greaterDataSet) != 0:
            if len(smallerDataSet) != 0:
                myTree[labels[bestFeat]]['>' + str(value)] = CreateTree(greaterDataSet, subLabels, validData, mode)
                myTree[labels[bestFeat]]['<=' + str(value)] = CreateTree(smallerDataSet, subLabels, validData, mode)
            else: return MajorityClass(greaterDataSet)
        else: return MajorityClass(smallerDataSet)
        # print(myTree)
        # print('== ' * len(dataSet[0]))
    if dataType=='str':
        featVals = [example[bestFeat] for example in dataSet]
        uniqueVals = set(featVals) # to get the rest featurevalue
        # print('uniqueVals:' + str(uniqueVals))
        for value in uniqueVals: # go through all the rest of branches
            reduceDataSet=SplitDiscreteDataSet(dataSet,bestFeat,value)
            # print('value: '+str(value))
            # print('reduceDataSet:'+str(reduceDataSet))
            myTree[labels[bestFeat]][value]=CreateTree(reduceDataSet,subLabels, validData, mode)

    # postpruning
    if mode == True:
        # print ("----------Pruning------------")
        testingResult = testing(myTree, validData, labels_copy)
        # print(classList)
        major = MajorityCnt(classList)
        # print("major: ", major)
        testMajorResult = testingMajor(major, validData)
        # print("testingResult: ", testingResult)
        # print("testMajorResult: ", testMajorResult)
        # pprint.pprint(myTree)
        # if (testMajorResult < testingResult):
        if (testMajorResult < minErrorNum) and (testMajorResult < testingResult):
            minErrorNum = testMajorResult # if the node is not good, using the major
            # print ("*********** pruning this branch ************")
            return MajorityCnt(classList)

    return myTree

def classify(inputTree,featLabels,testVec):
    newlist = list()
    for i in inputTree.keys():
        newlist.append(i)
    firstStr = newlist[0]
    secondDict=inputTree[firstStr]
    featIndex=featLabels.index(firstStr)
    secondlist = list()
    classLabel = 'None'
    for i in secondDict.keys():
        secondlist.append(i)
    for key in secondlist:
        if (("<=" in key) and (testVec[featIndex] <= (float(re.findall(r"\d+\.?\d*",key)[0])))) or\
             ((">" in key) and (testVec[featIndex] > (float(re.findall(r"\d+\.?\d*",key)[0])))):
            if type(secondDict[key]).__name__=='dict':
                classLabel=classify(secondDict[key],featLabels,testVec)
            else:classLabel=secondDict[key]

        elif (('x' in key) or ('o' in key) or ('b' in key)):
            if testVec[featIndex] == key:  # if the key in branch matches the testVec, go to the next
                dictType = type(secondDict[key]).__name__
                if dictType == 'dict':  # the type of dict means the branch has a child tree
                    classLabel = classify(secondDict[key], featLabels, testVec)  # do recursion
                else:classLabel = secondDict[key]  # if it is leaf, return the class label
            # else :
            #     print ("secondlist: ", secondlist)
            #     print ("testVec: ", testVec[featIndex])
    return classLabel



def testing(myTree,data_test,labels):
    error=0.0
    for i in range(len(data_test)):
        testVec = data_test[i]
        predLabel = classify(myTree,labels,testVec)
        tureLabel = data_test[i][-1]
        if predLabel != tureLabel:
            error+=1
    accuracy = 1-(error/len(data_test))
    # print ('myTree error', error)
    # print ('myTree accuracy', accuracy)
    return float(error)

def getAccuracy(myTree,data_test,labels):
    error = 0.0
    for i in range(len(data_test)):
        testVec = data_test[i]
        predLabel = classify(myTree, labels, testVec)
        tureLabel = data_test[i][-1]
        if predLabel != tureLabel:
            error += 1
    accuracy = 1 - (error / len(data_test))
    return float(accuracy)



if __name__ == '__main__':

    label0 = ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Type"]
    label1 = ["top-left", "top-middle", "top-right", "middle-left", "middle-middle", "middle-right", "bottom-left",
              "bottom-middle", "bottom-right", "Class"]
    sumAccuracy0 = 0
    sumAccuracy1 = 0
    accuracyArray0 = []
    accuracyArray1 = []
    for i in range(10): # ten times
        dataSet0 = LoadDataSet(0, "glass.data")
        dataSet1 = LoadDataSet(1, "tic-tac-toe.data")

        for j in range(10): # ten fold
            rest0, test0 = GetTest(dataSet0, j)
            rest1, test1 = GetTest(dataSet1, j)
            train0, valid0 = GetValid(rest0)
            train1, valid1 = GetValid(rest1)
            tree0 = CreateTree(train0, label0, valid0, mode=False)
            tree1 = CreateTree(train1, label1, valid1, mode=False)
            accuracyArray0.append(getAccuracy(tree0, test0, label0))
            accuracyArray1.append(getAccuracy(tree1, test1, label1))
            sumAccuracy0 += getAccuracy(tree0, test0, label0)
            sumAccuracy1 += getAccuracy(tree1, test1, label1)
            # print ("glass ", i, "time", j, "fold: ", sumAccuracy0)
            # print ("tic-tac-toe ", i, "time", j, "fold: ", sumAccuracy1)

    var0 = np.var(accuracyArray0)
    var1 = np.var(accuracyArray1)
    print ("glass accuracy (after): ", sumAccuracy0 / 100)
    print ("tic-tac-toe accuracy (after): ", sumAccuracy1 / 100)
    print ("glass accuracy variance (after): ", var0)
    print ("tic-tac-toe accuracy variance (after): ", var1)









