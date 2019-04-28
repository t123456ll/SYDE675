import math
import numpy as np
import operator
import urllib
import re
import copy
import matplotlib.pyplot as plt

# download dataset
url1 = "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"
url2 = "https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data"
minErrorNum = 1000
def addNoise(noisePercent, dataSet, mode):
    classList = [example[-1] for example in dataSet]
    classList = list(set(classList))
    len_data = len(dataSet)
    len_noise = int(noisePercent * len_data)

    if mode == 0: # Contradictory examples
        dataSet = np.delete(dataSet, np.s_[:len_noise], 0)
        copyData = copy.deepcopy(dataSet)
        noise = copyData[:len_noise]
        for one in noise:
            if one[-1] in classList:
                p = (classList.index(one[-1])+1)%len(classList) # replace the data label with next label
                one[-1] = classList[p]
            else: continue
        dataSet = np.vstack((noise, dataSet))

    if mode == 1: # Misclassified examples
        noise = dataSet[:len_noise]
        for one in noise:
            if one[-1] in classList:
                p = (classList.index(one[-1]) + 1)%len(classList) # replace the data label with next label
                one[-1] = classList[p]
            else: continue

    dataSet = shuffleDataSet(dataSet)
    return dataSet.tolist()




def shuffleDataSet(dataSet):
    indices = np.arange(dataSet.shape[0])
    np.random.shuffle(indices)
    dataSet = dataSet[indices]
    return dataSet

def loadDataSet(mode, url):
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

    dataSet = shuffleDataSet(dataSet)  # shuffle the dataSet

    return dataSet

def getTest(dataSet, it):
    len_data = len(dataSet)
    len_test = int(math.ceil(len_data / 10))
    start = it * len_test
    finish = (it+1) * len_test
    test_data = dataSet[start: finish].tolist()
    rest1 = dataSet[:start]
    rest2 = dataSet[finish:]
    rest_data = np.vstack((rest1,rest2))

    return rest_data, test_data

def getValid(dataSet):
    dataSet = shuffleDataSet(dataSet)  # shuffle the dataSet
    len_data = len(dataSet)
    len_valid = int(round(len_data * 0.2))
    valid_data = dataSet[:len_valid].tolist()
    train_data = dataSet[len_valid:]

    return train_data, valid_data


def classCount(dataSet): # record the appear time of each class
    labelCount={}
    for one in dataSet:
        if one[-1] not in labelCount.keys():
            labelCount[one[-1]]=0
        labelCount[one[-1]]+=1
    return labelCount

def calcShannonEntropy(dataSet): # calculate the shannon entropy
    labelCount=classCount(dataSet)
    numEntries=len(dataSet)
    Entropy=0.0
    for i in labelCount:
        prob=float(labelCount[i])/numEntries
        Entropy-=prob*math.log(prob,2)
    return Entropy

def majorityClass(dataSet):
    labelCount=classCount(dataSet)
    sortedLabelCount=sorted(labelCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedLabelCount[0][0]

def splitDiscreteDataSet(dataSet,i,value):
    subDataSet=[]
    for one in dataSet:
        if one[i]==value:
            reduceData=one[:i]
            reduceData.extend(one[i+1:])
            subDataSet.append(reduceData)
    return subDataSet

def splitContinuousDataSet(dataSet,i,value,direction):
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

def chooseSplitList(featureVList, dataSet):
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


def chooseBestFeat(dataSet,labels):
    baseEntropy=calcShannonEntropy(dataSet)
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
            splitList = chooseSplitList(featVals, dataSet)
            for j in splitList:
            # for j in range(len(splitList)):
                newEntropy=0.0
                gainRatio=0.0
                splitInfo=0.0
                value = j
                subDataSet0 = splitContinuousDataSet(dataSet,i,value,0)
                subDataSet1 = splitContinuousDataSet(dataSet,i,value,1)
                # print ("dataset : ", dataSet)
                prob0 = float(len(subDataSet0))/len(dataSet)
                newEntropy += prob0*calcShannonEntropy(subDataSet0)
                prob1 = float(len(subDataSet1))/len(dataSet)
                newEntropy += prob1*calcShannonEntropy(subDataSet1)
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
                subDataSet=splitDiscreteDataSet(dataSet,i,value)
                prob=float(len(subDataSet))/len(dataSet)
                splitInfo-=prob*math.log(prob,2)
                newEntropy+=prob*calcShannonEntropy(subDataSet)
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
        ##bestFeatValue=labels[bestFeat]+'<='+str(bestSplitValue)
    else :  # dataType=='str':
        bestFeatValue=labels[bestFeat]
        # print('baseEntropy ' + str(bestFeat) + ':' + str(baseEntropy))
        # print('splitinfo ' + str(bestFeat) + ':' + str(splitInfo))
        # print('gainratio ' + str(bestFeat) + ':' + str(gainRatio))
        # print('infogain ' + str(bestFeat) + ':' + str(baseEntropy - newEntropy))
    return bestFeat,bestFeatValue

def majorityCnt(classList):
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
        major = majorityCnt(class_feat)
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



def createTree(dataSet, labels, validData, mode):   # using mode to choose whether using pruning, using minErrorNum to record the best error num
    global minErrorNum
    lenData = len(dataSet[0])
    classList=[example[-1] for example in dataSet]
    if len(set(classList)) == 1: # have only one type
        return classList[0]
    if lenData == 1 : # no feature can be used to classify
        return majorityClass(dataSet)
    if lenData == 2: # in order to avoid the same value lead to [] (empty subset)
        checkFeat = [example[0] for example in dataSet]
        if len(set(checkFeat)) == 1:
            return majorityClass(dataSet)
    Entropy = calcShannonEntropy(dataSet)
    labels_copy = copy.deepcopy(labels)  # using the deep copy to avoid the effect on original data
    bestFeat,bestFeatValue=chooseBestFeat(dataSet,labels)
    # print('bestFeat:'+str(bestFeat)+'--'+str(labels[bestFeat])+', bestFeatVal:'+str(bestFeatValue))
    myTree={labels[bestFeat]:{}}
    subLabels = labels[:bestFeat]
    subLabels.extend(labels[bestFeat+1:]) # del the used feature
    # print('subLabels:'+str(subLabels))

    dataType = type(dataSet[0][bestFeat]).__name__
    if dataType=='int' or dataType=='float':
        value=bestFeatValue
        greaterDataSet=splitContinuousDataSet(dataSet,bestFeat,value,0)
        smallerDataSet=splitContinuousDataSet(dataSet,bestFeat,value,1)
        # print('greaterDataset:' + str(greaterDataSet))
        # print('smallerDataSet:' + str(smallerDataSet))
        # print('== ' * len(dataSet[0]))
        value = round(value, 5)
        if len(greaterDataSet) != 0:
            if len(smallerDataSet) != 0:
                myTree[labels[bestFeat]]['>' + str(value)] = createTree(greaterDataSet, subLabels, validData, mode)
                myTree[labels[bestFeat]]['<=' + str(value)] = createTree(smallerDataSet, subLabels, validData, mode)
            else: return majorityClass(greaterDataSet)
        else: return majorityClass(smallerDataSet)
        # print(myTree)
        # print('== ' * len(dataSet[0]))
    if dataType=='str':
        featVals = [example[bestFeat] for example in dataSet]
        uniqueVals = set(featVals) # to get the rest featurevalue
        # print('uniqueVals:' + str(uniqueVals))
        for value in uniqueVals: # go through all the rest of branches
            reduceDataSet=splitDiscreteDataSet(dataSet,bestFeat,value)
            # print('value: '+str(value))
            # print('reduceDataSet:'+str(reduceDataSet))
            myTree[labels[bestFeat]][value]=createTree(reduceDataSet,subLabels, validData, mode)

    # postpruning
    if mode == True:
        # print ("----------Pruning------------")
        testingResult = testing(myTree, validData, labels_copy)
        # print(classList)
        major = majorityCnt(classList)
        # print("major: ", major)
        testMajorResult = testingMajor(major, validData)
        # print("testingResult: ", testingResult)
        # print("testMajorResult: ", testMajorResult)
        # pprint.pprint(myTree)
        # if (testMajorResult < testingResult):
        if (testMajorResult < minErrorNum) and (testMajorResult < testingResult):
            minErrorNum = testMajorResult # if the node is not good, using the major
            # print ("*********** pruning this branch ************")
            return majorityCnt(classList)

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


    dataSet0 = loadDataSet(0, "glass.data")
    dataSet1 = loadDataSet(1, "tic-tac-toe.data")
    noisePerList = [0, 0.05, 0.1, 0.15]
    glass_contradictory = []
    glass_misclassified = []
    tic_contradictory = []
    tic_misclassified = []

    for noise in noisePerList:
        sumAccuracy00 = 0
        sumAccuracy10 = 0
        sumAccuracy01 = 0
        sumAccuracy11 = 0
        for i in range(10): # ten times

            dataSet0 = shuffleDataSet(np.array(dataSet0))
            dataSet1 = shuffleDataSet(np.array(dataSet1))

            for j in range(10): # ten fold

                rest0, test0 = getTest(dataSet0, j)
                rest1, test1 = getTest(dataSet1, j)
                train0, valid0 = getValid(rest0)
                train1, valid1 = getValid(rest1)

                train00 = addNoise(noise, train0, 0)  # continuous & contradictory examples
                train10 = addNoise(noise, train1, 0)  # discrete & contradictory examples
                train01 = addNoise(noise, train0, 1)  # continuous & misclassified examples
                train11 = addNoise(noise, train1, 1)  # discrete & misclassified examples

                # noise: contradictory examples
                tree00 = createTree(train00, label0, valid0, mode = False)
                tree10 = createTree(train10, label1, valid1, mode = False)

                sumAccuracy00 += getAccuracy(tree00, test0, label0)
                sumAccuracy10 += getAccuracy(tree10, test1, label1)

                # noise: missclassified examples
                tree01 = createTree(train01, label0, valid0, mode=False)
                tree11 = createTree(train11, label1, valid1, mode=False)

                sumAccuracy01 += getAccuracy(tree01, test0, label0)
                sumAccuracy11 += getAccuracy(tree11, test1, label1)

        glass_contradictory.append(sumAccuracy00/100)
        glass_misclassified.append(sumAccuracy01/100)
        tic_contradictory.append(sumAccuracy10/100)
        tic_misclassified.append(sumAccuracy11/100)

        print ("noise percentage: ", noise)
        print ("glass accuracy (contradictory): ", sumAccuracy00/100)
        print ("tic-tac-toe accuracy (contradictory): ", sumAccuracy10/100)
        print ("glass accuracy (misclassified): ", sumAccuracy01/100)
        print ("tic-tac-toe accuracy (misclassified): ", sumAccuracy11/100)
        print ("-----------------------------------------------------")


    # plot the bar

    label_x = ['0%', '5%', '10%', '15%']
    # glass_contradictory = [0.6181, 0.5704, 0.5595, 0.5305]
    # glass_misclassified = [0.6186, 0.5690, 0.5229, 0.5048]
    # tic_contradictory = [0.8372, 0.7554, 0.7042, 0.6355]
    # tic_misclassified = [0.8372, 0.7771, 0.7254, 0.6767]

    x = range(len(glass_contradictory))

    # draw the glass dataset accuracy
    fig1 = plt.figure(0)
    ax = fig1.add_subplot(121)
    glass_bar0 = plt.bar(left=x, height=glass_contradictory, width=0.4, alpha=0.8, color='dodgerblue', label="contradictory noise")
    glass_bar1 = plt.bar(left=[i + 0.4 for i in x], height=glass_misclassified, width=0.4, color='deepskyblue', label="misclassified noise")
    plt.ylim(0, 1)     # the range of y
    plt.ylabel("accuracy")
    plt.xticks([index + 0.2 for index in x], label_x)
    plt.xlabel("noise percentage")
    plt.title("The Accuracy of Glass Dataset")
    plt.plot(x, glass_contradictory, "b--", linewidth=1, marker = '.')
    plt.plot([i + 0.4 for i in x], glass_misclassified, "b-", linewidth=1, marker = '.')
    plt.legend()

    for rect in glass_bar0:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height), ha="center", va="bottom")
    for rect in glass_bar1:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height), ha="center", va="bottom")

    # draw the tic tac toe dataset accuracy
    ax = fig1.add_subplot(122)
    tic_bar0 = plt.bar(left=x, height=tic_contradictory, width=0.4, alpha=0.8, color='lightcoral', label="contradictory noise")
    tic_bar1 = plt.bar(left=[i + 0.4 for i in x], height=tic_misclassified, width=0.4, color='mistyrose', label="misclassified noise")
    plt.ylim(0, 1)  # the range of y
    plt.ylabel("accuracy")
    plt.xticks([index + 0.2 for index in x], label_x)
    plt.xlabel("noise percentage")
    plt.title("The Accuracy of tic-tac-toe Dataset")
    plt.plot(x, tic_contradictory, "r--", linewidth=1, marker='.')
    plt.plot([i + 0.4 for i in x], tic_misclassified, "r-", linewidth=1, marker='.')
    plt.legend()

    for rect in tic_bar0:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height), ha="center", va="bottom")
    for rect in tic_bar1:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height), ha="center", va="bottom")

    plt.show()







