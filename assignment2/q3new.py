import numpy as np
c
from scipy.spatial.distance import pdist
import collections
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import operator
from sklearn.neighbors import KNeighborsClassifier

############################## Question3 ###############################
# download dataset
X, y = loadlocal_mnist(
        images_path='train-images-idx3-ubyte',
        labels_path='train-labels-idx1-ubyte')
X_test, y_test = np.float32(np.array(X))[0:10000], np.array(y)[0:10000]

X, y = loadlocal_mnist(
        images_path='t10k-images-idx3-ubyte',
        labels_path='t10k-labels-idx1-ubyte')
X_train, y_train = np.float32(np.array(X)), np.array(y)


def InterClassDistance(dataset, labels):
    distance_list = []
    feature_mean = []
    total_mean = []
    sum_distance = 0
    for label in set(labels):
        mean_list = []
        indices = [count for count, item in enumerate(labels) if item == label]
        data = dataset[indices]
        for feature in data.T:
            mean_list.append(np.mean(feature))
        mean_list = np.array(mean_list)
        feature_mean.append(np.mean(mean_list))
    for f in np.array(feature_mean).T:
        total_mean.append(np.mean(f))
    for x in range(0, len(feature_mean)):
        distance = np.linalg.norm(feature_mean[x] - total_mean)  # calculate the Euclidean distance
        distance_list.append(distance)
        sum_distance += distance
    return (sum_distance)

def SFS(dataset, labels, selected_list): # Sequential Forward Selection
    J_list = [] # record the inter class distance
    feature_array = dataset.T
    for feature in dataset.T:
        selected_list = np.append(selected_list, [feature], axis=0)
        J = InterClassDistance(selected_list.T, labels)
        J_list.append(J)
        selected_list = np.delete(selected_list, -1, 0) # remove the preview test feature
    idx = np.array(J_list).argsort()[::-1][0]
    selected_list = np.append(selected_list, [dataset.T[idx]], axis=0) # add feature into selected list
    feature_array = np.delete(feature_array, idx, 0) # delete the selected feature
    return selected_list, feature_array.T


def SBS(dataset, selected_list, labels): # Sequential Backward Selection
    J_list = [] # record the inter class distance (J)
    feature_array = dataset.T
    for feature in dataset.T:
        selected_list = np.append(selected_list, [feature], axis=0)
        J = InterClassDistance(selected_list.T, labels)
        J_list.append(J)
        selected_list = np.delete(selected_list, -1, 0) # remove the preview test feature
    idx = np.array(J_list).argsort()[::-1][-1]
    feature_set = np.delete(feature_array, idx, 0) # delete the worst feature
    return feature_set.T


def BDS(dataset, d, labels):
    len_dataset = len(dataset)
    selected = np.empty(shape=[0, len_dataset]) # create a new empty list to save the select feature
    for count in range(0, d):
        selected, dataset = SFS(dataset, labels, selected)
        dataset = SBS(dataset, selected, labels)
        print("count: ", count)
    return np.array(selected).T


def kNNClassify(newInput, dataSet, labels, k):
    predictLabel = []
    for i in range(len(newInput)):
        numSamples = dataSet.shape[0]
        diff = np.array(np.tile(newInput[i], (numSamples, 1)) - dataSet)  # Subtract element-wise
        squaredDiff = diff ** 2  # squared for the subtract
        squaredDist = np.sum(squaredDiff, axis=1)  # sum is performed by row
        distance = squaredDist ** 0.5

        sortedDistIndices = np.argsort(distance)
        classCount = {}  # define a dictionary (can be append element)
        for j in range(k):
            voteLabel = labels[sortedDistIndices[j]]
            classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

        ## step 5: the max voted class will return
        sortedClass = sorted(classCount.items(), key=operator.itemgetter(1),reverse=True)
        predictLabel.append(sortedClass[0][0])

    return predictLabel

# def KNN_classifier(test_data, training_data, labels, k):
#     distance_list = []
#     targets = []
#     for x_train in range(len(training_data)):
#         distance = np.sqrt(np.sum(np.square(test_data - training_data[x_train, :])))
#         distance_list.append([distance, x_train])
#     distance_list = sorted(distance_list)
#     for m in range(0, k):
#         index = distance_list[m][1]
#         targets.append(labels[index])
#     return collections.Counter(targets).most_common(1)[0][0]
#
#
# def KNN_accuracy(test_data, test_label, training_data, training_label, K):
#     KNN_result = KNN_TestResult(test_data, training_data, training_label, K)
#     True_result = test_label
#     score = accuracy_score(True_result, KNN_result)
#     return score
#
#
# def KNN_TestResult(test_data, training_data, training_label, K):
#     result_list = []
#     a = 0
#     for x in test_data:
#         a += 1
#         result = KNN_classifier(x, training_data, training_label, K)
#         result_list.append(result)
#     return np.array(result_list)


def index_list(origin_data, reduced_data):
    index_list = []
    for data in reduced_data.T:
        index = origin_data.T.tolist().index(data.tolist())
        index_list.append(index)
    return index_list


def LDA_algorithm(data, labels, k):
    label_list = list(set(labels))
    X_classify = {}
    mean_classify = {}

    for label in set(labels):
        indices = [m for m, item in enumerate(labels) if item == label]
        X_classify[label] = np.array(data[indices])
        mean_classify[label] = np.mean(X_classify[label], axis=0)

    mean_list = [np.mean(feature) for feature in data.T]
    Sb, Sw = np.zeros((784, 784)), np.zeros((784, 784))
    for i in label_list:
        Sw += np.dot((X_classify[i] - mean_classify[i]).T,
                     X_classify[i] - mean_classify[i])
        Sb += len(X_classify[i]) * np.dot((mean_classify[i] - mean_list).reshape(
            (len(mean_list), 1)), (mean_classify[i] - mean_list).reshape((1, 784)))

    eig_value, eig_vector = np.linalg.eig(np.linalg.pinv(Sw).dot(Sb))
    sorted_indices = np.argsort(eig_value)
    topk_eig_vecs = eig_vector[:, sorted_indices[:-k - 1:-1]]
    return topk_eig_vecs.real


if __name__ == '__main__':

######### Q3.a ###########
    data_all = np.append(X_train, X_test, axis=0)
    label_all = np.append(y_train, y_test, axis=0)
    data = BDS(data_all, 3, y_train)
    print (data)

    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(data[:10000], label_all[:10000])
    test_result = neigh.predict(data[10000:])
    test_score = neigh.score(data[10000:], label_all[10000:])
    print(test_score, 'test_score')
    index = index_list(data_all, data)
    print(index)

    '''
    new_array = np.zeros(784)
    new_array[index] = 256
    file_name = "recon_" + str(10) + ".png"
    img = new_array.reshape(28,28)
    cv2.imwrite(file_name, img)
    '''




    '''
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    
    # LDA
    sklearn_lda = LDA(n_components=5)
    w = sklearn_lda.fit_transform(data1, label)
    '''
    w = LDA_algorithm(data_all, label_all, 10)
    train_data = np.dot(X_train, w)
    test_data = np.dot(X_test, w)

    from sklearn.neighbors import KNeighborsClassifier

    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(train_data, y_train)
    test_result = neigh.predict(test_data)
    test_score = neigh.score(test_data, y_test)
    print(test_score, 'test_score')









