import numpy as np
from mlxtend.data import loadlocal_mnist
from scipy.spatial.distance import pdist
import cv2
import collections
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def loaddata_training():
    X, y = loadlocal_mnist(
        images_path='train-images-idx3-ubyte',
        labels_path='train-labels-idx1-ubyte')
    return np.float32(np.array(X))[0:10000], np.array(y)[0:10000]


def loaddata_test():
    X, y = loadlocal_mnist(
        images_path='t10k-images-idx3-ubyte',
        labels_path='t10k-labels-idx1-ubyte')
    return np.float32(np.array(X)), np.array(y)


training_data, training_label = loaddata_training()
test_data, test_label = loaddata_test()


def Euclidean_distace(vec1, vec2):
    distance = np.linalg.norm(vec1 - vec2)
    return distance


def interclass_distance(dataset, labels):
    feature_mean = []
    distance_list = []
    total_mean = []
    for label in set(labels):
        mean_list = []
        indices = [m for m, x in enumerate(labels) if x == label]
        data = dataset[indices]
        feature_list = data.T
        for feature in data.T:
            mean_list.append(np.mean(feature))
        mean_list = np.array(mean_list)
        feature_mean.append(mean_list)
    for f in np.array(feature_mean).T:
        total_mean.append(np.mean(f))

    for x in range(0, len(feature_mean)):
        distance_list.append(Euclidean_distace(feature_mean[x], total_mean))
    return (sum(distance_list))


def sfs_algorithm(dataset, labels, selected_list):
    grade_list = []
    feature_set = dataset.T
    for feature in dataset.T:
        selected_list = np.append(selected_list, [feature], axis=0)
        grade = interclass_distance(selected_list.T, labels)
        grade_list.append(grade)
        selected_list = np.delete(selected_list, -1, 0)
    index = grade_list.index(max(grade_list))
    # print (index, 'max_index')
    # print ([grade_list[index]], 'max_distance')
    selected_list = np.append(selected_list, [dataset.T[index]], axis=0)
    feature_set = np.delete(feature_set, index, 0)
    return selected_list, feature_set.T


def sbs_algorithm(dataset, selected_list, labels):
    grade_list = []
    feature_set = dataset.T
    for feature in dataset.T:
        selected_list = np.append(selected_list, [feature], axis=0)
        grade = interclass_distance(selected_list.T, labels)
        grade_list.append(grade)
        selected_list = np.delete(selected_list, -1, 0)
    index = grade_list.index(min(grade_list))
    feature_set = np.delete(feature_set, index, 0)
    return feature_set.T


def BDS_algorithm(dataset, d, labels):
    selected_list = np.empty(shape=[0, len(dataset)])
    feature_len = len(dataset.T)
    reduce_count = feature_len - d
    for count in range(0, d):
        selected_list, dataset = sfs_algorithm(dataset, labels, selected_list)
        if reduce_count > 0:
            dataset = sbs_algorithm(dataset, selected_list, labels)
            reduce_count -= 1
    return np.array(selected_list).T


def KNN_classifier(test_data, training_data, labels, k):
    distance_list = []
    targets = []
    for x_train in range(len(training_data)):
        distance = np.sqrt(np.sum(np.square(test_data - training_data[x_train, :])))
        distance_list.append([distance, x_train])
    distance_list = sorted(distance_list)
    for m in range(0, k):
        index = distance_list[m][1]
        targets.append(labels[index])
    return collections.Counter(targets).most_common(1)[0][0]


def KNN_accuracy(test_data, test_label, training_data, training_label, K):
    KNN_result = KNN_TestResult(test_data, training_data, training_label, K)
    True_result = test_label
    score = accuracy_score(True_result, KNN_result)
    return score


def KNN_TestResult(test_data, training_data, training_label, K):
    result_list = []
    a = 0
    for x in test_data:
        a += 1
        result = KNN_classifier(x, training_data, training_label, K)
        result_list.append(result)
    return np.array(result_list)


def index_list(origin_data, reduced_data):
    index_list = []
    for data in reduced_data.T:
        index = origin_data.T.tolist().index(data.tolist())
        index_list.append(index)
    return index_list

d_list = [10, 50, 150, 392]
data1 = np.append(training_data, test_data, axis=0)
label = np.append(training_label, test_label, axis=0)
from sklearn.neighbors import KNeighborsClassifier
print("------------- BDS feature selection --------------")

for d in d_list:
    data = BDS_algorithm(data1, d, training_label)
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(data[:10000], label[:10000])
    test_result = neigh.predict(data[10000:])
    test_score = neigh.score(data[10000:], label[10000:])
    print('test_score: ', test_score, "d: ", d)

index = index_list(data1, data)
print(index)
index = np.load("index_data.npy")

new_array = np.zeros(784)
# 392 features
new_array[index] = 256
file_name = "recon_" + str(392) + ".png"
img = new_array.reshape(28,28)
cv2.imwrite(file_name, img)
# 150 features
new_array = np.zeros(784)
index = index[:150]
new_array[index] = 256
file_name = "recon_" + str(150) + ".png"
img = new_array.reshape(28,28)
cv2.imwrite(file_name, img)
# 50 features
new_array = np.zeros(784)
index = index[:50]
new_array[index] = 256
file_name = "recon_" + str(50) + ".png"
img = new_array.reshape(28,28)
cv2.imwrite(file_name, img)
# 10 features
index = index[:10]
new_array = np.zeros(784)
new_array[index] = 256
file_name = "recon_" + str(10) + ".png"
img = new_array.reshape(28,28)
cv2.imwrite(file_name, img)


def LDA_algorithm(data, labels, k):
    label_list = list(set(labels))
    X_classify = {}
    mean_classify = {}

    for label in set(labels):
        indices = [m for m, x in enumerate(labels) if x == label]
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


'''
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# LDA
sklearn_lda = LDA(n_components=5)
w = sklearn_lda.fit_transform(data1, label)
'''
print("---------------- LDA ---------------------")
from sklearn.neighbors import KNeighborsClassifier

for d in d_list:

    w = LDA_algorithm(training_data, training_label, d)
    new_train_data = np.dot(training_data, w)
    new_test_data = np.dot(test_data, w)

    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(new_train_data, training_label)
    test_result = neigh.predict(new_test_data)
    test_score = neigh.score(new_test_data, test_label)
    print('test_score: ', test_score, "d: ", d )
