import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.metrics import confusion_matrix
import matplotlib.patches as mpatches
from scipy.stats import multivariate_normal
from sklearn.metrics import mean_squared_error
from mlxtend.data import loadlocal_mnist
from sklearn.metrics import accuracy_score
import operator

##################### Question 1 ############################
#question 1.a
#because the mean is 0 and the variance is 1, this is Normal distribution
sampleNo = 1000
mean = np.array([0,0])
#class1
cov1 = np.eye(2)
ms1 = np.random.multivariate_normal(mean, cov1, sampleNo)
fig1 = plt.figure(0)
ax = fig1.add_subplot(121, aspect='equal')
plt.title('matrix a')
plt.scatter(ms1[:,0],ms1[:,1],s = 2,alpha = .5)
ax.grid(True)
#class2
cov2 = np.array([[1, 0.9], [0.9, 1]])
ms2 = np.random.multivariate_normal(mean, cov2, sampleNo)
ax = fig1.add_subplot(122, aspect='equal')
plt.title("matrix b")
plt.scatter(ms2[:,0],ms2[:,1],s = 2,alpha = .5)
ax.grid(True)
plt.show()


#question 1.b
fig2 = plt.figure(0)
def std_contour(cov,position):
    #calculate the center of contour
    mean_x = 0
    mean_y = 0

    #calculate the direction of contour
    eigVals, eigVects = np.linalg.eig(cov)
    index = eigVals.argsort()[::-1]
    sortedEigval = eigVals[index]
    sortedEeigvec = eigVects[:, index]
    long = -sortedEeigvec[:, 0]  # guarantee the direction of contour is corresponding with the sample datas
    tan1 = long[1]/long[0]
    angle1 = math.degrees(math.atan(tan1))


    #calculate the length of X & Y
    len_long = np.sqrt(sortedEigval[0])
    len_short = np.sqrt(sortedEigval[1])


    #draw contour
    ax = fig2.add_subplot(position, aspect='equal')
    ax.grid(True)
    e = Ellipse(xy=(mean_x, mean_y), width=len_long * 2,
                height=len_short * 2, angle=angle1,
                fill=False, linewidth=1.2, edgecolor="red")
    ax.add_artist(e)

plt.subplot(121, aspect='equal')
plt.scatter(ms1[:,0],ms1[:,1],s = 2,alpha = .5)
plt.subplot(122, aspect='equal')
plt.scatter(ms2[:,0],ms2[:,1],s = 2,alpha = .5)
std_contour(cov1,121)
std_contour(cov2,122)
plt.show()

#question 1.c
def cov(x,y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    sum = np.sum((x-x_mean)*(y-y_mean.T))
    return sum/(len(x)-1) #the sample variance should be divided by n-1 rather than n
def cov_matrix(x,y):
    xx = cov(x,x)
    xy = cov(x,y)
    yy = cov(y,y)
    print ("[" + "["+str(xx)+' '+str(xy)+"]" + '\n '
           + "["+str(xy)+' '+str(yy)+"]" + ']')
#class1
cov_matrix(ms1[:,0],ms1[:,1])
#class2
cov_matrix(ms2[:,0],ms2[:,1])

##################### Question 2 ############################
#question2.a
sampleNo = 1000
sampleNo1 = 600
sampleNo2 = 900
sampleNo3 = 1500


#class1
cov1 = np.array([[1, -1], [-1, 2]])
mean1 = np.array([3,2])
ms1 = np.random.multivariate_normal(mean1, cov1, sampleNo1)
#class2
cov2 = np.array([[1, -1], [-1, 7]])
mean2 = np.array([5,4])
ms2 = np.random.multivariate_normal(mean2, cov2, sampleNo2)
#class3
cov3 = np.array([[0.5, 0.5], [0.5, 3]])
mean3 = np.array([2,5])
ms3 = np.random.multivariate_normal(mean3, cov3, sampleNo3)


def new_std_contour(mean, cov):
    #calculate the center of contour
    mean_x = mean[0]
    mean_y = mean[1]
    plt.plot(mean_x,mean_y,'r+')

    #calculate the direction of contour
    eigVals, eigVects = np.linalg.eig(cov)
    index = eigVals.argsort()[::-1]
    sortedEigval = eigVals[index]
    sortedEeigvec = eigVects[:, index]
    long = -sortedEeigvec[:, 0]  # guarantee the direction of contour is corresponding with the sample datas
    tan = long[1] / long[0]
    angle = math.degrees(math.atan(tan))

    #calculate the length of X & Y
    len_long = np.sqrt(sortedEigval[0])
    len_short = np.sqrt(sortedEigval[1])

    ##draw contour
    e = Ellipse(xy=(mean_x, mean_y), width=len_long * 2,
                height=len_short * 2, angle=angle,
                fill=False, linewidth=2.5, edgecolor="red")
    ax.add_artist(e)


#decision boundary
def ML_classifier(matrix):
    mean1 = [3,2]
    mean2 = [5,4]
    mean3 = [2,5]
    cov1 = [[1, -1], [-1, 2]]
    cov2 = [[1, -1], [-1, 7]]
    cov3 = [[0.5, 0.5], [0.5, 3]]
    label = []

    for i in range(len(matrix)):
        x = matrix[i][0]
        y = matrix[i][1]
        mvn1 = multivariate_normal(mean1, cov1)
        likelihood1 = mvn1.pdf(matrix[i])
        mvn2 = multivariate_normal(mean2, cov2)
        likelihood2 = mvn2.pdf(matrix[i])
        mvn3 = multivariate_normal(mean3, cov3)
        likelihood3 = mvn3.pdf(matrix[i])

        if ((likelihood1 >= likelihood2) and (likelihood1 >= likelihood3)):
            label.append(1)
        elif ((likelihood2 >= likelihood1) and (likelihood2 >= likelihood3)):
            label.append(2)
        elif ((likelihood3 >= likelihood2) and (likelihood3 >= likelihood1)):
            label.append(3)

    return np.array(label)

def MAP_classifier(matrix):
    mean1 = [3,2]
    mean2 = [5,4]
    mean3 = [2,5]
    cov1 = [[1, -1], [-1, 2]]
    cov2 = [[1, -1], [-1, 7]]
    cov3 = [[0.5, 0.5], [0.5, 3]]
    label = []

    for i in range(len(matrix)):
        mvn1 = multivariate_normal(mean1, cov1)
        posteriori1 = mvn1.pdf(matrix[i])*0.2
        mvn2 = multivariate_normal(mean2, cov2)
        posteriori2 = mvn2.pdf(matrix[i])*0.3
        mvn3 = multivariate_normal(mean3, cov3)
        posteriori3 = mvn3.pdf(matrix[i])*0.5
        if ((posteriori1 >= posteriori2) and (posteriori1 >= posteriori3)):
            label.append(1)
        elif ((posteriori2 >= posteriori1) and (posteriori2 >= posteriori3)):
            label.append(2)
        elif ((posteriori3 >= posteriori2) and (posteriori3 >= posteriori1)):
            label.append(3)

    return np.array(label)


def plot_decision_boundary(model, axis, style):
    # draw the mean and standard deviation contour
    new_std_contour(mean1, cov1)
    new_std_contour(mean2, cov2)
    new_std_contour(mean3, cov3)
    # meshgrid function can draw grid on figure and return coordinate matrix
    X0, X1 = np.meshgrid(
    # Randomly generate two sets of numbers, the starting value and density are determined by the starting value of the axis
        np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 10)).reshape(-1, 1),
        np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 10)).reshape(-1, 1),
    )
    X_grid_matrix = np.c_[X0.ravel(), X1.ravel()]

    # train and predict
    if (model == 'ML'):
        y_predict = ML_classifier(X_grid_matrix)
        y_predict_matrix = y_predict.reshape(X0.shape)
    if (model == 'MAP'):
        y_predict = MAP_classifier(X_grid_matrix)
        y_predict_matrix = y_predict.reshape(X0.shape)

    # set the filled color
    from matplotlib.colors import ListedColormap
    my_colormap1 = ListedColormap(['lightcoral', 'mistyrose', 'pink'])
    my_colormap2 = ListedColormap(['deepskyblue', 'dodgerblue', 'skyblue'])

    # draw the Ml and MAP decision boundary with different color
    if (style == 2):
        plt.contourf(X0, X1, y_predict_matrix, cmap=my_colormap2, alpha=0.8)
        plt.contour(X0, X1, y_predict_matrix, colors='deepskyblue', linewidths=1.5, alpha=0.7)
    elif (style == 1):
        plt.contourf(X0, X1, y_predict_matrix, cmap=my_colormap1, alpha=0.8)
        plt.contour(X0, X1, y_predict_matrix, colors='firebrick', linewidths=1.5, alpha=0.7)
    elif (style == 3):
        plt.contour(X0, X1, y_predict_matrix, colors='firebrick', linewidths=1.5, alpha=0.7)
    elif (style == 4):
        plt.contour(X0, X1, y_predict_matrix, colors='deepskyblue', linewidths=1.5, alpha=0.7)

# plot the ML classifier
fig = plt.figure(0)
ax = fig.add_subplot(111, aspect='equal')
ax.grid(True)
plot_decision_boundary('ML', axis=[-2, 8, -6, 13], style=1)
ML_class1 = mpatches.Patch(color='lightcoral', label='class1')
ML_class2= mpatches.Patch(color='mistyrose', label='class2')
ML_class3 = mpatches.Patch(color='pink', label='class3')
plt.legend(handles=[ML_class1,ML_class2,ML_class3])
plt.title('ML classifier')
plt.show()

# plot the MAP classifier
fig = plt.figure(0)
ax = fig.add_subplot(111, aspect='equal')
ax.grid(True)
plot_decision_boundary('MAP', axis=[-2, 8, -6, 13], style=2)
MAP_class1 = mpatches.Patch(color='deepskyblue', label='class1')
MAP_class2= mpatches.Patch(color='dodgerblue', label='class2')
MAP_class3 = mpatches.Patch(color='skyblue', label='class3')
plt.legend(handles=[MAP_class1,MAP_class2,MAP_class3])
plt.title('MAP classifier')
plt.show()

# plot both of them and compare with each other
fig = plt.figure(0)
ax = fig.add_subplot(111, aspect='equal')
ax.grid(True)
plot_decision_boundary('ML', axis=[-2, 8, -6, 13], style=3)
plot_decision_boundary('MAP', axis=[-2, 8, -6, 13], style=4)
ML_line = mpatches.Patch(color='firebrick', label='ML')
MAP_line = mpatches.Patch(color='deepskyblue', label='MAP')
plt.legend(handles=[ML_line,MAP_line])
plt.title('compare ML with MAP')
plt.show()


# question2.b
# plot the unclassified sample data
fig = plt.figure(0)
ax = fig.add_subplot(111, aspect='equal')
label1 = np.ones((sampleNo1,1),int)
label2 = np.ones((sampleNo2,1),int)*2
label3 = np.ones((sampleNo3,1),int)*3
label_true = np.vstack((label1,label2,label3))
ms = np.vstack((ms1,ms2,ms3))
plt.scatter(ms[:,0],ms[:,1],s = 2,alpha = .5)
plt.show()

# assign the true label to different class
label_predict_ML = ML_classifier(ms).T
label_predict_MAP = MAP_classifier(ms).T

# plot the classified sample data and compared with true label
fig = plt.figure(0)
ax = fig.add_subplot(111, aspect='equal')
plt.scatter(ms1[:,0],ms1[:,1],s = 2,alpha = .5,label='class1')#class1
plt.scatter(ms2[:,0],ms2[:,1],s = 2,alpha = .5,label='class2')#class2
plt.scatter(ms3[:,0],ms3[:,1],s = 2,alpha = .5,label='class3')#class3
plot_decision_boundary('ML', axis=[-2, 8, -6, 13], style=3)
plot_decision_boundary('MAP', axis=[-2, 8, -6, 13], style=4)
ML_line = mpatches.Patch(color='firebrick', label='ML')
MAP_line = mpatches.Patch(color='deepskyblue', label='MAP')
plt.legend(handles=[ML_line,MAP_line])
plt.show()

# calculate the confusion matrix
ML_confusion = confusion_matrix(label_true, label_predict_ML)
MAP_confusion = confusion_matrix(label_true, label_predict_MAP)
print(ML_confusion)
print(MAP_confusion)

# experimental P(ε)
def calculate_error(matrix):
    sum = matrix[0][0]+matrix[1][1]+matrix[2][2]
    return (3000-sum)/3000
print('the error of ML', calculate_error(ML_confusion))
print('the error of MAP', calculate_error(MAP_confusion))



##################### Question 3 ############################
# load mnist date
X, y = loadlocal_mnist(
        images_path='train-images-idx3-ubyte',
        labels_path='train-labels-idx1-ubyte')
X, y = np.float32(np.array(X)),np.array(y) # change the data type from int8 to float

#question3.a
def normalization(matrix):
    matrix = np.array(matrix)
    len_data = matrix.shape[0]
    mean=np.mean(matrix,axis=0)     #calculate the mean of each feature
    newData=matrix-mean
    return newData,mean

def pca(matrix, n):
    global n_eigVect
    newData,meanVal=normalization(matrix)
    cov=np.cov(newData,rowvar=0)          # rowvar is 0 means that a row is a recrod
    covMat=np.mat(cov)
    eigVals,eigVects=np.linalg.eig(covMat)# calculate the eigenvalue and eigenvector
    idx_eigVal=np.argsort(eigVals)            # sort the eigenvalue
    n_idx_eigVal=idx_eigVal[-1:-(n+1):-1]   # retain the index of top n eigenvalues
    n_eigVect=eigVects[:,n_idx_eigVal]        # get the corresponding eigenvectors
    newMat=newData * n_eigVect                # get the low dimension data
    return newMat

lowDDataMat = pca(X, 100)
print('Dimensions: %s x %s' % (lowDDataMat.shape[0], lowDDataMat.shape[1]))

#question3.b
def percentage(eigVals,percentage):
    sortArray=np.sort(eigVals)   # sorted in increasing order
    sortArray=sortArray[-1::-1]  # sorted in decreasing order
    arraySum=sum(sortArray)
    sum=0
    num=0
    for i in sortArray:
        sum+=i
        num+=1
        if sum>=arraySum*percentage:
            return num

def pca_percentage(matrix,percentage):
    newData,meanVal=normalization(matrix)
    covMat=np.mat(np.cov(newData,rowvar=0))
    eigVals,eigVects=np.linalg.eig(covMat)
    n=percentage(eigVals,percentage)          # calculate the how many eigvalue can reach the percentage
    idx_eigVal = np.argsort(eigVals)            # sort the eigenvalue
    n_idx_eigVal = idx_eigVal[-1:-(n + 1):-1]  # retain the index of top n eigenvalues
    n_eigVect = eigVects[:, n_idx_eigVal]      # get the corresponding eigenvectors
    newMat=newData*n_eigVect
    return newMat
lowDataMat = pca_percentage(X, 0.95)
print('Dimensions: %s x %s' % (lowDataMat.shape[0], lowDataMat.shape[1]))

#question3.c
def pca_reconstruction(matrix, n):
    newData,meanVal=normalization(matrix)
    cov=np.cov(newData,rowvar=0)
    eigVals,eigVects=np.linalg.eig(np.mat(cov))
    idx_eigVal = np.argsort(eigVals)           # sort the eigenvalue
    n_idx_eigVal = idx_eigVal[-1:-(n + 1):-1]  # retain the index of top n eigenvalues
    n_eigVect = eigVects[:, n_idx_eigVal]     # get the corresponding eigenvectors
    newMat=newData*n_eigVect
    reconMat=(newMat*n_eigVect.T)+meanVal  # reconstruct the matrix
    return np.real(reconMat)

true = X
y = []
x = range(1,785)
for i in x:
    pred = pca_reconstruction(X, i)
    y.append(mean_squared_error(true,pred).real)
plt.plot(x, y, 'r')
plt.show()

#question3.d
def show_image(image_array):
    plt.gray()
    plt.imshow(image_array.reshape([28, 28]))
    plt.axis('off')

site = 151
for j in [1, 10, 50, 250, 784]:
    ax = plt.subplot(site)
    site += 1
    re_X = pca_reconstruction(X,j)
    show_image(re_X[0])
plt.show()

#question3.e
newData,meanVal=normalization(X)
covMat=np.cov(newData,rowvar=0)
eigVals,eigVects=np.linalg.eig(np.mat(covMat))
sortedEigVal=np.sort(eigVals)
plt.plot(range(784,0,-1),sortedEigVal) # show the eigenvalue in descending order
plt.show()


##################### Question 4 ############################
X_test, y_test = loadlocal_mnist(
        images_path='t10k-images-idx3-ubyte',
        labels_path='t10k-labels-idx1-ubyte')
X_test, y_test = np.float32(np.array(X_test)),np.array(y_test)
X_train, y_train = loadlocal_mnist(
        images_path='train-images-idx3-ubyte',
        labels_path='train-labels-idx1-ubyte')
X_train, y_train = np.float32(np.array(X_train)),np.array(y_train)

#question 4.a
def kNNClassify(newInput, dataSet, labels, k):
    numSamples = dataSet.shape[0]
    diff = np.tile(newInput, (numSamples, 1)) - dataSet  # Subtract element-wise
    squaredDiff = diff ** 2  # squared for the subtract
    squaredDist = np.sum(squaredDiff, axis=1)  # sum is performed by row
    distance = squaredDist ** 0.5

    sortedDistIndices = np.argsort(distance)
    classCount = {}  # define a dictionary (can be append element)
    for j in range(k):
        voteLabel = labels[sortedDistIndices[j]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

    # return the max voted class
    sortedClass = sorted(classCount.items(), key=operator.itemgetter(1),reverse=True)

    return sortedClass[0][0]

for k in [1,3,5,11]:
    predictLabel = []
    for j in range(len(X_test)):
        predictLabel.append(kNNClassify(X_test[j],X_train,y_train,k))
    trueLabel = y_test
    accuracy = accuracy_score(trueLabel,predictLabel)
    print("k=", k, " accuracy:", accuracy)
    print("############################")

#question 4.b
def new_kNNClassify(newInput, dataSet, labels, k):
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
for d in [5,50,100,500]:
    low_train=pca(X_train,d)
    result=[]
    meanRemoved,meanvalue = normalization(X_test)
    testingMat = np.real(meanRemoved * n_eigVect)
    for k in [1,3,5,11]:
        result = new_kNNClassify(testingMat, low_train, y_train, k)
        accuracy = accuracy_score(y_test, result)
        print ('d=',d,'k=',k,'accuracy is ', accuracy)
        print('############################')
