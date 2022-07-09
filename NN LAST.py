import numpy as np
from keras.datasets import mnist

(train_X, train_y), (test_X, test_y) = mnist.load_data()

def grid_photo(array, row, col): #Griding the photo
    r, h = array.shape
    return array.reshape(h // row, row, -1, col).swapaxes(1, 2).reshape(-1, row, col)


def calc_cent(img): #calculate the centroid
    feature_vector = []
    for grid in grid_photo(img, 4, 4):
        Xc = 0
        Yc = 0
        sum = 0
        for index, x in np.ndenumerate(grid):
            sum += x
            Xc += x * index[0]
            Yc += x * index[1]

        if sum != 0:
            feature_vector.append(Xc / sum)
            feature_vector.append(Yc / sum)
        else:
            feature_vector.append(0)
            feature_vector.append(0)
    return np.array(feature_vector)


def one_hot(x): #To make the labels one hot vector
    one_hot_encode = []
    for c in x:
        arr = list((np.zeros(10, dtype=int)))
        arr[c] = 1
        one_hot_encode.append(arr)
    return np.array(one_hot_encode)


def sigmoid(s):
    return 1 / (1 + np.exp(-s))


def derv_sigmoid(x):
    return x * (1 - x)


class Neural_Network():
    def __init__(self):
        self.inputSize = train.shape[1]
        self.outputSize = train_y.shape[1]
        self.nuerons = 4  #can be changed as you want
        self.W1 = np.random.randn(self.inputSize, self.nuerons)
        self.W2 = np.random.randn(self.nuerons, self.outputSize)

    def forward(self, X):
        self.layer1 = sigmoid(np.dot(X, self.W1))
        out = sigmoid(np.dot(self.layer1, self.W2))
        return out

    def backward(self, X, y, o, learningRate):
        delta2 = learningRate * np.dot(self.layer1.T, derv_sigmoid(o) * (y - o))

        delta1 = np.dot(X.T,
                        (np.dot((y - o) * derv_sigmoid(o),
                                self.W2.T) * derv_sigmoid(self.layer1)))
        self.W2 += delta2
        self.W1 += delta1

    def train(self, X, y):
        o = self.forward(X)
        self.backward(X, y, o, 0.05)

    def predict(self, t):
        x = self.forward(t)
        return x


def calculate_accuracy(y_test_vector, prediction):
    counter = 0

    for i in range(len(y_test_vector)):
        if y_test_vector[i] == prediction[i]:
            counter = counter + 1

    return (counter / len(prediction)) * 100


def returnLabeled(X): #To return the labels from one hot vector to value
    Labeled = []
    for c in X:
        max = 0
        index = 0
        maxind = 0
        arr = list((np.zeros(1, dtype=int)))
        for value in c:
            if value > max:
                max = value
                maxind = index
            index = index + 1
        arr[0] = maxind
        Labeled.append(arr)
    return np.array(Labeled)


train_y = one_hot(train_y)
test_y = one_hot(test_y)
train = [calc_cent(i) for i in train_X]
train = np.array(train)
test = [calc_cent(i) for i in test_X]
test = np.array(test)
NN = Neural_Network()
for i in range(10):
    NN.train(train, train_y)
tr = NN.forward(train)
print("Train Accuracy \n")
Actual_Train = returnLabeled(train_y)
Predicted_Train = returnLabeled(tr)
print(calculate_accuracy(Actual_Train, Predicted_Train))

print("Test Accuracy \n")
pre = NN.predict(test)
Actual_Test = returnLabeled(test_y)
Predicted_Test = returnLabeled(pre)
print(calculate_accuracy(Actual_Test, Predicted_Test),"%")
