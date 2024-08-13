
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import matplotlib.pyplot as plt

def read_files(path: str, ans: int, target_dim: tuple = (256, 256)):
    files = os.listdir(path)
    X = None
    for i, name in enumerate(files):
        img = cv2.imread(path + '/' + name, 0) # 0 means black-white picture
        if img.shape != 0:
            img = cv2.resize(img, (256, 256))
            vect = img.reshape(1, 256 ** 2) / 255.

            X = vect if (X is None) else np.vstack((X, vect))
        print(f"{i}/{len(files)}")
    print()
    y = np.ones((len(X),1)) * ans
    return X, y


X_b, y_b = read_files("lesson1_dataset/box", 1)
X_nb, y_nb = read_files("lesson1_dataset/no_box", 0)

y_ = np.vstack((y_nb, y_b))
X_ = np.vstack((X_nb, X_b))

#Разделение набора данных на обучающий (80%) и тестовый (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X_, y_, test_size=0.2, random_state=42)

def sigmoid(z):
   a = 1 / (np.exp(-z) + 1)
   return a

def propogate(w, b, X, Y):
    m = X.shape[1]

    z = np.dot(w.T, X) + b
    A = sigmoid(z)
    cost = (-np.sum(Y*np.log(A)+(1-Y)*np.log(1-A)))/m

    dw = (np.dot(X, (A-Y).T))/m
    db = np.average(A-Y)

    cost = np.squeeze(cost)
    grads = {"dw" : dw,
             "db" : db}
    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    costs = []
    for i in range(num_iterations):
        grads, cost = propogate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate*dw
        b = b - learning_rate*db


        if i % 100 == 0:
            costs.append(cost)
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" %(i, cost))


    params = {"w" : w,
              "b" : b}
    grads = {"dw": dw,
             "db": db}


    return params, grads, costs


def predict(w, b, X):
    m = X.shape[0]

    Y_prediction = np.zeros((1, m))
    #print(w.shape[0])
    if (X.shape[0]==36):
        w = w.reshape((X.shape[0], 36))
    else:
        w = w.reshape((X.shape[0], 144))

    z = np.dot(w.T, X) + b
    A = sigmoid(z)

    for i in range(X.shape[0]):
        if A[0, i] > 0.5:
            Y_prediction[[0], [i]] = 1
        else:
            Y_prediction[[0], [i]] = 0

    return Y_prediction


def init_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0
    return w, b

def model(X_train, X_test, y_train, y_test, num_iterations, learning_rate, print_cost = False):
    w, b = init_zeros(X_train.shape[0])

    parameters, grads, costs = optimize(w, b, X_train, y_train, num_iterations, learning_rate, print_cost)
    w = parameters["w"]
    b = parameters["b"]

    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)


    print("train accuracy: {} %".format(100-np.mean(np.abs(Y_prediction_train - y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - y_test)) * 100))

    '''
    print("train accuracy: ", accuracy_score(y_train, Y_prediction_train))
    print("test accuracy: ", accuracy_score(y_test, Y_prediction_test))
    
    '''
    dict = {"costs": costs,
            "Y_prediction_test": Y_prediction_test,
            "Y_prediction_train": Y_prediction_train,
            "w": w,
            "b": b,
            "learning_rate" : learning_rate,
            "num_iterations" : num_iterations}
    return dict


#d = model(X_train, X_test, y_train, y_test, num_iterations = 3000, learning_rate = 0.003, print_cost = True)

learning_rates = [5, 0.5, 0.03, 0.004, 0.0004]
models = {}
for i in learning_rates:
    print("learning rate is: ", i)
    models[i] = model(X_train, X_test, y_train, y_test, num_iterations = 1000, learning_rate = i, print_cost = False)
    print("---------------------------------------------------------")

for i in learning_rates:
    plt.plot(np.squeeze(models[i]["costs"]), label= str(models[i]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel("iterations (hundreds)")


legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()

