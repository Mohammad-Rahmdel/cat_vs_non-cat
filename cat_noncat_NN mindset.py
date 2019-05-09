"""                     
cat vs non-cat recognition using logistic regression and machine learning
Instructions:
https://github.com/Kulbear/deep-learning-coursera/blob/master/Neural%20Networks%20and%20Deep%20Learning/Logistic%20Regression%20with%20a%20Neural%20Network%20mindset.ipynb

to install PIL use this command:
sudo pip3 install -U Pillow

"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# to depict image[index] use the code below
# index = 2
# plt.imshow(train_set_x_orig[index])
# plt.show()
# print ("y = " + str(train_set_y[:,index]) + ", it's a '" + classes[np.squeeze(train_set_y[:,index])].decode("utf-8") +  "' picture.")

# print (train_set_x_orig.shape)
# print("m_train = " + str(train_set_x_orig.shape[0]))
# print("m_test = " + str(test_set_x_orig.shape[0]))
# print("num_px = " + str(train_set_x_orig.shape[1]))



train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T 
# a tricky way to vectorize a matrix of shape (a,b,c,d) to a matrix of shape (b*c*d, a)
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T 


#standardize our dataset.  scaling features from (0-255) to (0-1)
train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.


# activation function :
def sigmoid(z):
    return 1 / (1 + np.exp(-z)) 

def initialize_with_zeros(dim):
    w = np.zeros(shape=(dim, 1))
    b = 0
    return w, b

def propagate(w, b, X, Y):
    """
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    """

    m = X.shape[1]
    y_hat = sigmoid(np.dot(w.T,X) + b)
    db = (1/m)*(np.sum(y_hat - Y))
    dw = (1/m)*(np.dot(X,(y_hat - Y).T))
    cost = (-1/m) * (np.dot(Y,np.log(y_hat).T) + np.dot((1-Y),np.log(1-y_hat).T))
    cost = np.squeeze(cost)
    return cost, dw, db


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    
    Returns:
    w, b, costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    """

    costs = []
    for i in range(num_iterations):
        cost, dw, db = propagate(w, b, X, Y)
        w = w - learning_rate * dw  
        b = b - learning_rate * db
    
        if i % 100 == 0:
            costs.append(cost)
            # Print the cost every 100 training examples
            if print_cost:
                print ("Cost after iteration %i: %f" % (i, cost))

    return costs, w, b


def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''

    Y_hat = sigmoid(np.dot(w.T,X) + b)
    Y_prediction = np.round(Y_hat) #0 (if activation <= 0.5) or 1 (if activation > 0.5)
    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    Builds the logistic regression model by calling the function you've implemented previously
    
    Returns:
    d -- dictionary containing information about the model.
    """

    w, b = initialize_with_zeros(X_train.shape[0])
    costs, w, b = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    Y_hat_train = predict(w, b, X_train)
    Y_hat_test = 0
    Y_hat_test = predict(w, b, X_test)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_hat_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_hat_test - Y_test)) * 100))
    
    d = {"costs": costs,
         "Y_prediction_test": Y_hat_test, 
         "Y_prediction_train" : Y_hat_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d


import time
tic = time.time()
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1000, learning_rate = 0.005, print_cost=False)
toc = time.time()
print('{0}{1:.2f}{2}'.format("Calculation Time = ",(toc - tic)," seconds"))
result = test_set_y + d["Y_prediction_test"].astype(int) # 0 and 2 are correct predictions
result = result.tolist()
result = result[0]
n_true = sum(i==2 or i==0 for i in result)
print(str(n_true) + " true predictions from 50 samples.")

# plot the cost function by #iteration  
# costs = np.squeeze(d['costs'])
# plt.plot(costs)
# plt.ylabel('cost')
# plt.xlabel('iterations (100x)')
# plt.title("Learning rate =" + str(d["learning_rate"]))
# plt.show()




# to compare differenet learning rates use the code below
# learning_rates = [0.03, 0.01, 0.003, 0.001, 0.0001]
# models = {}
# for i in learning_rates:
#     print ("learning rate is: " + str(i))
#     models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)
#     print ("-------------------------------------------------------")

# for i in learning_rates:
#     plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

# plt.ylabel('cost')
# plt.xlabel('iterations')

# legend = plt.legend(loc='upper center', shadow=True)
# frame = legend.get_frame()
# frame.set_facecolor('0.90')
# plt.show()
