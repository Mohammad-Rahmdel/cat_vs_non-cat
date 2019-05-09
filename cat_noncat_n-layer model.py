import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_app_utils_v2 import *

import functions


plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


np.random.seed(1)



train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.


print("xxx")


def two_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    """
    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.
    
    Arguments:
    X (n_x, number of examples)
    Y (containing 0 if cat, 1 if non-cat):(1, number of examples)
    layers_dims:(n_x, n_h, n_y)
    print_cost -- If set to True, this will print the cost every 100 iterations 
    
    Returns:
    parameters -- a dictionary containing W1, W2, b1, and b2
    """

    
    np.random.seed(1) 
    costs = [] 
    # parameters = initialize_parameters(n_x, n_h, n_y) 
    parameters = initialize_parameters_deep(layers_dims)

    for i in range(num_iterations):
        
        Y_hat, caches = L_model_forward(X, parameters)
        
        
        if i%100 == 0:
            if print_cost==True:
                cost = compute_cost(Y_hat, Y)
                costs.append(cost)
                print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))

        grads = L_model_backward(Y_hat, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)
    

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    if print_cost:
        plt.show()
    
    return parameters


def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False): #lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    """
    np.random.seed(1) 
    costs = [] 
    # parameters = initialize_parameters(n_x, n_h, n_y) 
    parameters = initialize_parameters_deep(layers_dims)

    for i in range(0, num_iterations):
        
        Y_hat, caches = L_model_forward(X, parameters)
        
        
        if i%100 == 0:
            if print_cost==True:
                cost = compute_cost(Y_hat, Y)
                costs.append(cost)
                print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))

        grads = L_model_backward(Y_hat, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)
    

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    if print_cost:
        plt.show()
    
    return parameters
   

layers_dims = [12288, 20, 7, 5, 1]
parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations=2500, print_cost=False)

pred_train = predict(train_x, train_y, parameters)
pred_test = predict(test_x, test_y, parameters)


# print_mislabeled_images(classes, test_x, test_y, pred_test)  #doesn't work




# Predict your image :))

# my_image = "joey.jpeg" # change this to the name of your image file 
# my_label_y = [0] # the true class of your image (1 -> cat, 0 -> non-cat)

# num_x = 64

# fname = "images/" + my_image    # copy your image in this directory 
# image = np.array(ndimage.imread(fname, flatten=False))
# my_image = scipy.misc.imresize(image, size=(num_x, num_x)).reshape((num_x*num_x*3,1))
# my_predicted_image = predict(my_image, my_label_y, parameters)

# plt.imshow(image)
# print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
# plt.show()