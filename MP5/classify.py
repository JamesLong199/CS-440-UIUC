# classify.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018
# Extended by Daniel Gonzales (dsgonza2@illinois.edu) on 3/11/2020

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.

train_set - A Numpy array of 32x32x3 images of shape [7500,3072].
            This can be thought of as a list of 7500 vectors that are each
            3072 dimensional.  We have 3072 dimensions because there are
            each image is 32x32 and we have 3 color channels.
            So 32*32*3 = 3072. RGB values have been scaled to range 0-1.

train_labels - List of labels corresponding with images in train_set
example: Suppose I had two images [X1,X2] where X1 and X2 are 3072 dimensional vectors
         and X1 is a picture of a dog and X2 is a picture of an airplane.
         Then train_labels := [1,0] because X1 contains a picture of an animal
         and X2 contains no animals in the picture.

dev_set - A Numpy array of 32x32x3 images of shape [2500, 3072].
          It is the same format as train_set
"""

import numpy as np
import math

def trainPerceptron(train_set, train_labels, learning_rate, max_iter):
    # TODO: Write your code here
    # return the trained weight and bias parameters
    w = np.zeros(len(train_set[0]))      # weight vector(w1, w2, ..., w3072, b), initialized to 0
    b = 0

    iter_count = max_iter             # counting no. of iteration(epoch)
    while iter_count > 0:
        iter_count -= 1

        for i in range(len(train_set)):
            x = np.array(train_set[i])          # feature vector(x1, x2, ..., x3072)
            label = train_labels[i]

            f = np.dot(w, x)
            if f + b > 0:    # predicted 1
                if label == 0:        # label -1
                    w = np.subtract(w, np.multiply(x, learning_rate))
                    b -= learning_rate

            else:                      # predicted 0
                if label == 1:         # label 1
                    w = np.add(w, np.multiply(x, learning_rate))
                    b += learning_rate

    return w, b

def classifyPerceptron(train_set, train_labels, dev_set, learning_rate, max_iter):
    # TODO: Write your code here
    # Train perceptron model and return predicted labels of development set
    ret_label = []
    train_param = trainPerceptron(train_set, train_labels, learning_rate, max_iter)
    w = np.array(train_param[0])
    b = train_param[1]

    for i in range(len(dev_set)):
        x = np.array(dev_set[i])
        f = np.dot(w, x)
        if f + b > 0:          # predict 1
            ret_label.append(1)
        else:                  # predict 0
            ret_label.append(0)
    return np.array(ret_label)

def sigmoid(x):
    # TODO: Write your code here
    # return output of sigmoid function given input x
    return 1/(1 + 1/math.exp(x))


def trainLR(train_set, train_labels, learning_rate, max_iter):
    # TODO: Write your code here
    # return the trained weight and bias parameters
    w = np.zeros(len(train_set[0]))  # weight vector(w1, w2, ..., w3072, b), initialized to 0
    b = 0

    iter_count = max_iter  # counting no. of iteration(epoch)
    while iter_count > 0:
        iter_count -= 1
        N = len(train_set)
        dL_w = np.zeros(len(train_set[0]))
        dL_b = 0

        for i in range(N):
            x = np.array(train_set[i])  # feature vector(x1, x2, ..., x3072)
            f = sigmoid(np.dot(w, x) + b)
            y_i = train_labels[i]
            # first_term = y_i * 1/f * (1/4 - math.pow((f-1/2), 2))
            first_term = y_i * 1 / f * (1 / 4 - (f - 1 / 2) * (f - 1 / 2))
            # second_term = (1 - y_i) * 1/(1-f) * (1/4 - math.pow((f-1/2), 2))
            second_term = (1 - y_i) * 1 / (1 - f) * (1 / 4 - (f - 1 / 2) * (f - 1 / 2))
            dL_b += 1/N * (second_term - first_term)
            dL_w += np.dot(x, 1/N * (second_term - first_term))

        w = np.subtract(w, dL_w * learning_rate)
        b -= dL_b * learning_rate

    return w, b

def classifyLR(train_set, train_labels, dev_set, learning_rate, max_iter):
    # TODO: Write your code here
    # Train LR model and return predicted labels of development set
    ret_label = []
    train_param = trainLR(train_set, train_labels, learning_rate, max_iter)
    w = np.array(train_param[0])
    b = train_param[1]

    for i in range(len(dev_set)):
        x = np.array(dev_set[i])
        f = sigmoid(np.dot(w, x) + b)
        if f >= 0.5:  # predict 1
            ret_label.append(1)
        else:  # predict 0
            ret_label.append(0)
    return np.array(ret_label)

def classifyEC(train_set, train_labels, dev_set, k):
    # Write your code here if you would like to attempt the extra credit
    return []
