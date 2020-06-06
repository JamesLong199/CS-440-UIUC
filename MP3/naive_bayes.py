# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Modified by Jaewook Yeom 02/02/2020

"""
This is the main entry point for Part 1 of MP3. You should only modify code
within this file for Part 1 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as numpy
import math
from collections import Counter


def naiveBayes(train_set, train_labels, dev_set, smoothing_parameter, pos_prior):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    smoothing_parameter - The smoothing parameter you provided with --laplace (1.0 by default)

    pos_prior - positive prior probability (between 0 and 1)
    """

    # TODO: Write your code here
    
    dev_labels = []
    cnt_pos = Counter()
    cnt_neg = Counter()
    N_pos = 0
    N_neg = 0
    X_pos = 0
    X_neg = 0
    index = 0
    k = smoothing_parameter

    for each_doc in train_set:
        for each_word in each_doc:
            if train_labels[index] == 1:             # if positive review
                if cnt_pos[each_word] == 0:          # if a new type
                    X_pos += 1                       # increase number of types
                cnt_pos[each_word] += 1              # increase count
                N_pos += 1                           # increase total number of token

            else:                                    # if negative review
                if cnt_neg[each_word] == 0:
                    X_neg += 1
                cnt_neg[each_word] += 1
                N_neg += 1
        index += 1

    for each_doc in dev_set:
        pos_prob = math.log(pos_prior)
        neg_prob = math.log(1 - pos_prior)
        for each_word in each_doc:
            pos_likelihood = (cnt_pos[each_word] + k) / (N_pos + k * X_pos)
            neg_likelihood = (cnt_neg[each_word] + k) / (N_neg + k * X_neg)
            pos_prob += math.log(pos_likelihood)
            neg_prob += math.log(neg_likelihood)

        label = 0
        if pos_prob > neg_prob:
            label = 1

        dev_labels.append(label)

    # return predicted labels of development set (make sure it's a list, not a numpy array or similar)
    return dev_labels