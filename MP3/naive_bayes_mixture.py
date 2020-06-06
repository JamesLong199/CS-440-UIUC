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
This is the main entry point for Part 2 of this MP. You should only modify code
within this file for Part 2 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""


import numpy as numpy
import math
import copy
from collections import Counter





def naiveBayesMixture(train_set, train_labels, dev_set, bigram_lambda,unigram_smoothing_parameter, bigram_smoothing_parameter, pos_prior):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    bigram_lambda - float between 0 and 1

    unigram_smoothing_parameter - Laplace smoothing parameter for unigram model (between 0 and 1)

    bigram_smoothing_parameter - Laplace smoothing parameter for bigram model (between 0 and 1)

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
    k = unigram_smoothing_parameter

    bi_cnt_pos = Counter()
    bi_cnt_neg = Counter()
    bi_N_pos = 0
    bi_N_neg = 0
    bi_X_pos = 0
    bi_X_neg = 0
    bi_k = bigram_smoothing_parameter

    index = 0
    last_word = ""                       # for combining bigrams

    pos_count = 0
    neg_count = 0

    for each_doc in train_set:
        if train_labels[index] == 1:
            pos_count += 1
        else:
            neg_count += 1

        for each_word in each_doc:
            if train_labels[index] == 1:  # if positive review
                if cnt_pos[each_word] == 0:  # if a new type
                    X_pos += 1  # increase number of types
                cnt_pos[each_word] += 1  # increase count
                N_pos += 1  # increase total number of token

                if last_word != "":
                    bigram = last_word + " " + each_word  # form a bigram
                    if bi_cnt_pos[bigram] == 0:  # if a new type of bigram
                        bi_X_pos += 1     # increase number of types of bigram
                    bi_cnt_pos[bigram] += 1  # increase count of bigram
                    bi_N_pos += 1   # increase total number of bigram token

            else:  # if negative review
                if cnt_neg[each_word] == 0:
                    X_neg += 1
                cnt_neg[each_word] += 1
                N_neg += 1

                if last_word != "":
                    bigram = last_word + " " + each_word
                    if bi_cnt_neg[bigram] == 0:
                        bi_X_neg += 1
                    bi_cnt_neg[bigram] += 1
                    bi_N_neg += 1
            del last_word
            last_word = copy.deepcopy(each_word)

        last_word = ""
        index += 1

    print("no. of positive review: ", pos_count)
    print("no. of negative review: ", neg_count)

    pre_pos_count = 0
    pre_neg_count = 0

    for each_doc in dev_set:
        pos_prob = math.log(pos_prior)
        neg_prob = math.log(1 - pos_prior)
        bi_pos_prob = math.log(pos_prior)
        bi_neg_prob = math.log(1 - pos_prior)

        for each_word in each_doc:
            pos_likelihood = (cnt_pos[each_word] + k) / (N_pos + k * X_pos)
            neg_likelihood = (cnt_neg[each_word] + k) / (N_neg + k * X_neg)
            pos_prob += math.log(pos_likelihood)
            neg_prob += math.log(neg_likelihood)

            if last_word != "":
                bigram = last_word + " " + each_word
                bi_pos_likelihood = (bi_cnt_pos[bigram] + bi_k) / (bi_N_pos + bi_k * bi_X_pos)
                bi_neg_likelihood = (bi_cnt_neg[bigram] + bi_k) / (bi_N_neg + bi_k * bi_X_neg)
                bi_pos_prob += math.log(bi_pos_likelihood)
                bi_neg_prob += math.log(bi_neg_likelihood)

            del last_word
            last_word = copy.deepcopy(each_word)

        last_word = ""

        label = 0
        total_pos_prob = (1 - bigram_lambda) * pos_prob + bigram_lambda * bi_pos_prob
        total_neg_prob = (1 - bigram_lambda) * neg_prob + bigram_lambda * bi_neg_prob
        if total_pos_prob > total_neg_prob:
            label = 1
        if label == 0:
            pre_neg_count += 1
        else:
            pre_pos_count += 1

        dev_labels.append(label)

    print("predicted no. of positive review: ", pre_pos_count)
    print("predicted no. of negative review: ", pre_neg_count)

    # return predicted labels of development set (make sure it's a list, not a numpy array or similar)
    return dev_labels

