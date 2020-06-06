# tf_idf_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Modified by Jaewook Yeom 02/02/2020

"""
This is the main entry point for the Extra Credit Part of this MP. You should only modify code
within this file for the Extra Credit Part -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np
import math
from collections import Counter
import time
import heapq



def compute_tf_idf(train_set, train_labels, dev_set):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    Return: A list containing words with the highest tf-idf value from the dev_set documents
            Returned list should have same size as dev_set (one word from each dev_set document)
    """



    # TODO: Write your code here

    ret = []
    term_cnt = Counter()        # counting no. of documents that contain the given word in train_set

    total_doc = 0
    for each_doc in train_set:
        total_doc += 1          # counting total no. of documents in train_set
        flag_cnt = Counter()    # keep track of whether the word has been counted in one document
        for each_word in each_doc:
            if flag_cnt[each_word] == 0:     # if the word have not been counted before
                term_cnt[each_word] += 1

    for each_doc in dev_set:
        tf_idf_q = []         # priority queue for storing (tf_idf, word)
        word_count = 0        # counting total no. of words in a document
        freq_cnt = Counter()  # counting frequency for each word in the document

        for each_word in each_doc:
            word_count += 1
            freq_cnt[each_word] += 1

        for each_term in freq_cnt:
            neg_tf_idf = 0 - freq_cnt[each_term] / word_count * math.log(total_doc / (1 + term_cnt[each_term]))
            heapq.heappush(tf_idf_q, (neg_tf_idf, each_term))

        ret.append(tf_idf_q[0][1])


    # return list of words (should return a list, not numpy array or similar)
    return ret