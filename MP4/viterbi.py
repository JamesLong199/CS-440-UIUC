"""
This is the main entry point for MP4. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

from collections import Counter
import copy
import math

def baseline(train, test):
    '''
    TODO: implement the baseline algorithm. This function has time out limitation of 1 minute.
    input:  training data (list of sentences, with tags on the words)
            E.g. [[(word1, tag1), (word2, tag2)...], [(word1, tag1), (word2, tag2)...]...]
            test data (list of sentences, no tags on the words)
            E.g  [[word1,word2,...][word1,word2,...]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g. [[(word1, tag1), (word2, tag2)...], [(word1, tag1), (word2, tag2)...]...]
    '''
    predicts = []
    word_tag_cnt = Counter()      # counting no. (word w, tag T) occurrence
    word_cnt = Counter()          # counting no. (word w) occurrence
    tag_cnt = Counter()           # counting no. (tag T) occurrence

    for each_sentence in train:
        for each_pair in each_sentence:
            word_tag_cnt[each_pair] += 1    # increase the count of (word w, tag T)
            word_cnt[each_pair[0]] += 1     # increase the count of (word w)
            tag_cnt[each_pair[1]] += 1      # increase the count of (tag T)

    max_tag = tag_cnt.most_common(1)[0][0]

    for each_sentence in test:
        ret_sentence = []                   # sentence to be returned
        for each_word in each_sentence:
            max_ = ["", 0]                  # find argmax P(T | W)
            for each_tag in tag_cnt:        # find argmax P(T | W)

                curr_count = word_tag_cnt[(each_word, each_tag)]
                if curr_count >= max_[1]:
                    max_[0] = each_tag
                    max_[1] = curr_count

            if max_[1] > 0:                 # if not an unseen word
                ret_sentence.append((each_word, max_[0]))
            else:                           # if an unseen word
                ret_sentence.append((each_word, max_tag))   # assign tag seen the most often

        predicts.append(ret_sentence)

    return predicts


def viterbi_p1(train, test):
    '''
    TODO: implement the simple Viterbi algorithm. This function has time out limitation for 3 mins.
    input:  training data (list of sentences, with tags on the words)
            E.g. [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words)
            E.g [[word1,word2...]]
    output: list of sentences with tags on the words
            E.g. [[(word1, tag1), (word2, tag2)...], [(word1, tag1), (word2, tag2)...]...]
    '''

    predicts = []
    k = 0.00001                        # laplace smoothing constant

    tag_pair_cnt = Counter()           # (tag a, tag b), no. of occurrence
    tag_cnt = Counter()                # no. of words with tag T
    word_cnt = Counter()               # no. of words w
    word_tag_cnt = Counter()           # (word w, tag T), no. of occurrence

    N_word = 0

    for each_sentence in train:                        # training AI
        prev_tag = ""                                  # storing the previous tag
        for each_pair in each_sentence:
            N_word += 1
            curr_word = each_pair[0]
            curr_tag = each_pair[1]
            if prev_tag != "":
                curr_tag_pair = (prev_tag, curr_tag)
                tag_pair_cnt[curr_tag_pair] += 1       # increment occurrence of (Tag a, Tag b)

            tag_cnt[curr_tag] += 1                     # increment occurrence of (Tag T)
            word_tag_cnt[(curr_word, curr_tag)] += 1   # increment occurrence of (word w, Tag T)
            word_cnt[curr_word] += 1                   # increment occurrence of word w

            prev_tag = curr_tag                        # update prev_tag

    X_tag = len(tag_cnt)                               # number of unique tags in train set
    X_word = len(word_cnt)                             # number of unique words in train set

    for each_sentence in test:                         # inference
        ret_sentence = []
        trellis = []
        for i in range(0, len(each_sentence)):         # initialize the trellis
            col = []
            for each_tag in tag_cnt:
                col.append([0, each_tag, (-1, -1)])    # each node in trellis: (node value, tag value, back_ptr)
            trellis.append(col)

        word_idx = 0                                   # reference word in the trellis
        for each_word in each_sentence:
            if word_idx != 0:                          # if not the initial word
                for tag_idx in range(0, X_tag):        # loop through N current tags
                    each_tag = trellis[word_idx][tag_idx][1]
                    max_path_cost = float("-inf")
                    max_tag_idx = -1
                    for prev_tag_idx in range(0, X_tag):       # loop through N previous tags for each current tag
                        prev_tag = trellis[word_idx-1][prev_tag_idx][1]

                        # compute P(Tag t | Tag t-1)
                        p_tag_cond = math.log((tag_pair_cnt[(prev_tag, each_tag)] + k) / (tag_cnt[prev_tag] + k * X_tag))
                        # compute P(word t | Tag t)
                        p_word_tag = math.log((word_tag_cnt[(each_word, each_tag)] + k) / (tag_cnt[each_tag] + k * (X_word+1)))

                        # compute total path cost of current node
                        total_path_cost = p_tag_cond + p_word_tag + trellis[word_idx-1][prev_tag_idx][0]
                        if total_path_cost >= max_path_cost:
                            max_path_cost = total_path_cost
                            max_tag_idx = prev_tag_idx

                    trellis[word_idx][tag_idx][0] = max_path_cost                        # update current node value
                    trellis[word_idx][tag_idx][2] = (word_idx - 1, max_tag_idx)          # update current node back_ptr

            else:                                      # initial word (word_idx == 0)
                for tag_idx in range(0, X_tag):
                    each_tag = trellis[0][tag_idx][1]
                    # compute initial probability
                    p_tag = math.log((tag_cnt[each_tag] + k) / (N_word + k * (X_word + 1)))
                    p_word_tag = math.log((word_tag_cnt[(each_word, each_tag)] + k) / (tag_cnt[each_tag] + k * (X_word + 1)))
                    trellis[0][tag_idx][0] = p_tag + p_word_tag

            word_idx += 1

        max_final_val = float("-inf")
        max_final_idx = -1
        for j in range(0, X_tag):                     # find the last node of trellis
            curr_final_val = trellis[len(each_sentence)-1][j][0]
            if curr_final_val >= max_final_val:
                max_final_val = curr_final_val
                max_final_idx = j

        count = len(each_sentence) - 1
        curr_node = trellis[count][max_final_idx]
        while count >= 0:                             # from last word, backtrack to the first word
            ret_sentence.append((each_sentence[count], curr_node[1]))
            back_ptr = curr_node[2]
            curr_node = trellis[back_ptr[0]][back_ptr[1]]
            count -= 1

        predicts.append(ret_sentence[::-1])

    return predicts

def viterbi_p2(train, test):
    '''
    TODO: implement the optimized Viterbi algorithm. This function has time out limitation for 3 mins.
    input:  training data (list of sentences, with tags on the words)
            E.g. [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words)
            E.g [[word1,word2...]]
    output: list of sentences with tags on the words
            E.g. [[(word1, tag1), (word2, tag2)...], [(word1, tag1), (word2, tag2)...]...]
    '''

    predicts = []
    k = 0.00001  # laplace smoothing constant

    tag_pair_cnt = Counter()  # (tag a, tag b), no. of occurrence
    tag_cnt = Counter()  # no. of words with tag T
    word_cnt = Counter()  # no. of words w
    word_tag_cnt = Counter()  # (word w, tag T), no. of occurrence

    N_word = 0

    for each_sentence in train:  # training AI
        prev_tag = ""  # storing the previous tag
        for each_pair in each_sentence:
            N_word += 1
            curr_word = each_pair[0]
            curr_tag = each_pair[1]
            if prev_tag != "":
                curr_tag_pair = (prev_tag, curr_tag)
                tag_pair_cnt[curr_tag_pair] += 1  # increment occurrence of (Tag a, Tag b)

            tag_cnt[curr_tag] += 1  # increment occurrence of (Tag T)
            word_tag_cnt[(curr_word, curr_tag)] += 1  # increment occurrence of (word w, Tag T)
            word_cnt[curr_word] += 1  # increment occurrence of word w

            prev_tag = curr_tag  # update prev_tag

    X_tag = len(tag_cnt)  # number of unique tags in train set
    X_word = len(word_cnt)  # number of unique words in train set

    hapax = []                                 # create a list of hapax words
    for each_word in word_cnt:
        if word_cnt[each_word] == 1:
            hapax.append(each_word)

    N_hapax = len(hapax)
    tag_hapax_dict = {}

    for each_tag in tag_cnt:
        tag_hapax_count = 0
        for each_hapax in hapax:
            tag_hapax_count += word_tag_cnt[(each_hapax, each_tag)]
        p_tag_l_hapax = (tag_hapax_count + k) / (N_hapax + k * X_tag)
        tag_hapax_dict[each_tag] = p_tag_l_hapax

    for each_sentence in test:  # inference
        ret_sentence = []
        trellis = []
        for i in range(0, len(each_sentence)):  # initialize the trellis
            col = []
            for each_tag in tag_cnt:
                col.append([0, each_tag, (-1, -1)])  # each node in trellis: (node value, tag value, back_ptr)
            trellis.append(col)

        word_idx = 0  # reference word in the trellis
        for each_word in each_sentence:
            if word_idx != 0:  # if not the initial word
                for tag_idx in range(0, X_tag):  # loop through N current tags
                    each_tag = trellis[word_idx][tag_idx][1]
                    max_path_cost = float("-inf")
                    max_tag_idx = -1
                    for prev_tag_idx in range(0, X_tag):  # loop through N previous tags for each current tag
                        prev_tag = trellis[word_idx - 1][prev_tag_idx][1]

                        # compute P(Tag t | Tag t-1)
                        p_tag_cond = math.log(
                            (tag_pair_cnt[(prev_tag, each_tag)] + k) / (tag_cnt[prev_tag] + k * X_tag))
                        # compute P(word t | Tag t)
                        k_n = tag_hapax_dict[each_tag] * k
                        p_word_tag = math.log(
                            (word_tag_cnt[(each_word, each_tag)] + k_n) / (tag_cnt[each_tag] + k_n * (X_word + 1)))

                        # compute total path cost of current node
                        total_path_cost = p_tag_cond + p_word_tag + trellis[word_idx - 1][prev_tag_idx][0]
                        if total_path_cost >= max_path_cost:
                            max_path_cost = total_path_cost
                            max_tag_idx = prev_tag_idx

                    trellis[word_idx][tag_idx][0] = max_path_cost  # update current node value
                    trellis[word_idx][tag_idx][2] = (word_idx - 1, max_tag_idx)  # update current node back_ptr

            else:  # initial word (word_idx == 0)
                for tag_idx in range(0, X_tag):
                    each_tag = trellis[0][tag_idx][1]
                    # compute initial probability
                    p_tag = math.log((tag_cnt[each_tag] + k) / (N_word + k * (X_word + 1)))
                    k_n = tag_hapax_dict[each_tag] * k
                    p_word_tag = math.log(
                        (word_tag_cnt[(each_word, each_tag)] + k_n) / (tag_cnt[each_tag] + k_n * (X_word + 1)))
                    trellis[0][tag_idx][0] = p_tag + p_word_tag

            word_idx += 1

        max_final_val = float("-inf")
        max_final_idx = -1
        for j in range(0, X_tag):  # find the last node of trellis
            curr_final_val = trellis[len(each_sentence) - 1][j][0]
            if curr_final_val >= max_final_val:
                max_final_val = curr_final_val
                max_final_idx = j

        count = len(each_sentence) - 1
        curr_node = trellis[count][max_final_idx]
        while count >= 0:  # from last word, backtrack to the first word
            ret_sentence.append((each_sentence[count], curr_node[1]))
            back_ptr = curr_node[2]
            curr_node = trellis[back_ptr[0]][back_ptr[1]]
            count -= 1

        predicts.append(ret_sentence[::-1])

    return predicts

