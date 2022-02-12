# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
import numpy as np
import math
from tqdm import tqdm
from collections import Counter
import reader

"""
This is the main entry point for MP4. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""


"""
  load_data calls the provided utility to load in the dataset.
  You can modify the default values for stemming and lowercase, to improve performance when
       we haven't passed in specific values for these parameters.
"""
 
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming is {stemming}")
    print(f"Lowercase is {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels



# Keep this in the provided template
def print_paramter_vals(laplace,pos_prior):
    print(f"Unigram Laplace {laplace}")
    print(f"Positive prior {pos_prior}")


"""
You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
Notice that we may pass in specific values for these parameters during our testing.
"""

def naiveBayes(train_set, train_labels, dev_set, laplace=1.0, pos_prior=0.5,silently=False):
    # Keep this in the provided template
    print_paramter_vals(laplace,pos_prior)

    frequency_dict = {}
    vocabulary = []
    for sample in train_set:
        vocabulary += sample

    index = np.where(train_labels == 0)
    index = index[0][0]
    pos_train = train_set[:index]
    neg_train = train_set[index:]
    pos_train_labels = train_labels[:index]
    neg_train_labels = train_labels[index:]



    vocabulary = set(vocabulary)
    pos_frequency_dict = dict.fromkeys(vocabulary, 0)
    neg_frequency_dict = dict.fromkeys(vocabulary, 0)

    total_words_pos = 0
    total_words_neg = 0

    for i in pos_train:
        for word in i:
            pos_frequency_dict[word] += 1
        total_words_pos += len(i)

    for i in neg_train:
        for word in i:
            neg_frequency_dict[word] += 1
        total_words_neg += len(i)

    unk_pos = math.log(laplace/(total_words_pos + laplace*(len(pos_frequency_dict))))
    unk_neg = math.log(laplace/(total_words_neg + laplace*(len(neg_frequency_dict))))

    pos_prob_dict = {}
    neg_prob_dict = {}

    for i in pos_frequency_dict:
        pos_prob_dict[i] = math.log((pos_frequency_dict[i] + laplace)/(total_words_pos + laplace*(len(pos_frequency_dict))))

    for i in neg_frequency_dict:
        neg_prob_dict[i] = math.log((neg_frequency_dict[i] + laplace)/(total_words_neg + laplace*(len(neg_frequency_dict))))

    yhats = []
    for doc in tqdm(dev_set,disable=silently):
        pos_prob = 0
        neg_prob = 0
        for word in doc:
            if word in pos_prob_dict:
                pos_prob += pos_prob_dict[word]
            else:
                pos_prob += unk_pos

            if word in neg_prob_dict:
                neg_prob += neg_prob_dict[word]
            else:
                neg_prob += unk_neg
        pos_prob += math.log(pos_prior)
        neg_prob += math.log(1-pos_prior)

        if pos_prob > neg_prob:
            yhats.append(1)
        else:
            yhats.append(0)

    return yhats



# Keep this in the provided template
def print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior):
    print(f"Unigram Laplace {unigram_laplace}")
    print(f"Bigram Laplace {bigram_laplace}")
    print(f"Bigram Lambda {bigram_lambda}")
    print(f"Positive prior {pos_prior}")


# main function for the bigrammixture model
def bigramBayes(train_set, train_labels, dev_set, unigram_laplace=1.0, bigram_laplace=1.0, bigram_lambda=0.05, pos_prior=0.5, silently=False):

    # Keep this in the provided template
    print(bigram_lambda)
    print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)

    index = np.where(train_labels == 0)
    index = index[0][0]
    pos_train = train_set[:index]
    neg_train = train_set[index:]
    pos_train_labels = train_labels[:index]
    neg_train_labels = train_labels[index:]

    pos_frequency_dict = {}
    neg_frequency_dict = {}

    total_words_pos = 0
    total_words_neg = 0

    for word in pos_train:
        for i in range(len(word) - 1):
            bigram = str(word[i]) + " " + str(word[i+1])
            if bigram in pos_frequency_dict:
                pos_frequency_dict[bigram] += 1
            else:
                pos_frequency_dict[bigram] = 1
        total_words_pos += (len(word) - 1)

    for word in neg_train:
        for i in range(len(word) - 1):
            bigram = str(word[i]) + " " + str(word[i+1])
            if bigram in neg_frequency_dict:
                neg_frequency_dict[bigram] += 1
            else:
                neg_frequency_dict[bigram] = 1
        total_words_neg += (len(word) - 1)

    unk_pos_1 = math.log(bigram_laplace/(total_words_pos + bigram_laplace*(len(pos_frequency_dict))))
    unk_neg_1 = math.log(bigram_laplace/(total_words_neg + bigram_laplace*(len(neg_frequency_dict))))

    pos_prob_dict_1 = {}
    neg_prob_dict_1 = {}

    for i in pos_frequency_dict:
        pos_prob_dict_1[i] = math.log((pos_frequency_dict[i] + bigram_laplace)/(total_words_pos + bigram_laplace*(len(pos_frequency_dict))))

    for i in neg_frequency_dict:
        neg_prob_dict_1[i] = math.log((neg_frequency_dict[i] + bigram_laplace)/(total_words_neg + bigram_laplace*(len(neg_frequency_dict))))

    frequency_dict = {}
    vocabulary = []
    for sample in train_set:
        vocabulary += sample

    index = np.where(train_labels == 0)
    index = index[0][0]
    pos_train = train_set[:index]
    neg_train = train_set[index:]
    pos_train_labels = train_labels[:index]
    neg_train_labels = train_labels[index:]



    vocabulary = set(vocabulary)
    pos_frequency_dict = dict.fromkeys(vocabulary, 0)
    neg_frequency_dict = dict.fromkeys(vocabulary, 0)

    total_words_pos = 0
    total_words_neg = 0

    for i in pos_train:
        for word in i:
            pos_frequency_dict[word] += 1
        total_words_pos += len(i)

    for i in neg_train:
        for word in i:
            neg_frequency_dict[word] += 1
        total_words_neg += len(i)

    unk_pos_2 = math.log(bigram_laplace/(total_words_pos + bigram_laplace*(len(pos_frequency_dict))))
    unk_neg_2 = math.log(bigram_laplace/(total_words_neg + bigram_laplace*(len(neg_frequency_dict))))

    pos_prob_dict_2 = {}
    neg_prob_dict_2 = {}

    for i in pos_frequency_dict:
        pos_prob_dict_2[i] = math.log((pos_frequency_dict[i] + bigram_laplace)/(total_words_pos + bigram_laplace*(len(pos_frequency_dict))))

    for i in neg_frequency_dict:
        neg_prob_dict_2[i] = math.log((neg_frequency_dict[i] + bigram_laplace)/(total_words_neg + bigram_laplace*(len(neg_frequency_dict))))

    # print(pos_prob_dict_1)
    # print(neg_prob_dict_1)
    yhats = []
    count  = 0
    for word in tqdm(dev_set,disable=silently):
        pos_prob_1 = 0
        neg_prob_1 = 0
        for i in range(len(word) - 1):
            bigram = str(word[i]) + " " + str(word[i+1])
            if bigram in pos_prob_dict_1:
                pos_prob_1 += pos_prob_dict_1[bigram]
            else:
                pos_prob_1 += unk_pos_1

            if bigram in neg_prob_dict_1:
                neg_prob_1 += neg_prob_dict_1[bigram]
            else:
                neg_prob_1 += unk_neg_1
        pos_prob_1 += math.log(pos_prior)
        neg_prob_1 += math.log(1-pos_prior)

        pos_prob_2 = 0
        neg_prob_2 = 0
        for i in word:
            if i in pos_prob_dict_2:
                pos_prob_2 += pos_prob_dict_2[i]
            else:
                pos_prob_2 += unk_pos_2

            if i in neg_prob_dict_2:
                neg_prob_2 += neg_prob_dict_2[i]
            else:
                neg_prob_2 += unk_neg_2
        pos_prob_2 += math.log(pos_prior)
        neg_prob_2 += math.log(1-pos_prior)

        pos_prob = bigram_lambda * pos_prob_1 + ( 1 - bigram_lambda) * pos_prob_2
        neg_prob = bigram_lambda * neg_prob_1 + ( 1 - bigram_lambda) * neg_prob_2

        if pos_prob > neg_prob:
            yhats.append(1)
        else:
            yhats.append(0)



    return yhats


