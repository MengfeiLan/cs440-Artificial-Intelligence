"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""
from collections import Counter
def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]

    '''

    word_dict = {}
    tag_all = {}
    for sample in train:
    	for w_t in sample:
    		word, tag = w_t
    		if word not in word_dict:
    			word_dict[word] = {}
    		if tag not in word_dict[word]:
    			word_dict[word][tag] = 0
    		word_dict[word][tag] += 1
    		if tag not in tag_all:
    			tag_all[tag] = 0
    		tag_all[tag] += 1

    max_tag_for_hapax = max(tag_all, key=tag_all.get)
    result = []

    for sample in test:
    	predict = []
    	for w in sample:
    		if w in word_dict:
    			tag = max(word_dict[w], key = word_dict[w].get)
    			predict.append((w, tag))
    		else: 
    			predict.append((w, max_tag_for_hapax))
    	result.append(predict)

    return result
