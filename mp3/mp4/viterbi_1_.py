"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""

from collections import Counter
import math
def viterbi_1(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    smoth_data = 0.001
    words = set()
    tags = set()
    tag_count = {}
    for sample in train:
    	for w, t in sample: 
    		tags.add(t)
    		words.add(w)
    		if t not in tag_count:
    			tag_count[t] = 0
    		tag_count[t] += 1

    start_t = {}
    for sample in train:
    	# print(sample)
    	if sample[1] not in start_t:
    		start_t[sample[1]] = 0
    	start_t[sample[1]] += 1
    start_p = {}

    for key, i in start_t.items():
    	start_p[str(key[0]) + "_" + str(key[1])] = math.log((i + smoth_data) / (len(train) + smoth_data*len(tags)))
    print(start_p)

# transition

    transition = []
    for sample in train:
    	for i in range(1, len(sample)):
    		transition.append((sample[i][1], sample[i-1][1]))    
    transition_ = dict(Counter(transition))
    transition_p = {}
    for start in tags:
    	total = 0
    	key_list = []
    	for key, i in transition_.items():
    		if key[0] == start:
    			key_list.append(key)
    			total = total + i
    	for key in key_list:
    		transition_p[str(key[0]) + "_" + str(key[1])] = math.log((transition_[key] + smoth_data)/ (total + smoth_data*(len(tags) + 1)))
# emission

    emission = {}
    for sample in train:
    	for w_t in sample:
    		if w_t not in emission:
    			emission[w_t] = 0
    		emission[w_t] += 1
    emission_p = {}
    for key, i in emission.items():
    	emission_p[str(key[0]) + "_" + str(key[1])] = math.log((emission[key] + smoth_data) / (tag_count[key[1]] + len(words) * smoth_data))

# uk_transition = math.log(smoth_data / (len(train) + smoth_data * len(tags)))
    uk_transition = {}
    for tag in tags:
    	total = 0
    	for key, i in transition_.items():
    		if key[0] == tag:
    			total = total + i
    	uk_transition[tag] = math.log(smoth_data / (total + smoth_data*(len(tags))))

    uk_emission = {}
    for tag in tags:
    	uk_emission[tag] = math.log(smoth_data / (tag_count[tag] + smoth_data * (len(words))))
# print(emission_p)
    print(uk_transition)
    print(uk_emission)
# predict
    predicts = []
    for sample in test:
        start_flag = 0
        predict = []
        for w in sample:
        	if start_flag == 0:
        		predict.append((w, "START"))
        		start_flag = 1
        		continue
        	if start_flag == 1:
        		start_flag = 2
        		b = max(start_p, key=start_p.get)
        		predict.append((w, b))
        		v = start_p[b]
        		continue
        	v_list = {}
        	for tag in tags:
        		key_t = str(tag) + "_" + str(b)
        		if key_t in transition_p:
        			trans_value = transition_p[key_t]
        		else:
        			trans_value = uk_transition[tag]
        		key_e = str(w) + "_" + str(tag)
        		if key_e in emission_p:
        			emiss_value = emission_p[key_e]
        		else:
        			emiss_value = uk_emission[tag]
        		v_new = v * trans_value * emiss_value
        		v_list[tag] = v_new
        	b = max(v_list, key=v_list.get)
        	v = v_list[b]
        	predict.append((w, b))
        # print(predict)
        predicts.append(predict)

    return predicts

