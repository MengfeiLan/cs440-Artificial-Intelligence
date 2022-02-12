"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""
import math
import numpy as np
def viterbi_1(train, test):

	smooth_data = 0.000000001
	tags = set()
	tag_count = {}
	words = set()
	word_count = {}

	for sample in train:
		for w, t in sample: 
			tags.add(t)
			words.add(w)
			if t not in tag_count:
				tag_count[t] = 0
			if w not in word_count:
				word_count[w] = 0
			tag_count[t] += 1
			word_count[w] += 1
	tags = list(tags)
# find start tagging
	start_t = {}
	for sample in train:
		if sample[1][1] not in start_t:
			start_t[sample[1][1]] = 0
		start_t[sample[1][1]] += 1


	start_prob = {}
	for key, i in start_t.items():
		start_prob[key] = math.log((i + smooth_data) / (len(train) + smooth_data*len(tags)))
	uk_start = math.log((smooth_data) / (len(train) + smooth_data*len(tags)))

# transition
	transition = {}
	transition_target = {}

	for sample in train:
		for i in range(1, len(sample) - 2):
			if (sample[i + 1][1], sample[ i ][1]) not in transition:
				transition[(sample[i + 1][1], sample[i][1])] = 0
			transition[(sample[i + 1][1], sample[i][1])] += 1
			if (sample[i][1]) not in transition_target:
				transition_target[sample[i][1]] = 0
			transition_target[sample[i][1]] += 1

	transition_p = {}
	for key, i in transition.items():
		transition_p[key[0] + "_" + key[1]] = math.log((i + smooth_data) / (transition_target[key[1]] + smooth_data*len(tags)))

	uk_transition = {}
	for key, i in tag_count.items():
		uk_transition[key] = math.log((smooth_data) / (tag_count[key] + smooth_data*len(tags)))
# emission
	emission = {}
	emssion_target = {}

	for sample in train:
		for i in range(1, len(sample) - 1):
			if sample[i] not in emission:
				emission[sample[i]] = 0
			emission[sample[i]] += 1

	emission_p = {}
	for key, i in emission.items():
		emission_p[key[0] + "_" + key[1]] = math.log((i + smooth_data) / (word_count[key[1]] + smooth_data*len(tags)))

	uk_emission = math.log((smooth_data) / (len(words) + smooth_data*len(tags)))

# start
	
	start_p = {}

	for key, i in emission_p.items():

		tag = key.split("_")
		tag = tag[-1]
		if tag not in start_prob:
			start_p[key] = uk_start
		else:
			start_p[key] = start_prob[tag] + emission_p[key]

	predicts = []
	for sample in test:
		more = transfer(start_p, emission_p, transition_p, tags, uk_start, uk_emission, uk_transition, sample[1:-1])
		more.insert(0, (sample[0], "START"))
		more.append((sample[-1], "END"))
		predicts.append(more)

	return predicts
 



def transfer(start_p, emission_p, transition_p, tags, uk_start, uk_emission, uk_transition, sentence):
	matrix = [[None for c in range(len(tags))] for r in range(len(sentence))]
	pos_list = [[None for c in range(len(tags))] for r in range(len(sentence))]
	start_flag = True

	for i in range(len(sentence)):
		v_list = {}
		if start_flag == True:
			for j in range(len(tags)):
				start_value = str(sentence[i]) + "_" + str(tags[j])
				if start_value in start_p:
					v = abs(start_p[start_value])
				else: 
					v = uk_start
				b = tags[j]
				matrix[i][j] = v
				pos_list[i][j] = b
			start_flag = False
			continue
		for j in range(len(tags)):
			v_list = {}
			for k in range(len(tags)):
				transition_value = str(tags[j]) + "_" + str(tags[k])
				if transition_value in transition_p:
					transition_prob = transition_p[transition_value]
				else: 
					transition_prob = uk_emission
				emission_value = str(sentence[i]) + "_" + str(tags[j])
				if emission_value in emission_p:
					emission_prob = emission_p[emission_value]
				else: 
					emission_prob = uk_emission

				v_list[(pos_list[i-1][k] + "_" + str(tags[k]) + "_" + str(tags[j]))] = abs(matrix[i-1][k] * transition_prob * emission_prob)
			b = min(v_list, key = v_list.get)
			v = v_list[b]
			matrix[i][j] = v
			pos_list[i][j] = b
	min_value = min(matrix[-1])
	min_index = matrix[-1].index(min_value)
	end = pos_list[-1][min_index]
	results = end.split("_")
	result_list = []
	for i in range(int((len(results))/2)):
		result_list.append((sentence[i], results[2*i]))

	result_list.append((sentence[-1], results[-1]))

	return result_list


