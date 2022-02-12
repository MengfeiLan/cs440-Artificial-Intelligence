"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""
import math
import numpy as np
def viterbi_1(train, test):

	smooth_data = 0.001
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
		if sample[1] not in start_t:
			start_t[sample[1]] = 0
		start_t[sample[1]] += 1

	start_word = {}
	for sample in train:
		if sample[1][0] not in start_word:
			start_word[sample[1][0]] = 0
		start_word[sample[1][0]] += 1

	start_p = {}
	for key, i in start_t.items():
		start_p[key[0] + "_" + key[1]] = math.log((i + smooth_data) / (start_word[key[0]] + smooth_data*len(tags)))
	uk_start = math.log((smooth_data) / (len(train) + smooth_data*len(tags)))

# transition
	transition = {}
	transition_target = {}

	for sample in train:
		for i in range(1, len(sample) - 2):
			if (sample[i][1], sample[ i + 1 ][1]) not in transition:
				transition[(sample[i][1], sample[i+1][1])] = 0
			transition[(sample[i][1], sample[i+1][1])] += 1
			if (sample[i][1]) not in transition_target:
				transition_target[sample[i][1]] = 0
			transition_target[sample[i][1]] += 1

	transition_p = {}
	for key, i in transition.items():
		transition_p[key[0] + "_" + key[1]] = math.log((i + smooth_data) / (transition_target[key[0]] + smooth_data*len(tags)))

	uk_transition = math.log((smooth_data) / (len(train) + smooth_data*len(tags)))

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
		emission_p[key[0] + "_" + key[1]] = math.log((i + smooth_data) / (word_count[key[0]] + smooth_data*len(tags)))

	uk_emission = math.log((smooth_data) / (len(words) + smooth_data*len(tags)))

	predicts = []
	for sample in test:
		transfer(start_p, emission_p, transition_p, tags, uk_start, uk_emission, uk_transition, sample[1:-1])
		



	return []

def transfer(start_p, emission_p, transition_p, tags, uk_start, uk_emission, uk_transition, sentence):
	array = np.zeros((len(sentence), len(tags)))
	list_total = {}
	for i in range(len(sentence)):
		if i == 0:
			v_list = {}
			for j in range(len(tags)):
				start_value = str(sentence[i]) + "_" + str(tags[j])
				if start_value in start_p:
					v_list[("START", tags[j])] = start_p[start_value]
				else:
					v_list[("START", tags[j])] = uk_start
			list_total{i} = v_list
			continue
		v_list = {}
		for j in range(len(tags)):
			emission_value = str(sentence[i] + "_" + str(tags[j]))
			if emission_value in emission_p:
				emission_prob = emission_p[emission_value]
			else:
				emission_prob = uk_emission
			for k in range(len(tags)):
				transition_value = str(tags[k]) + "_" + str(tags[j])
				if transition_value in transition_p:
					transition_prob = transition_p[transition_value]
				else:
					transition_prob = uk_transition


