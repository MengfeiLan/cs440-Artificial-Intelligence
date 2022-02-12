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


	initial = {}
	for key, i in start_t.items():
		initial[key] = math.log(i / len(train))

# transition
	transition_item = {}
	transition_target = {}

	for sample in train:
		for i in range(1, len(sample) - 2):
			if (sample[i + 1][1], sample[ i ][1]) not in transition_item:
				transition_item[(sample[i + 1][1], sample[i][1])] = 0
			transition_item[(sample[i + 1][1], sample[i][1])] += 1
			if (sample[i][1]) not in transition_target:
				transition_target[sample[i][1]] = 0
			transition_target[sample[i][1]] += 1

	transition = {}

	for t1 in tags:
		transition[t1] = {}
		for t2 in tags:
			if (t1, t2) in transition_item:
				transition[t1][t2] = math.log(transition_item[(t1, t2)]/len(train))
			else: 
				transition[t1][t2] = math.log((smooth_data) / (tag_count[t1] + smooth_data*len(tags)))
# emission
	emission_item = {}
	emssion_target = {}
	emission = {}

	for sample in train:
		for i in range(1, len(sample) - 1):
			if sample[i][1] not in emission_item:
				emission_item[sample[i][1]] = {}
			if sample[i][0] not in emission_item[sample[i][1]]:
				emission_item[sample[i][1]][sample[i][0]] = 0
			emission_item[sample[i][1]][sample[i][0]] += 1
	for i in words:
		emission[i] = {}
		for j in tags:
			if i in emission_item:
				if j in emission_item[i]:
					emission[i][j] = math.log(emission_item[i][j] / tag_count[j])
			else: 
				emission[i][j] = math.log((smooth_data) / (len(train) + smooth_data*len(tags))) 
	uk_emission = math.log((smooth_data) / (len(train) + smooth_data*len(tags))) 
	print(emission)
	predicts = []
	for sample in test:
		print(sample)
		more = transfer(initial, emission, transition, sample[1:-1], uk_emission, words)
		more.insert(0, (sample[0], "START"))
		more.append((sample[-1], "END"))
		predicts.append(more)

	return predicts
 



def transfer(initial, emission, transition, test, uk_emission, words):
	value_dict = {}

	key_list = []
	for key, i in initial.items():
		key_list.append(key)

	start_flag = True

	for i in range(len(test)):
		print(test[i])
		emission_base = {}
		if test[i] not in words:
			for key in key_list:
				emission_base[key] = uk_emission
		for key, value in emission.items():
			emission_base[key] = value[test[i]]
		if start_flag == True:
			value_dict[test[i]] = {}
			for fkey, fvalue in emission_base.items():
				value_dict[test[i]][fkey] = emission_base[fkey] + initial[fkey]
				start_flag = False
			continue
		value_dict[test[i]] = {}
		for key_next in key_list:
			transition_list = {}
			for key_previous, k in value_dict[test[i-1]].items():
				key_prev = key_previous.split("_")[-1]
				transition_list[str(key_previous) + "_" + str(key_next)] = transition[key_prev][key_next] + emission_base[key_next] + k
			b = max(transition_list, key = transition_list.get)
			v = transition_list[b]
			value_dict[test[i]][str(b)] = v
	b = max(value_dict[test[-1]], key = value_dict[test[-1]].get)
	print(b)

	prediction = []

	results = b.split("_")
	for i in range(len(results)):
		prediction.append((test[i], results[i]))


	return result_list


