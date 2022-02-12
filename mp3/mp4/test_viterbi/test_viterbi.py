# mp4.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created Fall 2018: Margaret Fleck, Renxuan Wang, Tiantian Fang, Edward Huang (adapted from a U. Penn assignment)
# Modified Spring 2020: Jialu Li, Guannan Guo, and Kiran Ramnath
# Modified Fall 2020: Amnon Attali, Jatin Arora
# Modified Spring 2021 by Kiran Ramnath (kiranr2@illinois.edu)

"""
This file should not be submitted - it is only meant to test your implementation of the Viterbi algorithm. 

See Piazza post @650 - This example is intended to show you that even though P("back" | RB) > P("back" | VB), 
the Viterbi algorithm correctly assigns the tag as VB in this context based on the entire sequence. 
"""
from utils import read_files, get_nested_dictionaries
import math
import numpy as np

def main():
	test, emission, transition, output = read_files()
	emission, transition = get_nested_dictionaries(emission, transition)
	initial = transition["START"]

	print("initial: ", initial)
	print("emission: ", emission)
	print("transition: ", transition)

	value_dict = {}

	key_list = []
	for key, i in initial.items():
		key_list.append(key)

	start_flag = True

	for i in range(len(test[0])):
		emission_base = {}
		for key, value in emission.items():
			emission_base[key] = math.log(value[test[0][i]])
		if start_flag == True:
			value_dict[test[0][i]] = {}
			for fkey, fvalue in emission_base.items():
				value_dict[test[0][i]][fkey] = emission_base[fkey] + math.log(initial[fkey])
				start_flag = False
			continue
		value_dict[test[0][i]] = {}
		for key_next in key_list:
			transition_list = {}
			for key_previous, k in value_dict[test[0][i-1]].items():
				key_prev = key_previous.split("_")[-1]
				transition_list[str(key_previous) + "_" + str(key_next)] = math.log(transition[key_prev][key_next]) + emission_base[key_next] + k
			b = max(transition_list, key = transition_list.get)
			v = transition_list[b]
			value_dict[test[0][i]][str(b)] = v
	b = max(value_dict[test[0][-1]], key = value_dict[test[0][-1]].get)
	print(b)

	prediction = []

	results = b.split("_")
	for i in range(len(results)):
		prediction.append((test[0][i], results[i]))


	print('Your Output is:',prediction,'\n Expected Output is:',output)


if __name__=="__main__":
	main()