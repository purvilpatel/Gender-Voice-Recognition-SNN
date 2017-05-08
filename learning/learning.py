from izhikevich_original import izhikevich as neuron
import util

import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import sys
import math
import random

A = 0.001
B = -0.0005
TAU_PLUS = 5.0 # in ms
TAU_MINUS = 5.0 # in ms
SPECIAL_CASE = -999
TIME_WINDOW = 0.1

increase_count = 0
decrease_count = 0
special_count = 0

#################################################################################
### for given spike time array find what is the previous pre synaptic spike time
### spike_time : array of pre synaptic spike times
### time_stamp : post synaptic spike time
###    if post synaptic spike accours before any of the pre synaptic spike
###    return SPECIAL_CASE = -999 to indicate this special case
#################################################################################
def get_prev_spike(spike_time_array, time_stamp):
	# post synaptic spike before pre synaptic spike, SPECIAL_CASE
	if time_stamp < spike_time_array[0]:
		return SPECIAL_CASE

	# post synaptic spike, after last pre synaptic spike
	if time_stamp >= spike_time_array[len(spike_time_array)-1]:
		return spike_time_array[-1]

	for i in xrange(1,len(spike_time_array)):
		if spike_time_array[i] > time_stamp:
			return spike_time_array[i-1]


############################################################################
### for given spike time array find what is the next pre synaptic spike time
### spike_time : array of pre synaptic spike times
### time_stamp : post synaptic spike time
###    if post synaptic spike accours before any of the pre synaptic spike
###    return SPECIAL_CASE = -999 to indicate this special case
############################################################################
def get_next_spike(spike_time_array, time_stamp):	
	# if post synaptic spike after last pre synaptic spike return 100ms
	if time_stamp > spike_time_array[-1]:
		return 100

	# if post synaptic spike before first pre synaptic spike return first syanptic spike
	if time_stamp < spike_time_array[0]:
		return spike_time_array[0]

	for i in xrange(len(spike_time_array) - 1, -1, -1):
		if spike_time_array[i] <= time_stamp:
			if i == len(spike_time_array) - 1:
				return spike_time_array[i]
			return spike_time_array[i + 1]


##########################################################################
### Implementaion of Hebbian learning Rule
### pre_synaptic_spike_time_array : array of pre synaptic spike times
### post_synaptic_spike_time : post synaptic spike time
### weight : synaptic weight
##########################################################################
def synapse_learning(pre_synaptic_spike_time_array, post_synaptic_spike_time, weight):
	global special_count, increase_count, decrease_count

	# get previous pre synaptic time for given time. see "get_prev_spike" for more info
	prev_pre_synaptic_spike_time = get_prev_spike(pre_synaptic_spike_time_array, post_synaptic_spike_time)
	# get next pre synaptic time for given time. see "get_prev_spike" for more info
	next_pre_synaptic_spike_time = get_next_spike(pre_synaptic_spike_time_array, post_synaptic_spike_time)

	# if special case than decrease weight as post spike happened before pre spike
	if(prev_pre_synaptic_spike_time == SPECIAL_CASE):
		special_count = special_count + 1
		return decrease_weight(next_pre_synaptic_spike_time, post_synaptic_spike_time, weight)

	# if post spike fired within TIME_WINDOW of pre spike, increase weight
	if(post_synaptic_spike_time - prev_pre_synaptic_spike_time <= TIME_WINDOW ):
		increase_count = increase_count + 1
		return increase_weight(prev_pre_synaptic_spike_time, post_synaptic_spike_time, weight)
	# if post spike happened outside TIME_WINDOW of pre spike, decrease weight
	else:
		decrease_count = decrease_count + 1
		return decrease_weight(next_pre_synaptic_spike_time, post_synaptic_spike_time, weight)


##########################################################################
### Implementaion of anti-Hebbian learning Rule
### pre_synaptic_spike_time_array : array of pre synaptic spike times
### post_synaptic_spike_time : post synaptic spike time
### weight : synaptic weight
##########################################################################
def inverese_synapse_learning(pre_synaptic_spike_time_array, post_synaptic_spike_time, weight):
	global special_count, increase_count, decrease_count
	
	# get previous pre synaptic time for given time. see "get_prev_spike" for more info
	prev_pre_synaptic_spike_time = get_prev_spike(pre_synaptic_spike_time_array, post_synaptic_spike_time)
	# get next pre synaptic time for given time. see "get_prev_spike" for more info
	next_pre_synaptic_spike_time = get_next_spike(pre_synaptic_spike_time_array, post_synaptic_spike_time)

	# if special case than increase weight as post spike happened before pre spike
	if(prev_pre_synaptic_spike_time == SPECIAL_CASE):
		special_count = special_count + 1
		return increase_weight(prev_pre_synaptic_spike_time, post_synaptic_spike_time, weight)

	# if post spike happened within TIME_WINDOW of pre spike, decrease weight
	if(post_synaptic_spike_time - prev_pre_synaptic_spike_time <= TIME_WINDOW ):
		increase_count = increase_count + 1
		return decrease_weight(next_pre_synaptic_spike_time, post_synaptic_spike_time, weight)
	# if post spike happened outside TIME_WINDOW of pre spike, increase weight
	else:
		decrease_count = decrease_count + 1
		return increase_weight(prev_pre_synaptic_spike_time, post_synaptic_spike_time, weight)


##########################################################################
### Implementaion of LTP
### pre_synaptic_spike_time_array : array of pre synaptic spike times
### post_synaptic_spike_time : post synaptic spike time
### weight : synaptic weight
##########################################################################
def increase_weight(pre_synaptic_spike_time, post_synaptic_spike_time, weight):
	delta_w = 0.01 * A * math.exp( - (abs(post_synaptic_spike_time - pre_synaptic_spike_time) / TAU_PLUS))
	return (weight + (delta_w * weight))


##########################################################################
### Implementaion of LTD
### pre_synaptic_spike_time_array : array of pre synaptic spike times
### post_synaptic_spike_time : post synaptic spike time
### weight : synaptic weight
##########################################################################
def decrease_weight(pre_synaptic_spike_time, post_synaptic_spike_time, weight):
	delta_w = 0.01 * B * math.exp( - (abs(pre_synaptic_spike_time - post_synaptic_spike_time) / TAU_PLUS))
	return (weight + (delta_w * weight))


##########################################################################
### Implementaion of weight normalisation
### weights : array of synaptic weight
##########################################################################
def normalise_weight(weights):
	# find max and min weights from array
	max_weight = max(weights)
	min_weight = min(weights)

	# scale all the weights in array according to max and min weights
	for i in xrange(0,len(weights)):
		weights[i] = (weights[i] - min_weight) / (max_weight - min_weight)


##########################################################################
### Implementaion of weight initialization
### no_synapse : number of synapses
##########################################################################
def initialize_weight(no_synapse):
	weights = []

	# scale all the weights in array according to max and min weights
	for i in xrange(0,no_synapse):
		weights.append(random.random())

	return weights
