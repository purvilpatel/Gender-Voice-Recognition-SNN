from izhikevich_original import izhikevich as neuron
import util

import matplotlib.pyplot as plt
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


"""
############################################################################
### for given spike time array find what is the next pre synaptic spike time
### spike_time : array of pre synaptic spike times
### time_stamp : post synaptic spike time
###    if post synaptic spike accours before any of the pre synaptic spike
###    return SPECIAL_CASE = -999 to indicate this special case
############################################################################
def get_next_spike(spike_time_array, time_stamp):	
	if time_stamp >= spike_time_array[len(spike_time_array)-1]:
		return spike_time_array[-1]

	if time_stamp < spike_time_array[0]:
		return SPECIAL_CASE

	for i in xrange(1,len(spike_time_array)):
		if spike_time_array[i] > time_stamp:
			return spike_time_array[i-1]

"""

##########################################################################
### Implementaion of Hebbian learning Rule
### pre_synaptic_spike_time_array : array of pre synaptic spike times
### post_synaptic_spike_time : post synaptic spike time
### weight : synaptic weight
##########################################################################
def synapse_learning(pre_synaptic_spike_time_array, post_synaptic_spike_time, weight):
	global special_count, increase_count, decrease_count
	# get pre synaptic time for given time. see "get_prev_spike" for more info
	pre_synaptic_spike_time = get_prev_spike(pre_synaptic_spike_time_array, post_synaptic_spike_time)

	# if special case than decrease weight as post spike happened before pre spike
	if(pre_synaptic_spike_time == SPECIAL_CASE):
		special_count = special_count + 1
		return decrease_weight(pre_synaptic_spike_time, post_synaptic_spike_time, weight)

	# if post spike happened within TIME_WINDOW of pre spike, increase weight
	if(post_synaptic_spike_time - pre_synaptic_spike_time >= TIME_WINDOW ):
		increase_count = increase_count + 1
		return increase_weight(pre_synaptic_spike_time, post_synaptic_spike_time, weight)
	# if post spike happened outside TIME_WINDOW of pre spike, decrease weight
	else:
		decrease_count = decrease_count + 1
		return decrease_weight(pre_synaptic_spike_time, post_synaptic_spike_time, weight)


##########################################################################
### Implementaion of anti-Hebbian learning Rule
### pre_synaptic_spike_time_array : array of pre synaptic spike times
### post_synaptic_spike_time : post synaptic spike time
### weight : synaptic weight
##########################################################################
def inverese_synapse_learning(pre_synaptic_spike_time_array, post_synaptic_spike_time, weight):
	global special_count, increase_count, decrease_count
	# get pre synaptic time for given time. see "get_prev_spike" for more info
	pre_synaptic_spike_time = get_prev_spike(pre_synaptic_spike_time_array, post_synaptic_spike_time)

	# if special case than increase weight as post spike happened before pre spike
	if(pre_synaptic_spike_time == SPECIAL_CASE):
		special_count = special_count + 1
		return increase_weight(pre_synaptic_spike_time, post_synaptic_spike_time, weight)

	# if post spike happened within TIME_WINDOW of pre spike, decrease weight
	if(post_synaptic_spike_time - pre_synaptic_spike_time >= TIME_WINDOW ):
		increase_count = increase_count + 1
		return decrease_weight(pre_synaptic_spike_time, post_synaptic_spike_time, weight)
	# if post spike happened outside TIME_WINDOW of pre spike, increase weight
	else:
		decrease_count = decrease_count + 1
		return increase_weight(pre_synaptic_spike_time, post_synaptic_spike_time, weight)


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
	delta_w = 0.01 * B * math.exp( - (abs(post_synaptic_spike_time - pre_synaptic_spike_time) / TAU_PLUS))
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


layer_size = 1
input_layer = []
output = neuron()

weights = [0.1, 0.05, 0.2, 0.3, 0.1, 0.5, 0.1, 0.3, 0.5, 0.1]
print weights
normalise_weight(weights)
print weights


weights = initialize_weight(10)
print weights
normalise_weight(weights)
print weights

'''
weights = [0.1]#, 0.05, 0.2, 0.3, 0.1, 0.5, 0.1, 0.3, 0.5, 0.1]
i_inj =   [25]#, 15, 18, 15, 25, 15, 17, 10, 50, 15]

K_SYN = 1
tau = 0.1
time = 10 #ms
steps = int(time / tau)

# create array of neuron objects
for i in xrange(0,layer_size):
	input_layer.append(neuron())

# run simulation with injected current
for i in xrange(0,steps):
	for j in xrange(0,layer_size):
		input_layer[j].input_snn_one_step(i_inj[j])

# extract spikes for each of the neuron in the input layer
spikes = util.extract_spike_trains(input_layer)

# generate spike time for each of the neuron in the input layer
spike_times = util.extract_spike_times(spikes, time, tau = 0.1)

pre_synaptic_spike_time_array = spike_times[0]

# plot input layer
util.plot_neurons(input_layer, i_inj, weights)

for l in xrange(1,10000):
	print l
	conductance, all_conductance, all_weighted_conductance = util.linear_additive_conductance(steps, spike_times,weights)

	additive_neuron = neuron()

	additive_neuron = util.additive_synaptic_current(steps, all_weighted_conductance, additive_neuron)
	additive_neuron_spikes = additive_neuron.generate_spike_train(util.SPIKE_THRESHOLD)
	"""
	plt.figure()
	plt.subplot(211)
	plt.title("Spike Coding C weight : " + str(weights) + " ---- " + str(l))
	plt.plot(additive_neuron.get_output())

	plt.subplot(212)
	plt.plot(additive_neuron_spikes)
	plt.title("Spike Train C, # spikes : " + str(additive_neuron_spikes.count(1)))
	"""
	temp_spike_time = []
	for i in xrange(0,steps):
		if(additive_neuron_spikes[i] > 0):
			temp_spike_time.append(float(i) * tau) # convert index to timestamp
	
	if additive_neuron_spikes.count(1) == 4:
		break
	for st in xrange(0, len(temp_spike_time)):
		post_synaptic_spike_time = temp_spike_time[st]
		weights[0] = synapse_learning(pre_synaptic_spike_time_array, post_synaptic_spike_time, weights[0])

plt.figure()
plt.subplot(211)
plt.title("Spike Coding C weight : " + str(weights) + " ---- " + str(l))
plt.plot(additive_neuron.get_output())

plt.subplot(212)
plt.plot(additive_neuron_spikes)
plt.title("Spike Train C, # spikes : " + str(additive_neuron_spikes.count(1)))
	
print increase_count, decrease_count, special_count, weights[0]

plt.show()'''
