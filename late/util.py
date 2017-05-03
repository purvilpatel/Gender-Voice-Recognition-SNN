from izhikevich_original import izhikevich as neuron
from  python_speech_features import base as speech_features

import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav
import math
import os


SPIKE_THRESHOLD = 35
K_SYN = 1
TAU = 2

################################################################################
### Extract spike timings from spike train of 0's and 1's of single neuron
################################################################################
def extract_single_spike_time(spike_train, time, tau = 0.1):
	steps = int(time / tau)
	spike_time = []
	for j in xrange(0,steps):
		if(spikes[j] > 0):
			spike_time.append(float(j) * tau) # convert index to timestamp
	return spike_time

################################################################################
### Extract spike timings from spike train of 0's and 1's
### Indices of 1's are convereted to spike time and returned as array of spikes
### Rows represents neurons and columns represents time stamps
################################################################################
def extract_spike_times(spikes, time, tau = 0.1):
	steps = int(time / tau)
	spike_time = []
	for i in xrange(0,len(spikes)):
		temp_spike_time = []  # temp 
		for j in xrange(0,steps):
				if(spikes[i][j] > 0):
					temp_spike_time.append(float(j) * tau) # convert index to timestamp
		spike_time.append(temp_spike_time)
	return spike_time

########################################################################
### Extract spike trains of 0's and 1's from array of Izhikevich neurons
########################################################################
def extract_spike_trains(neurons, threshold = SPIKE_THRESHOLD):
	spikes = []
	for i in xrange(0,len(neurons)):
		temp_spike_train = neurons[i].generate_spike_train(threshold) # generate individual spike train
		spikes.append(temp_spike_train)
	return spikes

############################################################################
### Get synaptic current for the receiver neuron
###		spike_times : input spikes times to the receiver neuron
###		weights : calculated weight of each synapse
###		time(in_ms) : simulation duration
###		receiver_neuron : if this is None than new neuron object is returned
###		V : default value of voltage -60
###		K_SYN : as per paper 1
###		TAU : as per paper 2
###		delta_t : time increment
############################################################################
def get_synaptic_current(spike_times, weights, time, receiver_neuron = None, V = -60, K_SYN = 1, TAU = 2, delta_t = 0.1):
	steps = int(time / delta_t)
	# get conductance values for number of of steps
	G = conductance(spike_times, weights, steps, K_SYN, TAU, delta_t)

	# create new neuron object if needed
	if(receiver_neuron == None):
		receiver_neuron = neuron()
	
	# run simulation for one syanptic current value. I_SYN[i] = -V * G[i] 
	for i in xrange(0,steps):
		receiver_neuron.input_snn_one_step(-receiver_neuron.get_prev_voltage() * G[i])

	return receiver_neuron, G

############################################################################
### Get synaptic conductance for the receiver neuron
###		spike_times : input spikes times to the receiver neuron
###		weights : calculated weight of each synapse
###		steps : duration of simulation in terms of steps
###		K_SYN : as per paper 1
###		TAU : as per paper 2
###		delta_t : time increment
############################################################################

def conductance(spike_times, weights, steps, K_SYN = 1, TAU = 2, delta_t = 0.1):
	G = []
	for t in xrange(0, steps):
		time_increment = float(t) * delta_t
		sum = 0
		for k in xrange(0,len(spike_times)):
			for j in xrange(0,len(spike_times[k])):
				sum = sum + (weights[k] * (K_SYN  * (time_increment - spike_times[k][j]) * math.exp(-(time_increment - spike_times[k][j]) / (TAU) / 1000.0)))
		G.append(sum)
	return G

#############################################################
### plot spike coding and spike train for given neuron layer
#############################################################
def plot_neurons(neuron_layer, i_inj, weights):
	row = len(neuron_layer)
	col = 2
	plt.figure()
	if row > 4:
		row = 4
	for i in xrange(0, row):
		plot = i*2 + 1
		plt.subplot(str(row) + str(col) + str(plot))
		plt.plot(neuron_layer[i].get_output())
		plt.title("Spike Coding " + str(i) + ", i_inj : " + str(i_inj[i]))

		plot = i*2 + 2
		plt.subplot(str(row) + str(col) + str(plot))
		plt.plot(neuron_layer[i].generate_spike_train(SPIKE_THRESHOLD))
		plt.title("Spike Train " + str(i))

	'''
	figure = plt.gcf() # get current figure
	figure.set_size_inches(10, 10)
	plt.savefig("./results/" + str(i_inj) + ", weights " + str(weights) + ".jpg",dpi = 100)
	'''

#############################################################
### conductance kernel
#############################################################
def single_conductance(steps, spike_time):
	G = []

	for i in xrange(0,steps):
		t = float(i)/10.0
		temp = K_SYN * t * math.exp(-t/TAU)
		G.append(K_SYN * t * math.exp(-t/TAU))

	if spike_time == 0:
		return G
	return shift(int(spike_time/0.1), G)


#############################################################
### additive conductance kernel
#############################################################
def linear_additive_conductance(steps, spike_times, weights):
	all_conductance = []
	conductance = []

	for i in xrange(0,len(spike_times)):
		conductance.append(conductance_for_spike_train(steps, spike_times[i]))

	all_conductance = add_array(conductance)
	all_weighted_conductance = add_weighted_array(conductance, weights)

	return  conductance, all_conductance, all_weighted_conductance


#############################################################
### additive conductance kernel at particular time
#############################################################
def get_current_linear_additive_conductance(steps, spike_times, weights, time):
	conductance = []

	for i in xrange(0,len(spike_times)):
		conductance.append(conductance_for_spike_train(steps, spike_times[i]))

	all_weighted_conductance = add_weighted_array(conductance, weights)

	return  all_weighted_conductance[time]


#############################################################
### individual spike train conductance
#############################################################
def conductance_for_spike_train(steps, spike_times):
	C = []
	for i in range(len(spike_times)):
		C.append(single_conductance(steps, spike_times[i]))

	return add_array(C)

#######################################################################
### generate synaptic current for receiver neuron based on conductance
#######################################################################
def additive_synaptic_current(steps, conductance, receiver_neuron = None):
	# create new neuron object if needed
	if(receiver_neuron == None):
		receiver_neuron = neuron()
	
	# run simulation for one syanptic current value. I_SYN[i] = -V * G[i] 
	for i in xrange(0,steps):
		receiver_neuron.input_snn_one_step(60.0 * conductance[i])

	return receiver_neuron


#######################################################################
### calculate synaptic current for receiver neuron based on conductance at time t
### conductance_at_t : conductance at time t
### receiver_neuron : reciever neuron
#######################################################################
def additive_synaptic_current_at_t(conductance_at_t, receiver_neuron):
	# run simulation for one syanptic current value. I_SYN[i] = -V * G[i] 
	receiver_neuron.input_snn_one_step(60.0 * conductance_at_t)

	return receiver_neuron ,receiver_neuron.get_voltage()


#############################################################
### Shift an array right add #key elements as left padding
#############################################################
def shift(key, array):
	return ([0.0] * int(key)) + array[:-int(key)]

def find_min(array):
	x = []
	for i in range(len(array)):
		x.append(min(array[i]))
	return min(x)

def add_array(array):
	sum = []
	for i in range(len(array[0])):
		temp = 0
		for j in range(len(array)):
			temp = temp + array[j][i]
		sum.append(temp)
	return sum

def add_weighted_array(array, weights):
	sum = []

	for i in range(len(array[0])):
		temp = 0.0
		for j in range(len(array)):
			temp = temp + float(weights[j] * array[j][i])
		sum.append(temp)
	return sum

def get_marker(steps, marker):
	mark = ([0.0] * int(steps))
	mark[marker]= 0.7
	return mark


#############################################################
### plot spike coding and spike train for given neuron layer
#############################################################
def plot_conductance(neuron_layer, conductance, i_inj):
	row = len(neuron_layer)
	col = 1
	color = ["r", "g", "b", "c", "r", "g", "b", "c", "r", "g", "b", "c", "r", "g", "b", "c"]
	plt.figure()
	if row > 9:
		row = 9
	for i in xrange(0, row):
		plt.subplot(str(row) + str(col) + str(i+1))
		plt.plot(neuron_layer[i].generate_spike_train(SPIKE_THRESHOLD),color[i+1])
		plt.plot(conductance[i], color[i])
		plt.title("Conductance of neuron " + str(i+1) + " - " + str(i_inj[i]))


SPEAKER_GENDER_FEMALE = 0
SPEAKER_GENDER_MALE = 1
SPEAKER_GENDER_ERROR = -1
#############################################################
### detect speaker's gender from file name
#############################################################
def get_sample_label(filename):
	if filename.find("female") != -1:
		return SPEAKER_GENDER_FEMALE
	if filename.find("male") != -1:
		return SPEAKER_GENDER_MALE
	return SPEAKER_GENDER_ERROR
	return NONE


#####################################################################
### write injected current to output_file
### this is done for all the files in given folder
####################################################################
def preapre_csv_file(data_folder, output_file):
	thefile = open(output_file, 'w')
	for files in os.listdir(data_folder):

		(rate,sig) = wav.read(data_folder + files)
		mfcc_feat = speech_features.mfcc(sig,rate, winlen=0.040, winstep= 0.042)
		features = []
	
		for j in range(0,13):
			temp=[]
			for i in range(0, len(mfcc_feat)):
				temp.append(mfcc_feat[i][j])
			features.append(temp)

		injected_current = []

		for j in range(0,40):
			for i in range(0,13):
				injected_current.append(features[i][j])
	
		for item in injected_current:
			thefile.write("%s," % item)
		thefile.write("\n")
		print len(injected_current)

####################################################################
### get injected current for given file using mfcc,
### add base current to mfcc featues to calculate injected current
####################################################################
base_current = 70
def get_injected_current(data_folder, filename, base_current):
	(rate,sig) = wav.read(data_folder + filename)
	mfcc_feat = speech_features.mfcc(sig,rate, winlen=0.040, winstep= 0.015)
	features = []
	
	for j in range(0,13):
		temp=[]
		for i in range(0, len(mfcc_feat)):
			temp.append(mfcc_feat[i][j])
		features.append(temp)
	
	print len(features), len(features[0])
	
	injected_current = []

	for j in range(0,40):
		for i in range(0,13):
			injected_current.append(base_current + features[i][j])

	return injected_current

##################################################
### save list to files
###
##################################################
def print_list_to_file(thelist, filename, mode, listname = ""):
	thefile = open(filename, mode)

	thefile.write(str(listname + ","))
	for item in thelist:
		thefile.write("%s," % item)
	thefile.write("\n")

	thefile.close()


###########################################
### convert csv file to python list
### filename : name of csv file
###########################################
import csv
def load_csv(filename):
	thelist = []
	f = open(filename, "r")
	reader = csv.reader(f)
	for row in reader:
		for col in row:
			thelist.append(float(col))
	return thelist



def single_conductance_at_t(spike_time, t):
	t = (t - spike_time)
	if t <= 0:
		return 0
	return K_SYN * t * math.exp(-t/TAU)
	

def conductance_for_spike_train_at_t(individual_spike_train, t):
	temp = 0.0
	for i in range(len(individual_spike_train)):
		temp = temp + single_conductance_at_t(individual_spike_train[i], t)

	return temp

def linear_additive_conductance_at_t(spike_times, weights, t):
	weighted_conductance = 0.0
	for i in xrange(0,len(spike_times)):
		weighted_conductance = weighted_conductance + float(conductance_for_spike_train_at_t(spike_times[i], t) * weights[i])

	return  weighted_conductance