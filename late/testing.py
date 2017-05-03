from izhikevich_original import izhikevich as neuron
from  python_speech_features import base as speech_features
import util
import learning

import math
import matplotlib.pylab as plt
import numpy
import os

#util.preapre_csv_file("./training_data/cut_samples/", "./output_file.csv")

layer_size = 520
input_layer = []
male_output_neuron = neuron()
female_output_neuron = neuron()

female_weights = util.load_csv("./csv/female_weights.csv")
male_weights = util.load_csv("./csv/male_weights.csv")

print len(female_weights)
print len(male_weights)

K_SYN = 1
tau = 0.1
time = 100 #ms
steps = int(time / tau)

filename = "_arctic_a0001_female_1.wav"
folder = "./training_data/audio_samples_8000/"

# create array of neuron objects
for i in range(0, layer_size):
	input_layer.append(neuron())

male_conductance = []
female_conductance = []

def train(filename):
	global female_hebbian, female_anti_hebbian, male_hebbian, male_anti_hebbian
	global layer_size, input_layer, male_output_neuron, female_output_neuron
	global female_weights, male_weights
	global K_SYN, tau, time, steps, male_conductance, female_conductance
	
	print "processing : ", str(folder + filename)

	# calculate MFCC features and injected_current to each neuron
	i_inj = util.get_injected_current(folder, filename, 70)

	# write i_inj to file, so this info can be used to calculate signature spike
	util.print_list_to_file(i_inj, "./csv/i_inj_new.csv", "a", str("i_injected " + filename))

	# determine teacher, this is used to determine whether output neuron
	# udergoes hebbian learning or anti-hebbian learning
	teacher = util.get_sample_label(filename)

	# run simulation with injected current
	for i in range(0,steps):
		for j in range(0,layer_size):
			input_layer[j].input_snn_one_step(i_inj[j])

	# extract spikes for each of the neuron in the input layer
	pre_synaptic_spikes = util.extract_spike_trains(input_layer)

	# generate spike time for each of the neuron in the input layer
	pre_synaptic_spike_times = util.extract_spike_times(pre_synaptic_spikes, time, tau = 0.1)

	# run simulation for given "time(steps)" 
	for t in range(0, steps):
		# cuurent time
		current_time = float(t) * tau

		print current_time

		# :::::::::::::::::::::::::    FEMALE OUTPUT NEURON     :::::::::::::::::::::::::
		# :::::::::::::::::::::::::    FEMALE OUTPUT NEURON     :::::::::::::::::::::::::

		# calculate conductance at time t for female output neuron
		weighted_conductance_at_t = util.linear_additive_conductance_at_t(pre_synaptic_spike_times, female_weights, t)
		female_conductance.append(weighted_conductance_at_t)

		# calculate synaptic output voltage of female output neuron at time t
		female_output_neuron, female_output_voltage = util.additive_synaptic_current_at_t(weighted_conductance_at_t, female_output_neuron)

		# :::::::::::::::::::::::::    MALE OUTPUT NEURON     :::::::::::::::::::::::::
		# :::::::::::::::::::::::::    MALE OUTPUT NEURON     :::::::::::::::::::::::::

		# calculate conductance at time t for female output neuron
		weighted_conductance_at_t = util.linear_additive_conductance_at_t(pre_synaptic_spike_times, male_weights, t)
		male_conductance.append(weighted_conductance_at_t)

		# calculate synaptic output voltage of female output neuron at time t
		male_output_neuron, male_output_voltage = util.additive_synaptic_current_at_t(weighted_conductance_at_t, male_output_neuron)


train(filename)

spikes = female_output_neuron.generate_spike_train(35)
plt.figure()
plt.subplot(211)
plt.title("Spike Coding Female")
plt.plot(female_output_neuron.get_output())

plt.subplot(212)
plt.plot(spikes)
plt.title("Spike Train female" + str(spikes.count(1)))

spikes = male_output_neuron.generate_spike_train(35)
plt.figure()
plt.subplot(211)
plt.title("Spike Coding male")
plt.plot(male_output_neuron.get_output())

plt.subplot(212)
plt.plot(spikes)
plt.title("Spike Train male" + str(spikes.count(1)))


plt.figure()
plt.subplot(211)
plt.title("female_conductance")
plt.plot(female_conductance)

plt.subplot(212)
plt.plot(male_conductance)
plt.title("male_conductance")


plt.show()