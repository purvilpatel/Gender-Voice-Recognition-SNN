from izhikevich_original import izhikevich as neuron
from  python_speech_features import base as speech_features
import util
import learning

import math
import matplotlib.pylab as plt
import numpy
import os

#util.preapre_csv_file("./training_data/cut_samples/", "./output_file_new.csv")

# remove the injected current file
if os.path.exists("./csv/i_inj_new.csv"):
	os.remove("./csv/i_inj_new.csv")

female_hebbian = 0
female_anti_hebbian = 0
male_hebbian = 0
male_anti_hebbian = 0

layer_size = 520
input_layer = []
male_output_neuron = neuron()
female_output_neuron = neuron()

female_weights = learning.initialize_weight(layer_size)
male_weights = learning.initialize_weight(layer_size)

util.print_list_to_file(female_weights, "./csv/female_weights_new.csv", "w", "female_weights")
util.print_list_to_file(male_weights, "./csv/male_weights_new.csv", "w", "male_weights")

K_SYN = 1
tau = 0.1
time = 100 #ms
steps = int(time / tau)

filename = "_arctic_a0001_female_1.wav"
folder = "./training_data/audio_samples_8000/"

male_conductance = []
female_conductance = []

# create array of neuron objects
for i in range(0, layer_size):
	input_layer.append(neuron())

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

	input_layer = []
	for i in range(0, layer_size):
		input_layer.append(neuron())

	female_output_neuron.reset_voltage()
	male_output_neuron.reset_voltage()


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

		# print current_time

		# :::::::::::::::::::::::::    FEMALE OUTPUT NEURON     :::::::::::::::::::::::::
		# :::::::::::::::::::::::::    FEMALE OUTPUT NEURON     :::::::::::::::::::::::::

		# calculate conductance at time t for female output neuron
		weighted_conductance_at_t = util.linear_additive_conductance_at_t(pre_synaptic_spike_times, female_weights, t)

		# calculate synaptic output voltage of female output neuron at time t
		female_output_neuron, female_output_voltage = util.additive_synaptic_current_at_t(weighted_conductance_at_t, female_output_neuron)

		# if voltage is >= SPIKE_THRESHOLD we have a spike
		if female_output_voltage >= util.SPIKE_THRESHOLD:
			for i in range(0, layer_size):
				# apply Hebbian learning, if teacher is female
				if teacher == util.SPEAKER_GENDER_FEMALE:
					female_weights[i] = learning.synapse_learning(pre_synaptic_spike_times[i], current_time, female_weights[i])
				# apply anti-Hebbian learning, if teacher is male
				elif teacher == util.SPEAKER_GENDER_MALE:
					female_weights[i] = learning.inverese_synapse_learning(pre_synaptic_spike_times[i], current_time, female_weights[i])

			if teacher == util.SPEAKER_GENDER_FEMALE:
				female_hebbian = female_hebbian + 1
			elif teacher == util.SPEAKER_GENDER_MALE:
				female_anti_hebbian = female_anti_hebbian + 1


		# :::::::::::::::::::::::::    MALE OUTPUT NEURON     :::::::::::::::::::::::::
		# :::::::::::::::::::::::::    MALE OUTPUT NEURON     :::::::::::::::::::::::::

		# calculate conductance at time t for female output neuron
		weighted_conductance_at_t = util.linear_additive_conductance_at_t(pre_synaptic_spike_times, male_weights, t)

		# calculate synaptic output voltage of female output neuron at time t
		male_output_neuron, male_output_voltage = util.additive_synaptic_current_at_t(weighted_conductance_at_t, male_output_neuron)

		# if voltage is >= SPIKE_THRESHOLD we have a spike
		if male_output_voltage >= util.SPIKE_THRESHOLD:
			for i in range(0, layer_size):
				# apply Hebbian learning, if teacher is female
				if teacher == util.SPEAKER_GENDER_MALE:
					male_weights[i] = learning.synapse_learning(pre_synaptic_spike_times[i], current_time, male_weights[i])
				# apply anti-Hebbian learning, if teacher is male
				elif teacher == util.SPEAKER_GENDER_FEMALE:
					male_weights[i] = learning.inverese_synapse_learning(pre_synaptic_spike_times[i], current_time, male_weights[i])

			if teacher == util.SPEAKER_GENDER_MALE:
				male_hebbian = male_hebbian + 1
			elif teacher == util.SPEAKER_GENDER_FEMALE:
				male_anti_hebbian = male_anti_hebbian + 1

	learning.normalise_weight(female_weights)
	learning.normalise_weight(male_weights)
	util.print_list_to_file(female_weights, "./csv/female_weights_new.csv", "a", str("updated_female_weights"+filename))
	util.print_list_to_file(male_weights, "./csv/male_weights_new.csv", "a", str("updated_male_weights"+filename))


# :::::::::::::::::::::::::    file loop here     :::::::::::::::::::::::::
# :::::::::::::::::::::::::    file loop here     :::::::::::::::::::::::::
# :::::::::::::::::::::::::    file loop here     :::::::::::::::::::::::::
for filename in os.listdir(folder):
	train(filename)

print "female_hebbian", female_hebbian
print "female_anti_hebbian", female_anti_hebbian
print "male_hebbian", male_hebbian
print "male_anti_hebbian", male_anti_hebbian

print learning.increase_count, learning.decrease_count, learning.special_count 

spikes = female_output_neuron.generate_spike_train(35)
plt.figure()
plt.subplot(211)
plt.title("Spike Coding Female")
plt.plot(female_output_neuron.get_output())

plt.subplot(212)
plt.plot(spikes)
plt.title("Spike Train Female" + str(spikes.count(1)))

spikes = male_output_neuron.generate_spike_train(35)
plt.figure()
plt.subplot(211)
plt.title("Spike Coding Male")
plt.plot(male_output_neuron.get_output())

plt.subplot(212)
plt.plot(spikes)
plt.title("Spike Train Male" + str(spikes.count(1)))

plt.show()