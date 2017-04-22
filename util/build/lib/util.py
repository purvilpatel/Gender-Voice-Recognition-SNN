from izhikevich import izhikevich as neuron
import matplotlib.pyplot as plt
import math

SPIKE_THRESHOLD = 35

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
		receiver_neuron.input_snn_one_step(V * G[i])

	return receiver_neuron

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
				sum = sum + (weights[k] * (K_SYN  * (time_increment - spike_times[k][j]) * math.exp(-(time_increment - spike_times[k][j]) / TAU / 1000.0)))
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