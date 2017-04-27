from izhikevich_original import izhikevich as neuron
import util
import matplotlib.pyplot as plt
import math

layer_size = 1
input_layer = []
output = neuron()

weights = [0.1]#, 0.5]#, 0.1, 0.3, 0.1, 0.5, 0.1, 0.3, 0.5, 0.1]
i_inj =   [25]#, 15]#, 18, 15, 25, 15, 17, 10, 50, 15]

K_SYN = 1
TAU = 2
tau = 0.1
time = 100 #ms
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


# plot input layer
util.plot_neurons(input_layer, i_inj, weights)

conductance, all_conductance, all_weighted_conductance = util.linear_additive_conductance(steps, spike_times,weights)

# plot input spike times and conductance
util.plot_conductance(input_layer, conductance, i_inj)

plt.figure()
plt.subplot(211)
plt.title("conductance")
plt.plot(all_conductance)

plt.subplot(212)
plt.title("weighted conductance")
plt.plot(all_weighted_conductance)

additive_neuron = neuron()

additive_neuron = util.additive_synaptic_current(steps, all_weighted_conductance, additive_neuron)
additive_neuron_spikes = additive_neuron.generate_spike_train(util.SPIKE_THRESHOLD)

plt.figure()
plt.subplot(211)
plt.title("Spike Coding C weight : " + str(weights))
plt.plot(additive_neuron.get_output())

plt.subplot(212)
plt.plot(additive_neuron_spikes)
plt.title("Spike Train C, # spikes : " + str(additive_neuron_spikes.count(1)))
plt.show()