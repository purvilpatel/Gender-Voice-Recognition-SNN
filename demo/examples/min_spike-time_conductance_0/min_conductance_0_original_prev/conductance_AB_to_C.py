from izhikevich_original import izhikevich as neuron
import util
import matplotlib.pyplot as plt
import math

layer_size = 1
input_layer = []
output = neuron()

weights = [1]#, 0.5, 0.1, 0.3]#, 0.1, 0.5, 0.1, 0.3, 0.2, 1]
i_inj = [25]#, 150, 185, 150]#, 250, 150, 175, 100, 500, 150]

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

# generate synapic current for output neuron
output, conductance = util.get_synaptic_current(spike_times, weights, time, output, V = -60)

plt.figure()
plt.title("Conductance")
plt.grid()
plt.plot(conductance)


# generate spikes train for output neuron
output_spikes = output.generate_spike_train(util.SPIKE_THRESHOLD)

# plot output layer
plt.figure()
plt.subplot(211)
plt.title("Spike Coding C weight : " + str(weights))
plt.plot(output.get_output())

plt.subplot(212)
plt.plot(output_spikes)
plt.title("Spike Train C, # spikes : " + str(output_spikes.count(1)))

'''
#### save figure to folder
figure = plt.gcf() # get current figure
figure.set_size_inches(10, 10)
plt.savefig("./results/" + str(i_inj) + ", weights " + str(weights) + " output.jpg",dpi = 100)
'''
plt.show()