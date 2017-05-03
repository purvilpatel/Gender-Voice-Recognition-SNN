from izhikevich_original import izhikevich as neuron
import util

import matplotlib.pyplot as plt

A = neuron()

tau = 0.1
time = 100 #ms
steps = int(time / tau)

a_i_inj = 21840

for i in xrange(0,steps):
    A.input_snn_one_step(a_i_inj)

spikes = A.generate_spike_train(35)

temp_spike_time = []
for i in xrange(0,steps):
	if(spikes[i] > 0):
		temp_spike_time.append(float(i) * tau) # convert index to timestamp

print temp_spike_time

plt.figure()
plt.subplot(211)
plt.title("Spike Coding A")
plt.plot(A.get_output())

plt.subplot(212)
plt.plot(spikes)
plt.title("Spike Train A : " + str(spikes.count(1)))
plt.show()