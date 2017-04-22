from izhikevich import izhikevich as neuron
import matplotlib.pyplot as plt

A = neuron()

tau = 0.1
time = 1000 #ms
steps = int(time / tau)

a_i_inj = 82

for i in xrange(0,steps):
    A.input_snn_one_step(a_i_inj)

spikes = A.generate_spike_train(35)

plt.figure()
plt.subplot(211)
plt.title("Spike Coding A")
plt.plot(A.get_output())

plt.subplot(212)
plt.plot(spikes)
plt.title("Spike Train A")
plt.show()
