from izhikevich import izhikevich as neuron
import matplotlib.pyplot as plt

A = neuron()
B = neuron()

tau = 0.1
time = 10 #ms
steps = int(50 / tau)

a_i_inj = 250

for i in xrange(0,steps):
    A.input_snn_one_step(a_i_inj)

    a_out = A.get_output_one_step()

    B.input_snn_one_step(-a_out)


spikes = A.generate_spike_train(35)

plt.figure()
plt.subplot(221)
plt.title("Spike Coding A")
plt.plot(A.get_output())

plt.subplot(222)
plt.plot(spikes)
plt.title("Spike Train A")

spikes = B.generate_spike_train(35)

plt.subplot(223)
plt.title("Spike Coding B")
plt.plot(B.get_output())

plt.subplot(224)
plt.plot(spikes)
plt.title("Spike Train B")
plt.show()