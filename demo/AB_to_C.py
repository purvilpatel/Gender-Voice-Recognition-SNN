from izhikevich import izhikevich as neuron
import matplotlib.pyplot as plt

A = neuron()
B = neuron()
C = neuron()

tau = 0.1
time = 50 #ms
steps = int(50 / tau)

a_i_inj = 150
b_i_inj = 250

for i in xrange(0,steps):
    A.input_snn_one_step(a_i_inj)
    B.input_snn_one_step(b_i_inj)

    a_out = A.get_output_one_step()
    b_out = B.get_output_one_step()
    
    C.input_snn_one_step(-(a_out+b_out))


spikes = A.generate_spike_train(35)

plt.figure()
plt.subplot(321)
plt.title("Spike Coding A")
plt.plot(A.get_output())

plt.subplot(322)
plt.plot(spikes)
plt.title("Spike Train A")

spikes = B.generate_spike_train(35)

plt.subplot(323)
plt.title("Spike Coding B")
plt.plot(B.get_output())

plt.subplot(324)
plt.plot(spikes)
plt.title("Spike Train B")

spikes = C.generate_spike_train(35)

plt.subplot(325)
plt.title("Spike Coding C")
plt.plot(C.get_output())

plt.subplot(326)
plt.plot(spikes)
plt.title("Spike Train C")
plt.show()