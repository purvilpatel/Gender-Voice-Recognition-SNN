import math
import matplotlib.pyplot as plt
import numpy as np

class izhikevich:

    def __init__ (self):
        self.v_rest = -70.
        self.a = 0.02
        self.b = 0.2
        self.c = -65.
        self.d = 8.
        self.tau = 0.1            # time update factor DELTA_T
        self.u_0 = -20.           # recovery factor
        self.ms = 0               # ms to run simputaion
        self.steps = 0            # number of steps to be n
        self.spike_train = []     # array of 0's and 1's representing spike train
        self.out = []             # array of volatges after running simultion
        self.current_step = 0     # number of steps executed so far
        self.v = self.v_rest      # v represents the membrane potential of the neuron
        self.u = self.u_0         # u represents a membrane recovery variable
        self.prev_v = self.v


     #################################################################################
     ### Implementation  from : 
     ### A Spiking Network that Learns to Extract Spike Signatures from Speech Signals
     ### run simulation for one injected current value
     #################################################################################
    def input_snn_one_step(self, i_inj):
        self.prev_v = self.v

    	self.v = self.v + (self.tau * ((0.04 * math.pow(self.v,2)) + (5 * self.v) + 140 - self.u + i_inj))
    	self.u = self.u + (self.tau *self.a*(self.b*self.v - self.u))

    	if self.v >= 35:
    		self.v = self.c
    		self.u = self.u + self.d

    	self.out.append(self.v)

    	self.current_step = self.current_step + 1


     #########################################################
     ### Return the output array of volatages after simulation
     #########################################################
    def get_output(self):
     	return self.out

     ################################################################
     ### Return the current value of output volatage after simulation
     ################################################################
    def get_output_one_step(self):
        return self.out[self.current_step-1]

     ######################################################################
     ### Return spike train of 0's and 1's according to specified threshold
     ######################################################################
    def generate_spike_train(self, thresold):
        if len(self.spike_train) == len(self.out):
            return self.spike_train
        else:
            self.spike_train = []
        for i in xrange(len(self.out)):
            if self.out[i] > thresold:
                self.spike_train.append(1)
            else:
                self.spike_train.append(0)
        return self.spike_train

    def get_prev_voltage(self):
        return self.prev_v

##########################################################
###  Simple Model of Spiking Neurons
###  http://www.izhikevich.org/publications/spikes.pdf
###  http://www.izhikevich.org/publications/izhikevich.m
##########################################################
'''def input_original_paper(i_inj):
	
		v = v + (tau * ((0.04 * math.pow(v,2)) + (5 * v) + 140 - u + i_inj))

		u = u + (tau *a*(b*v - u))
		
		if v >= 30:
			v = c
			u = u + d

			return v

output = []
for i in xrange(0, steps):
	out = input_original_paper(40)
	output.append(out)


plt.title(tau)
plt.plot(output)
plt.show()'''