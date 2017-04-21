import math
import matplotlib.pyplot as plt
import numpy as np

a = 0.02
b = 0.2
c = -65
d = 8.0
tau = 0.01

##########################################################
###  Simple Model of Spiking Neurons
###  http://www.izhikevich.org/publications/spikes.pdf
###  http://www.izhikevich.org/publications/izhikevich.m
##########################################################
def input_original_paper(i_inj):
	v=-70.0;
	u=-20.0;
	out = []
	for i in xrange(300):
		v = v + (tau * ((0.04 * math.pow(v,2)) + (5 * v) + 140 - u + i_inj))

		u = u + (tau *a*(b*v - u))
		out.append(v)

		print v

		if v >= 30:
			v = c
			u = u + d

	return out

out = input_original_paper(100)

plt.title(tau)
plt.plot(out)
plt.show()