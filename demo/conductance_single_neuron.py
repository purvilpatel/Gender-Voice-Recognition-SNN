import matplotlib.pyplot as plt
import math

tau = 0.1
time = 20 #ms
steps = int(time / tau)

a_i_inj = 250

K_SYN = 1
TAU = 2

def conductance():
	G = []
	for i in xrange(0,steps):
		t = float(i)/10.0
		G.append(K_SYN * t * math.exp(-t/TAU))
	return G

G = conductance()

plt.figure()
plt.title("Conductance change over time")
plt.plot(G)
plt.show()