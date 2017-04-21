import math
import matplotlib.pyplot as plt
import numpy as np

class izhikevich:

     def __init__ (self):
          self.v_rest = -60.
          self.v_th = -40.
          self.v_peak = 35.
          self.C = 100.
          self.a = 0.03
          self.b = -2.
          self.c = -50.
          self.d = 100.
          self.tau = 0.1            # time update factor DELTA_T
          self.u_0 = 0.             # recovery factor
          self.ms                   # ms to run simputaion
          self.steps                # number of steps to be n
          self.spike_train = []     # array of 0's and 1's representing spike train
          self.out = []             # array of volatges after running simultion
          self.current_step = 0     # number of steps executed so far
          self.v = self.v_rest      # v represents the membrane potential of the neuron
          self.u = self.u_0         # u represents a membrane recovery variable


     #################################################################################
     ### Implementation  from : 
     ### A Spiking Network that Learns to Extract Spike Signatures from Speech Signals
     ### run simulation for set of injected current values for given time
     #################################################################################
     def input_snn(self, i_inj, run_sim):
          self.ms = run_sim                       # ms to run simputaion
          self.steps = int(self.run_sim/self.tau) # number of steps to be n

          if self.steps < len(i_inj):  # if length of array is less than the required steps
               self.steps = len(i_inj) # run the simulation for array length, don't throw exception

          for i in xrange(self.ms):
               self.v = self.v + (self.tau * ((self.v-self.v_rest)*(self.v-self.v_th) + -self.u + (i_inj))/self.C)

               self.u = self.u + (self.tau * self.a*(self.b*(self.v-self.v_rest) - self.u))
               self.out.append(self.v)

               if self.v >= self.v_peak:
                    self.v = self.c
                    self.u = self.u + self.d


     #################################################################################
     ### Implementation  from : 
     ### A Spiking Network that Learns to Extract Spike Signatures from Speech Signals
     ### run simulation for one injected current value
     #################################################################################
     def input_snn_one_step(self, i_inj):
          
          self.v = self.v + (self.tau * ((self.v-self.v_rest)*(self.v-self.v_th) + -self.u + (i_inj))/self.C)

          self.u = self.u + (self.tau * self.a*(self.b*(self.v-self.v_rest) - self.u))
          self.out.append(self.v)

          if self.v >= self.v_peak:
               self.v = self.c
               self.u = self.u + self.d

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
          for i in xrange(len(self.out)):
               if self.out[i] > thresold:
                    self.spike_train.append(10)
               else:
                    self.spike_train.append(0)
          return self.spike_train