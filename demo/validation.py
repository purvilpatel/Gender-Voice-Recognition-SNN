import util
#from markov_model
import markov_model as hmm
import numpy as np
import warnings
import os

SPEAKER_GENDER_FEMALE = 0
SPEAKER_GENDER_MALE = 1
SPEAKER_GENDER_ERROR = -1



folder = "./validation_data/scottish/"

fldr = "scottish"

male = open(str(fldr+"_validation_male.csv"), "w")
female = open(str(fldr+"_validation_female.csv"), "w")

for filename in os.listdir(folder):
    print filename
    label = util.get_sample_label(filename)

    buffer = util.get_injected_current(folder, filename, 70)

    HMM = hmm.markov_model()
    vocab = HMM.match(buffer)

    if label == SPEAKER_GENDER_MALE:
        male.write("%s," % filename)
        male.write("%s," % str(label))
        male.write("%s," % str(vocab))
        male.write("\n")
    elif label == SPEAKER_GENDER_FEMALE:
        female.write("%s," % filename)
        female.write("%s," % str(label))
        female.write("%s," % str(vocab))
        female.write("\n")