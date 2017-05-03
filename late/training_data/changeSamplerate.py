import numpy as np
import scipy.io.wavfile as wavfile
from scipy import interpolate
import os
import wave

inputfolder = "./"
outfolder = "./cut_samples/"

print len(os.listdir(outfolder))

for file in os.listdir(outfolder):
	os.remove(outfolder+file)

for file in os.listdir(inputfolder):
	if file.find(".wav") == -1:
		continue
	win= wave.open(inputfolder+file, 'rb')
	old_samplerate, old_audio = wavfile.read(inputfolder+file)
	duration = old_audio.shape[0] / old_samplerate
	t0, t1= 0.0, 0.83	 # cut audio between one and two seconds
	i = 0
	while t1+0.82<duration:
		i = i + 1
		wout= wave.open(outfolder+file.replace(".wav","_")+str(i)+".wav", 'wb')	
		s0, s1= int(t0*win.getframerate()), int(t1*win.getframerate())
		win.readframes(s0)
		frames= win.readframes(s1-s0)
		wout.setparams(win.getparams())
		wout.writeframes(frames)
		wout.close()
		t0 = t0 + 0.83
		t1 = t0 + 0.82
	win.close()
	
print len(os.listdir(outfolder))