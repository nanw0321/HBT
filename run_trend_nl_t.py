import numpy as np
import os, h5py
from tqdm import tqdm
import winsound

njob = 500
Natom = 1000
t_low = 0.1
t_high = 100.
dsamp = 10 * 1e-9 		# sample thickness (m)

Tmodel = 'Gaus'
dirname = 'Plots/3.Trend_time/'

dircase = dirname + '{}/'.format(Tmodel)
if not os.path.exists(dircase):
	os.mkdir(dircase)
for j, numjob in tqdm(enumerate(range(njob))):
	tpulse = np.exp(np.random.uniform(low=np.log(t_low),high=np.log(t_high)))
	os.system('python run_parse_gpu.py -j {} -t {} -N {} -T "{}" -D "{}"'.format(
		numjob, tpulse, Natom, Tmodel, dircase))

# winsound.Beep(1500, 2000)

# # check if any run is missing
# for Tmodel in Tmodels:
# 	print(Tmodel)
# 	dircase = dirname + '{}/'.format(Tmodel)
# 	for i, tpulse in enumerate(tpulse_list):
# 		for j, numjob in enumerate(range(njob)):
# 			with h5py.File(dircase+'{}atom_{}fs_{}_{}nm.h5'.format(Natom,float(tpulse),Tmodel,round(dsamp*1e9,2)), 'r') as f:
# 				try: print('   ',Tmodel, tpulse, numjob,f['{}/contrast'.format(numjob)][0])
# 				except:
# 					print('   ',Tmodel, tpulse, numjob, 'missing')
# 					os.system('python run_parse_gpu.py -j {} -t {} -N {} -T "{}" -D "{}"'.format(
#  						numjob, tpulse, Natom, Tmodel, dircase))

   

# XCS ongoin experiment
# there might be some clusters in the metal salt that the theory people are not sure about.
# we don't want to simply characterize a specific sample.
# the experiment depends on what q we want to be at since we are on a fixed coherence bandwidth -> sample thickness
# want it to be a material science exp. not an x-ray science exp. since whenever we change the bond spacing, everything changes
# dynamic variables: heating etc.
# time separations: longer is better. Double bunch 30ns-ish is good, 200ns-ish is pushing the limit, longer than that proly not gonna work
# would it help to put sample in capillary and heat up a lot? Sample will be molten at 120C stage temp.
# the capillary might not heat up uniformly.
# sample: polymer w/ dynamic bonds, the calculating dynamics from viscosity can give an upper limit for temp setting.
# increasing the temperature wouldn't necessarily change the time scale for the bonds.
# md simulation can help figure out if the bond dynamics and sample dynamics are distinguishable at certain temperatures.
# XCS doesn't have a built-in heater/heating stage, but Peihao and Jerry built a heater that's capable of going up to a few hundred degrees.
# Paul has a heater at Argon but no one can find it due to new material science building.
