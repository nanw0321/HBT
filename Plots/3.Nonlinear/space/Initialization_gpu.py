import numpy as np

# x-ray parameters and constants
c = 299792458								# m/s
hbar = 6.582119514e-16						# eV*s
r = 1										# source to detector distance (m)
depthabs = 20e-6							# samnple absorption depth (m)

# k-alpha 1, k-alpha 2, k-beta 1,3
evs1 = 6403.8; As1 = 100; ws1 = 2.55; dts1 = 4e-15/ws1; ks1 = evs1/hbar/c
evs2 = 6390.8; As2 = 50; ws2 = 3.00; dts2 = 4e-15/ws2; ks2 = evs2/hbar/c
evs3 = 7058.0; As3 = 17; ws3 = 3.59; dts3 = 4e-15/ws3; ks3 = evs3/hbar/c

# convert information to arrays
evlist = np.asarray([evs1, evs2, evs3]) 	# photon energies
plist = np.asarray([As1, As2, As3])
plist = plist/plist.sum() 					# probability densities
omglist = evlist/hbar						# angular frequencies
dtslist = np.asarray([dts1,dts2,dts3])

# Focal spot information (in sample coordinate)
spotpos = 0e-6								# focal spot average position
spotsig = 2e-6								# focal spot sigma
spotsep = 0									# focal spot separation (0 if single spot)

# Detector pixel information
pxlsize = 5e-6								# single pixel size (m)
Npxl = 100									# number of pixels in each dimension
detsize = pxlsize*Npxl						# detector size (m)

xdet, ydet = np.indices((Npxl,Npxl))
xdet = (xdet-xdet.mean())*pxlsize
ydet = (ydet-ydet.mean())*pxlsize

# Detector sub-pixel information
nsubpxl = 10								# number of sub-pixels per pixel per dimension
nwindow = 10								# number of windows when evaluating speckle patter to prevent memory overflow
subpxlsize = pxlsize/nsubpxl			# single sub-pixel size(um)
Nsubpxl = int(nsubpxl*Npxl) 				# number of sub-pixels per dimension
Nsubpxl_w = int(Nsubpxl/nwindow)			# number of sub-pixels in each window per dimension
Npxl_w = int(Npxl/nwindow)					# number of pixels in each window per dimension

xdetsub, ydetsub = np.indices((Nsubpxl,Nsubpxl))
xdetsub = (xdetsub-xdetsub.mean())*subpxlsize
ydetsub = (ydetsub-ydetsub.mean())*subpxlsize

