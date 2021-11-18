import h5py, os, time, math
#os.environ['NUMBAPRO_CUDALIB']=r"C:\Users\NanW\.conda\envs\GPU\Library\bin"
from Initialization_gpu import *
from scipy.special import erf
from numba import cuda
print(cuda.gpus)

##########
# x-ray temporal structure
##########
def delta(tpulse,Natom):
	tlit = np.zeros(Natom)
	I_ratio = 1
	non_linearity = 1e6
	return tlit, I_ratio, non_linearity

def square(tpulse,Natom):
	if tpulse == 0: tpulse += 1e-3
	tlit = np.random.random(Natom)*tpulse
	I_ratio = 1
	non_linearity = 1/tpulse
	return tlit, I_ratio, non_linearity

def squaremono(tpulse,Natom):
	if tpulse == 0: tpulse += 1e-3
	tsamp = np.arange(1e7)/1e7 *(tpulse+20)						# sampling range
	tsamp = tsamp-tsamp.mean()									# shifted sampling range (centering at t=0)
	psamp = -erf((tsamp-tpulse/2)/(2**0.5 * 4.0/2.36)) + erf((tsamp+tpulse/2)/(2**0.5 * 4.0/2.36))
	psamp = psamp/psamp.sum()
	tlit = np.random.choice(tsamp,p=psamp,size=Natom)
	I_ratio = 1
	non_linearity = np.sum(psamp**2)
	return tlit, I_ratio, non_linearity

def Gaus(tpulse,Natom):
	if tpulse == 0: tpulse += 1e-3
	width = tpulse/2.36
	tlit = np.random.normal(0,width,Natom)
	# peak pulse intensity / average pulse intensity for non-linear effect observation
	tt = np.linspace(-tpulse/2-width,tpulse/2+width,int(tpulse*10000))
	I_t = np.exp(-tt**2/(2*width**2))
	I_ratio = I_t.max()/I_t.mean()
	non_linearity = 1/(2*np.sqrt(np.pi)*width)
	return tlit, I_ratio, non_linearity

def Gausmono(tpulse,Natom):
	if tpulse == 0: tpulse += 1e-3
	width = (tpulse**2+4.0**2)**0.5/2.36		# assume super short pulse through 2 bounce C111 stretches to 4fs
	tlit = np.random.normal(0,width,Natom)
	# peak pulse intensity / average pulse intensity for non-linear effect observation
	tt = np.linspace(-tpulse/2-width,tpulse/2+width,int(tpulse*10000))
	I_t = np.exp(-tt**2/(2*width**2))
	I_ratio = I_t.max()/I_t.mean()
	non_linearity = 1/(2*np.sqrt(np.pi)*width)
	return tlit, I_ratio, non_linearity

def SASE(tpulse,Natom):
	if tpulse == 0: tpulse += 1e-3
	tcs = np.random.random(2000) * 200. - 100.						# uniformly sampled pink beam SASE spike positions
	Amps = np.random.random(2000)
	Amps = Amps * np.exp(-(tcs**2/(2*(tpulse/2.36)**2))**2)
	Amps = Amps/Amps.sum()
	index = (Amps>=1e-4)				# mask for pulse duration window
	numSpike = index.sum()
	if numSpike==0:
		tcs = np.zeros(1)
		Amps = np.ones(1)
		index = np.asarray([True])
		numSpike = 1

	tcs = tcs[index]; Amps = Amps[index]
	widths = np.random.uniform(low=0.08,high=0.12,size=numSpike)
	Ns = np.rint(Amps*Natom)
	if Ns.sum() < Natom:
		Ndiff = int(Natom - Ns.sum())
		inds = np.random.choice(np.arange(numSpike),size=Ndiff)
		for ind in inds:
			Ns[ind] += 1

	if Ns.sum() > Natom:
		while Ns.sum() != Natom:
			Ns[np.random.choice(np.argwhere(Ns>=1)[:,0])] -= 1
	Ns = Ns.astype(np.int64)

	tlit = []
	for i in range(numSpike):
		tlit_piece = np.random.normal(tcs[i],widths[i],Ns[i])
		tlit.append(tlit_piece)
	tlit = np.concatenate(tlit); tlit = tlit-tlit.mean()

	nbins = int((tlit.max()-tlit.min())*20)

	# peak pulse intensity / average pulse intensity for non-linear effect observation
	tt = np.linspace(-tpulse,tpulse,nbins)
	I_t = np.zeros_like(tt)
	for j in range(numSpike):
		I_t += Amps[j] * np.exp(-(tt-tcs[j])**2/(2*widths[j]**2))
	I_ratio = I_t.max()/I_t.mean()
	non_linearity = np.sum((I_t/I_t.sum())**2)
	return tlit, I_ratio, non_linearity

def SASEmono(tpulse,Natom):
	if tpulse == 0: tpulse += 1e-3
	tcs = np.random.random(150) * 1000							# randomly throw in 150 centers over 1ps
	tcs = tcs[(tcs>5)&(tcs<5+tpulse)]							# crop window to get actual centers of Gaussian
	try: tcs[0]
	except: tcs = np.random.random(1)*tpulse+5.					# if none, make one
	numGaus = int(tcs.size)

	widths = np.random.uniform(low=1.65,high=1.75,size=numGaus)	# rms widths assuming mono stretches to 4fs FW (1.7fs rms)
	Amps = np.random.random(numGaus); Amps = Amps/Amps.sum()	# amplitudes of Gaussian
	Ns = np.rint(Amps * Natom)									# number in each Gaussian
	# if total number of atoms in each slice does not equal to Natom, randomly fill in the missing ones
	if Ns.sum() < Natom:
		Ndiff = int(Natom - Ns.sum())
		inds = np.random.choice(np.arange(numGaus),size=Ndiff)
		for ind in inds:
			Ns[ind] += 1
	if Ns.sum() > Natom:
		while Ns.sum() != Natom:
			indlist = np.arange(numGaus)
			plist = Ns/Ns.sum()
			ind = np.random.choice(indlist,p=plist)
			Ns[ind] -= 1							
	Ns = Ns.astype(np.int64)
	tlit = []
	# randomly sample light up time for each atom
	for i in range(numGaus):
		try:
			tlit_piece = np.random.normal(tcs[i],widths[i],Ns[i])
			tlit.append(tlit_piece)
		except:
			print(i)
			print(tcs[i])
			print(widths[i])
			print(Ns[i])
	tlit = np.concatenate(tlit); tlit = tlit-tlit.mean()

	# peak pulse intensity / average pulse intensity for non-linear effect observation
	tt = np.linspace(0,tpulse+10,int(tpulse*100000))
	I_t = np.zeros_like(tt)
	for j in range(numGaus):
		I_t += Amps[j] * np.exp(-(tt-tcs[j])**2/(2*widths[j]**2))
	I_ratio = I_t.max()/I_t.mean()
	non_linearity = np.sum((I_t/I_t.sum())**2)
	return tlit, I_ratio, non_linearity

##########
# source parameter generation
##########
def sframe(tpulse,Natom,dsamp,tlit,spotsig):
	radius = np.random.normal(spotpos,spotsig,Natom)		# radial positions (Gaussian)
	ths = 2 * np.pi * np.random.random(Natom)				# angular positions (Uniform)
	phs = 2 * np.pi * np.random.random(Natom)				# initial phases	(Uniform)
	pols = 2 * np.pi * np.random.random(Natom)				# light polarization (Uniform)
	omgs = np.random.choice(omglist,p=plist,size=Natom)		# angular frequencies (Sampled)	

	ks = omgs/c 											# wavenumbers
	taus = np.zeros_like(omgs)								# lifetimes
	taus[omgs==omglist[0]] = dts1
	taus[omgs==omglist[1]] = dts2
	taus[omgs==omglist[2]] = dts3

	# source position (in sample coordinate)
	xsource = radius * np.cos(ths)
	ysource = radius * np.sin(ths) * np.sqrt(2)
	K = 1/(1-np.exp(-dsamp*np.sqrt(2)/depthabs))			# coefficient for absorption depth calculation
	cdf = np.random.sample(Natom)							# cumulative probability (Uniform)
	zsource = -depthabs * np.log(1-cdf/K)					# depth in sample (x-ray direction)
	#zsource = np.random.random(Natom)*dsamp*np.sqrt(2)		# depth ignoring absorption depth

	ysource[:int(Natom/2)] = ysource[:int(Natom/2)] + spotsep/2		# offset focal spots
	ysource[int(Natom/2):] = ysource[int(Natom/2):] - spotsep/2

	# source position (in lab coordinate)
	xs = xsource
	ys = ysource/np.sqrt(2) - zsource
	zs = -ysource/np.sqrt(2)
	# Actual light-up time including path-length difference (t=0 when incident x-ray hits origin in lab coordinate)
	ts = tlit*1e-15 - ys/c
	return xs,ys,zs,ts,phs,omgs,ks,taus,pols

##########
# time axis calculation
##########
dtype1 = 'float64[:,:,:],float64[:,:,:],'
dtype2 = 'float64[:],float64[:],float64[:],float64[:],'
dtype3 = 'float64[:,:],float64[:,:],float64'
@cuda.jit('void('+dtype1+dtype2+dtype3+')')
def getaxis(rs_roi,tas_roi,
	xs,ys,zs,ts,
	xdet_roi,ydet_roi,r):
	i, j = cuda.grid(2)
	Nx_roi, Ny_roi = xdet_roi.shape; Natom = xs.size
	# loop through pixels
	if i<Nx_roi and j<Ny_roi:
		# loop through atoms
		for natom in range(Natom):
			# distance between atom and pixel
			rs_roi[natom,i,j] = math.sqrt(
				(xs[natom]-xdet_roi[i,j])**2+(ys[natom]-ydet_roi[i,j])**2+(zs[natom]-r)**2
				)
			# spatial component of the spherical wave: 1/r*exp[i(kr+phi)]
			tas_roi[natom,i,j] = ts[natom] + rs_roi[natom,i,j]/299792458.

##########
# spatial component calculation
##########
dtype1 = 'float64[:,:,:],complex128[:,:,:],float64[:,:,:],'
dtype2 = 'float64[:],float64[:],float64[:],float64[:],float64[:],float64[:],'
dtype3 = 'float64[:,:],float64[:,:],float64'
@cuda.jit('void('+dtype1+dtype2+dtype3+')')
def spatial(rs,Fs,tas,
	xs,ys,zs,ks,ts,phs,
	xdet,ydet,r):
	i, j = cuda.grid(2)
	Nx,Ny = xdet.shape; Natom = xs.size
	# loop through pixels
	if i<Nx and j<Ny:
		# loop through atoms
		for natom in range(Natom):
			# distance between atom and pixel
			rs[natom,i,j] = math.sqrt(
				(xs[natom]-xdet[i,j])**2+(ys[natom]-ydet[i,j])**2+(zs[natom]-r)**2
				)
			# spatial component of the spherical wave: 1/r*exp[i(kr+phi)]
			Fs[natom,i,j] = complex(1/rs[natom,i,j])*(
				math.cos(ks[natom]*rs[natom,i,j]+phs[natom])+1j*math.sin(ks[natom]*rs[natom,i,j]+phs[natom])
				)
			tas[natom,i,j] = ts[natom] + rs[natom,i,j]/299792458.

##########
# temporal component calculation
##########
# phase of all atoms
@cuda.jit('void(complex128[:,:],float64[:],float64[:])')
def temporal(T2,omgs,taxis):
	nslice,natom = cuda.grid(2)
	Nslice, Natom = T2.shape
	if nslice<Nslice and natom<Natom:
		T2[nslice,natom] = math.cos(omgs[natom]*taxis[nslice])+1j*math.sin(omgs[natom]*taxis[nslice])

##########
# speckle pattern simulation
##########
# field strength per time slice, reinitialize if changing parameters
dtype1 = 'complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],'
dtype2 = 'float64,complex128[:],float64[:,:,:],float64[:],float64[:],float64[:],float64[:]'
@cuda.jit('void('+dtype1+dtype2+')')
def getslice(E1slice,E2slice,E3slice,Fs,
	t,T2slice,ta,taus,dtslist,polx,poly):
	i,j = cuda.grid(2)
	Natom,Nx,Ny = Fs.shape
	# loop through pixels
	if i<Nx and j<Ny:
		# loop through atoms
		for natom in range(Natom):
			# check if wavefront has arrived
			if t>=ta[natom,i,j] and t<ta[natom,i,j]+8*taus[natom]:
				# if arrived, calculate time-dependent amplitude
				A = complex(math.exp(-(t-ta[natom,i,j])/taus[natom]))
				# check color and add to field container accordingly
				if taus[natom] == dtslist[0]:
					E1slice[i,j,0] += A * T2slice[natom] * Fs[natom,i,j] * polx[natom]
					E1slice[i,j,1] += A * T2slice[natom] * Fs[natom,i,j] * poly[natom]
				if taus[natom] == dtslist[1]:
					E2slice[i,j,0] += A * T2slice[natom] * Fs[natom,i,j] * polx[natom]
					E2slice[i,j,1] += A * T2slice[natom] * Fs[natom,i,j] * poly[natom]
				if taus[natom] == dtslist[2]:
					E3slice[i,j,0] += A * T2slice[natom] * Fs[natom,i,j] * polx[natom]
					E3slice[i,j,1] += A * T2slice[natom] * Fs[natom,i,j] * poly[natom]


# intensity per time slice (only used for comparing non-linear effect and speckle contrast)
@cuda.jit('void(float64[:,:,:],complex128[:,:,:,:],complex128[:,:,:,:],complex128[:,:,:,:])')
def get_imgs(Imgs,Es1,Es2,Es3):
	i,j = cuda.grid(2)
	Nslice,Nx,Ny,_ = Es1.shape
	if i<Nx and j<Ny:
		for nslice in range(Nslice):
			# adding intensities in two polarizations in each time slice
			Imgs[nslice,i,j] += abs(Es1[nslice,i,j,0])**2+abs(Es2[nslice,i,j,0])**2+abs(Es3[nslice,i,j,0])**2
			Imgs[nslice,i,j] += abs(Es1[nslice,i,j,1])**2+abs(Es2[nslice,i,j,1])**2+abs(Es3[nslice,i,j,1])**2

# overall intensity (image), reinitialize if changing parameters
@cuda.jit('void(float64[:,:],complex128[:,:,:,:],complex128[:,:,:,:],complex128[:,:,:,:])')
def get_img(Img,Es1,Es2,Es3):
	i,j = cuda.grid(2)
	Nslice,Nx,Ny,_ = Es1.shape
	if i<Nx and j<Ny:
		for nslice in range(Nslice):
			Img[i,j] += abs(Es1[nslice,i,j,0])**2+abs(Es2[nslice,i,j,0])**2+abs(Es3[nslice,i,j,0])**2
			Img[i,j] += abs(Es1[nslice,i,j,1])**2+abs(Es2[nslice,i,j,1])**2+abs(Es3[nslice,i,j,1])**2

# subpixel to real pixel
@cuda.jit('void(float64[:,:],float64[:,:],int64)')
def compression(Img,Imgw,nsubpxl):
	i,j = cuda.grid(2)
	Nx,Ny = Img.shape
	Nxw,Nyw = Imgw.shape
	if i <Nx and j<Ny:
		for ii in range(Nxw):
			for jj in range(Nyw):
				if (ii-ii%nsubpxl)==i and (jj-jj%nsubpxl)==j:
					Img[i,j]+=Imgw[ii,jj]