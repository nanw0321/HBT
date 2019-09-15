from Functions_gpu import *
import matplotlib.pyplot as plt

# simulation variables
Natom = 1000
tpulse = 0.1
dsamp = 10 * 1e-9
pulseshape = 'Gaussian'

# initialize image
img = np.zeros((Npxl,Npxl))

# create sub-pixel grids
nsubpxl = 10							# number of subpixels corresponding to each real pixel/dimension
nwindow = 10							# number of windows in each dimension to prevent memory overflow
Npxlw = int(Npxl*nsubpxl/nwindow)		# number of sub-pixels in each window/dimension
Npxlw_real = int(Npxl/nwindow)			# number of real pixels in each window/dimension
xdetw = np.linspace(0,detsize,int(Npxl*nsubpxl)).reshape((nwindow,Npxlw))
ydetw = xdet.copy()						# reshape for detector pixels in each window

# initialize GPU arrays before loop
rsg = cuda.device_array(shape=(Natom,Npxlw,Npxlw),dtype='float64')
Fsg = cuda.device_array(shape=(Natom,Npxlw,Npxlw),dtype='complex128')
tag = cuda.device_array(shape=(Natom,Npxlw,Npxlw),dtype='float64')
taming = cuda.device_array(shape=(Natom,21,21),dtype='float64')
rsming = cuda.device_array(shape=(Natom,21,21),dtype='float64')
tamaxg = cuda.device_array(shape=(Natom,21,21),dtype='float64')
rsmaxg = cuda.device_array(shape=(Natom,21,21),dtype='float64')

# generate source information and feed to GPU
if pulseshape == 'SASE': tlit, I_ratio, non_linearity = SASE(tpulse,Natom)
if pulseshape == 'SASEmono': tlit, I_ratio, non_linearity = SASEmono(tpulse,Natom)
if pulseshape == 'Gaussian': tlit, I_ratio, non_linearity = Gaus(tpulse,Natom)
if pulseshape == 'Gaussmono': tlit, I_ratio, non_linearity = Gausmono(tpulse,Natom)
if pulseshape == 'Square': tlit, I_ratio, non_linearity = square(tpulse,Natom)
if pulseshape == 'Squaremono': tlit, I_ratio, non_linearity = squaremono(tpulse,Natom)
if pulseshape == 'delta': tpulse = 0.; tlit, I_ratio, non_linearity = delta(tpulse,Natom)
(xs,ys,zs,ts,phs,omgs,ks,taus) = sframe(tpulse,Natom,dsamp,tlit)

# to gpu
nsubpxlg = cuda.to_device(nsubpxl)
rg = cuda.to_device(r)
xsg = cuda.to_device(xs)
ysg = cuda.to_device(ys)
zsg = cuda.to_device(zs)
tsg = cuda.to_device(ts)
phsg = cuda.to_device(phs)
omgsg = cuda.to_device(omgs)
ksg = cuda.to_device(ks)
tausg = cuda.to_device(taus)
dtslistg = cuda.to_device(dtslist)

# taxis calculation
xxmin = np.arange(21)*pxlsize/10; xxmin = xxmin-xxmin.mean()
yymin = xxmin.copy()

xxmax = np.arange(21)*pxlsize/10; xxmax = xxmax-xxmax.mean() + detsize/2
yymax = xxmax.copy()

xxming = cuda.to_device(xxmin); yyming = cuda.to_device(yymin)
xxmaxg = cuda.to_device(xxmax); yymaxg = cuda.to_device(yymax)

getaxis[(32,32),(32,32)](taming,rsming,xsg,ysg,zsg,xxming,yyming,rg[0])
getaxis[(32,32),(32,32)](tamaxg,rsmaxg,xsg,ysg,zsg,xxmaxg,yymaxg,rg[0])
tamin = taming.copy_to_host()
tamax = tamaxg.copy_to_host()

trange = tamax.max()-tamin.min() + 4*dtslist.max()
mm = int(np.round(trange*1e16)+1)
tmm = np.arange(mm)/10
taxis = tmm*1e-15+tamin.min()
taxisg = cuda.to_device(taxis)

# Loop through each window and calculate the intensities
xdetwg = cuda.to_device(xdetw)
ydetwg = cuda.to_device(ydetw)

for iwin in np.arange(nwindow):
	xlb = iwin*Npxlw_real
	xub = xlb + Npxlw_real
	for jwin in np.arange(nwindow):
		ylb = jwin*Npxlw_real
		yub = ylb + Npxlw_real
		# spatial component
		spatial[(32,32),(32,32)](rsg,Fsg,tag,xsg,ysg,zsg,ksg,tsg,phsg,xdetwg[iwin],ydetwg[jwin],rg[0])
		# temporal component
		T2g = cuda.device_array(shape=(mm,Natom),dtype='complex128')
		temporal[(64,32),(32,32)](T2g,omgsg,taxisg)
		# field strength calculation in each slice
		Es1g = cuda.device_array(shape=(mm,Npxlw,Npxlw),dtype='complex128')
		Es2g = cuda.device_array(shape=(mm,Npxlw,Npxlw),dtype='complex128')
		Es3g = cuda.device_array(shape=(mm,Npxlw,Npxlw),dtype='complex128')
		for i in range(mm):
			getslice[(32,32),(32,32)](Es1g[i],Es2g[i],Es3g[i],Fsg,taxisg[i],T2g[i],tag,tausg,dtslistg)
		# speckle pattern in current window (sub-pixel)
		Imgwg = cuda.device_array(shape=(Npxlw,Npxlw),dtype='float64')
		get_img[(32,32),(32,32)](Imgwg,Es1g,Es2g,Es3g)
		Imgw = Imgwg.copy_to_host()
		# speckle pattern in current window (real-pixel)
		sum1 = Imgw.reshape(Npxlw_real,nsubpxl,Npxlw).sum(axis=1)
		sum2 = (sum1.T.reshape(Npxlw_real,nsubpxl,Npxlw_real).sum(axis=1)).T
		img[xlb:xub,ylb:yub] = sum2

# save pattern
with h5py.File('pattern.h5') as f:
	f.create_dataset('pattern',data=img)