from Functions_gpu import *
import matplotlib.pyplot as plt
from tqdm import tqdm
#plt.ion()

tic = time.time()

# simulation variables
Natom = 1000
tpulse = 0.1
dsamp = 10 * 1e-9
pulseshape = 'Gaussian'

#siglow = 0.1; sighigh = 100		# focal spot size sigma range (um)
siglow = 1; sighigh = 100
lowlog = np.log(siglow); highlog = np.log(sighigh)

njobs = 19			# number of jobs
holder_on = 0		# using holder or not
io = 1				# storing simulation or not

# initialize gpu array
rsg = cuda.device_array(shape=(Natom,Npxl,Npxl),dtype='float64')
Fsg = cuda.device_array(shape=(Natom,Npxl,Npxl),dtype='complex128')
tag = cuda.device_array(shape=(Natom,Npxl,Npxl),dtype='float64')

# make directory for simulation results
dirname = pulseshape+'_non-linear_N={}_{}nm/'.format(Natom,dsamp*1e9)
try: os.stat(dirname)
except: os.mkdir(dirname)

# holder for repeating simulations
#source_holder = np.zeros((repspb,8,Natom))
if holder_on == 1:
	img_holder = np.zeros((njobs,Nsubpxl,Nsubpxl))
	sum_holder = np.zeros((njobs,Npxl,Npxl))
	non_linear_holder = np.zeros(njobs)
	Beamsize_holder = np.zeros(njobs)
	tlit_holder = np.zeros((njobs,Natom))
	contrast_holder = np.zeros(njobs)

for njob in tqdm(np.arange(njobs),desc='jobs'):
	njob += 181 	# offset if too many jobs and restarting in the middle
	spotsig = np.exp(np.random.uniform(low=lowlog,high=highlog))*1e-6		# uniformly sample focal spot size in log scale
	nl_s = 1/(4*np.pi*spotsig**2)
	# generate source information and feed to GPU
	if pulseshape == 'SASE': tlit, I_ratio, nl_t = SASE(tpulse,Natom)
	if pulseshape == 'SASEmono': tlit, I_ratio, nl_t = SASEmono(tpulse,Natom)
	if pulseshape == 'Gaussian': tlit, I_ratio, nl_t = Gaus(tpulse,Natom)
	if pulseshape == 'Gaussmono': tlit, I_ratio, nl_t = Gausmono(tpulse,Natom)
	if pulseshape == 'Square': tlit, I_ratio, nl_t = square(tpulse,Natom)
	if pulseshape == 'Squaremono': tlit, I_ratio, nl_t = squaremono(tpulse,Natom)
	if pulseshape == 'delta': tpulse = 0.; tlit, I_ratio, nl_t = delta(tpulse,Natom)
	(xs,ys,zs,ts,phs,omgs,ks,taus,pols) = sframe(tpulse,Natom,dsamp,tlit,spotsig)
	
	# to gpu
	nsubpxlg = cuda.to_device(nsubpxl)
	rg = cuda.to_device([r])
	xsg = cuda.to_device(xs)
	ysg = cuda.to_device(ys)
	zsg = cuda.to_device(zs)
	tsg = cuda.to_device(ts)
	phsg = cuda.to_device(phs)
	omgsg = cuda.to_device(omgs)
	ksg = cuda.to_device(ks)
	tausg = cuda.to_device(taus)
	polxg = cuda.to_device(np.cos(pols))
	polyg = cuda.to_device(np.sin(pols))
	dtslistg = cuda.to_device(dtslist)
	xdetsubg = cuda.to_device(xdetsub)
	ydetsubg = cuda.to_device(ydetsub)

	# calculate common time axis
	taxis_window_size = 100							# ROI size per dimension
	rsming = cuda.device_array(shape=(Natom,taxis_window_size,taxis_window_size),dtype='float64')
	tasming = cuda.device_array(shape=(Natom,taxis_window_size,taxis_window_size),dtype='float64')
	rsmaxg = cuda.device_array(shape=(Natom,taxis_window_size,taxis_window_size),dtype='float64')
	tasmaxg = cuda.device_array(shape=(Natom,taxis_window_size,taxis_window_size),dtype='float64')

	xxmin, yymin = np.indices((taxis_window_size,taxis_window_size))	# ROI taken at center of detector
	xxmin = xxmin-xxmin.mean(); xxmin = xxmin*subpxlsize
	yymin = yymin-yymin.mean(); yymin = yymin*subpxlsize

	xxmax, yymax = np.indices((taxis_window_size,taxis_window_size))	# ROI taken at corner of detector
	xxmax = xxmax-xxmax.mean(); xxmax = xxmax*subpxlsize + detsize/2
	yymax = yymax-yymax.mean(); yymax = yymax*subpxlsize + detsize/2

	xxming = cuda.to_device(xxmin); yyming = cuda.to_device(yymin)
	xxmaxg = cuda.to_device(xxmax); yymaxg = cuda.to_device(yymax)

	getaxis[(16,16),(16,16)](rsming,tasming,xsg,ysg,zsg,tsg,xxming,yyming,rg[0])
	getaxis[(16,16),(16,16)](rsmaxg,tasmaxg,xsg,ysg,zsg,tsg,xxmaxg,yymaxg,rg[0])

	trange = np.max(tasmaxg)-np.min(tasming) + 8*dtslist.max()
	mm = int(np.round(trange*1e16)+1)
	tmm = np.arange(mm)/10
	taxis = tmm*1e-15 + np.min(tasming)
	taxisg = cuda.to_device(taxis)

	# some temporal component
	T2g = cuda.device_array(shape=(mm,Natom),dtype='complex128')
	temporal[(64,32),(32,32)](T2g,omgsg,taxisg)

	# initialize image
	imgg = cuda.device_array(shape=(Nsubpxl,Nsubpxl),dtype='float64')

	# initialize GPU arrays before loop
	rswg = cuda.device_array(shape=(Natom,Nsubpxl_w,Nsubpxl_w),dtype='float64')
	Fswg = cuda.device_array(shape=(Natom,Nsubpxl_w,Nsubpxl_w),dtype='complex128')
	tawg = cuda.device_array(shape=(Natom,Nsubpxl_w,Nsubpxl_w),dtype='float64')

	# Loop through each window and calculate the intensities
	for iwin in tqdm(range(nwindow),desc='row'):
		# for jwin in tqdm(range(nwindow),desc='column'):
		for jwin in range(nwindow):
			# get pixel positions for current window
			xlb = iwin*Nsubpxl_w; xub = (iwin+1)*Nsubpxl_w
			ylb = jwin*Nsubpxl_w; yub = (jwin+1)*Nsubpxl_w
			xdetw = xdetsub[xlb:xub,ylb:yub]
			ydetw = ydetsub[xlb:xub,ylb:yub]
			xdetwg = cuda.to_device(np.ascontiguousarray(xdetw))
			ydetwg = cuda.to_device(np.ascontiguousarray(ydetw))

			# spatial component
			spatial[(16,16),(32,32)](rswg,Fswg,tawg,xsg,ysg,zsg,ksg,tsg,phsg,xdetwg,ydetwg,rg[0])
			# field strength calculation in each slice
			Es1wg = cuda.device_array(shape=(mm,Nsubpxl_w,Nsubpxl_w,2),dtype='complex128')
			Es2wg = cuda.device_array(shape=(mm,Nsubpxl_w,Nsubpxl_w,2),dtype='complex128')
			Es3wg = cuda.device_array(shape=(mm,Nsubpxl_w,Nsubpxl_w,2),dtype='complex128')
			for i in range(mm):
				getslice[(32,32),(32,32)](Es1wg[i],Es2wg[i],Es3wg[i],
					Fswg,taxisg[i],T2g[i],tawg,tausg,dtslistg,polxg,polyg)
			# speckle pattern in current window
			get_img[(32,32),(32,32)](imgg[xlb:xub,ylb:yub],Es1wg,Es2wg,Es3wg)

	img = imgg.copy_to_host()
	# recombine sub-pixels to get real pixel intensity (numerical integration)
	sum1 = img.reshape(Npxl,nsubpxl,Nsubpxl).sum(axis=1)
	sum2 = (sum1.T.reshape(Npxl,nsubpxl,Npxl).sum(axis=1)).T
	contrast = (sum2.std()/sum2.mean())**2

	# send to holder
	if holder_on == 1:
		img_holder[njob] = img
		sum_holder[njob] = sum2
		non_linear_holder[njob] = nl_t
		Beamsize_holder[njob] = spotsig
		tlit_holder[njob] = tlit
		contrast_holder[njob] = contrast

	# I/O
	if io == 1:
		with h5py.File(dirname+pulseshape+'_r={}_sig={}-{}um.h5'.format(r,siglow,sighigh),'a') as f:
			grpname = '{}'.format(njob)
			grp = f.create_group(grpname)
			grp.create_dataset('source', data=np.stack([xs,ys,zs,tlit,ts,phs,omgs,ks,taus,pols]))
			grp.create_dataset('pattern_sub',data=img)
			grp.create_dataset('pattern',data=sum2)
			grp.create_dataset('contrast', data=[contrast])
			grp.create_dataset('others', data=[spotsig,nl_t,nl_s])
