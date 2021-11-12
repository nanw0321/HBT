from Functions_gpu_subpxl import *
import time, argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-j", type = int, help = 'job number')
parser.add_argument("-t", type = float, help = 'pulse duration (fs)')
parser.add_argument("-N", type = int, help = '# of atoms')
parser.add_argument("-T", type = str, help = 'Delta, Square, Gaus, SASE, etc.')
parser.add_argument("-r", type = float, help = 'source radius rms (m)')
parser.add_argument("-D", type = str, help = 'directory')

args = parser.parse_args()

# simulation variables
numjob = 1 				# job ID
tpulse = 100.			# pulse duration (fs)
Natom = 1000			# number of atoms (photons)
dsamp = 10 * 1e-9 		# sample thickness (m)
Tmodel = 'Delta'		# incident pulse time structure
if_Mono = 1 			# monochromator?
tlit, I_ratio, nl_t = delta(tpulse, Natom)
dirname = './'

if args.j is not None:
	numjob = args.j
if args.t is not None:
	tpulse = args.t
if args.N is not None:
	Natom = args.N
if args.T is not None:
	Tmodel = args.T
else:
	print('temporal structure not given, using delta function w/o monochromator as demo')

if args.r is not None:
	spotsig = args.r
nl_s = 1/(4*np.pi*spotsig**2)

if args.D is not None:
	dirname = args.D

if Tmodel == 'Delta':	   tlit, I_ratio, nl_t = delta(tpulse, Natom); tpulse = 0.
if Tmodel == 'Square':		tlit, I_ratio, nl_t = square(tpulse, Natom)
if Tmodel == 'Square_mono': tlit, I_ratio, nl_t = squaremono(tpulse,Natom)
if Tmodel == 'Gaus':		tlit, I_ratio, nl_t = Gaus(tpulse,Natom)
if Tmodel == 'Gaus_mono':	tlit, I_ratio, nl_t = Gausmono(tpulse,Natom)
if Tmodel == 'SASE':		tlit, I_ratio, nl_t = SASE(tpulse,Natom)
if Tmodel == 'SASE_mono':	tlit, I_ratio, nl_t = SASEmono(tpulse,Natom)

# I/O file name
fname = dirname+'{}atom_{}fs_{}_{}nm.h5'.format(Natom,tpulse,Tmodel,round(dsamp*1e9,2))
# if args.r is not None:
# 	fname = dirname+'{}atom_{}fs_{}_{}nm_{}nm_sig.h5'.format(Natom,tpulse,Tmodel,round(dsamp*1e9,2),round(spotsig*1e9,2))

tic = time.time()

# generate source information and feed to the gpu
(xs,ys,zs,ts,phs,omgs,ks,taus,pols) = sframe(tpulse,Natom,dsamp,tlit,spotsig)

rsg = cuda.device_array(shape=(Natom,Npxl,Npxl),dtype='float64')		# holder for source positions
Fsg = cuda.device_array(shape=(Natom,Npxl,Npxl),dtype='complex128')	 # holder for field spatial component
tag = cuda.device_array(shape=(Natom,Npxl,Npxl),dtype='float64')		# holder for time of arrival

rg = cuda.to_device([r])			# source-detector distance
xsg = cuda.to_device(xs)			# source position
ysg = cuda.to_device(ys)			# """
zsg = cuda.to_device(zs)			# """
tsg = cuda.to_device(ts)			# emission time
phsg = cuda.to_device(phs)			# phase
omgsg = cuda.to_device(omgs)		# angular frequency
ksg = cuda.to_device(ks)			# wave vector
tausg = cuda.to_device(taus)		# lifetime
polxg = cuda.to_device(np.cos(pols))		# polarization x
polyg = cuda.to_device(np.sin(pols))		# polarization y
dtslistg = cuda.to_device(dtslist)  # lifetime for emission lines
#strans(xs,ys,zs,ts,phs,omgs,ks,taus,dtslist,xdet,ydet,r)

# calculate common time axis (since detector w/subpxl is huge, need to chop up to calculate full time axis)
taxis_widnow_size = 100
rsming = cuda.device_array(shape=(Natom, taxis_widnow_size,taxis_widnow_size),dtype='float64')
tasming = cuda.device_array(shape=(Natom, taxis_widnow_size,taxis_widnow_size),dtype='float64')
rsmaxg = cuda.device_array(shape=(Natom, taxis_widnow_size,taxis_widnow_size),dtype='float64')
tasmaxg = cuda.device_array(shape=(Natom, taxis_widnow_size,taxis_widnow_size),dtype='float64')

xxmin, yymin = np.indices((taxis_widnow_size,taxis_widnow_size))
xxmin = xxmin-xxmin.mean(); xxmin = xxmin*subpxlsize
yymin = yymin-yymin.mean(); yymin = yymin*subpxlsize

xxmax, yymax = np.indices((taxis_widnow_size,taxis_widnow_size))
xxmax = xxmax-xxmax.mean(); xxmax = xxmax*subpxlsize + detsize/2
yymax = yymax-yymax.mean(); yymax = yymax*subpxlsize + detsize/2

xxming = cuda.to_device(xxmin); yyming = cuda.to_device(yymin)
xxmaxg = cuda.to_device(xxmax); yymaxg = cuda.to_device(yymax)

getaxis[(16,16),(16,16)](rsming,tasming,xsg,ysg,zsg,tsg,xxming,yyming,rg[0])
getaxis[(16,16),(16,16)](rsmaxg,tasmaxg,xsg,ysg,zsg,tsg,xxmaxg,yymaxg,rg[0])

trange = np.max(tasmaxg)-np.min(tasming) + 16*dtslist.max()
mm = int(np.round(trange*1e16)+1)
tmm = np.arange(mm)/10
taxis = tmm*1e-15 + np.min(tasming)
taxisg = cuda.to_device(taxis)

# some temporal component for field calculations
T2g = cuda.device_array(shape=(mm,Natom),dtype='complex128')
temporal[(64,32),(32,32)](T2g,omgsg,taxisg)

# loop through sub-pixel detector windows and populate fields
Imgg = cuda.device_array(shape=(Npxl,Npxl),dtype='float64')
rswg = cuda.device_array(shape=(Natom,Nsubpxl_w,Nsubpxl_w),dtype='float64')
Fswg = cuda.device_array(shape=(Natom,Nsubpxl_w,Nsubpxl_w),dtype='complex128')
tawg = cuda.device_array(shape=(Natom,Nsubpxl_w,Nsubpxl_w),dtype='float64')

# calculate field in each time slice in each detector window
for iwin in tqdm(range(nwindow), desc='row'):
	for jwin in tqdm(range(nwindow), desc='column'):
		# get pixel positions for current window
		xlb = iwin*Nsubpxl_w; xub = (iwin+1)*Nsubpxl_w
		ylb = jwin*Nsubpxl_w; yub = (jwin+1)*Nsubpxl_w

		xdetw = xdetsub[xlb:xub,ylb:yub]
		ydetw = ydetsub[ylb:yub,ylb:yub]
		xdetwg = cuda.to_device(np.ascontiguousarray(xdetw))
		ydetwg = cuda.to_device(np.ascontiguousarray(ydetw))

		# (re)initialize speckle pattern for current window
		imgwg = cuda.device_array(shape=(Nsubpxl_w,Nsubpxl_w),dtype='float64')

		# spatial component
		spatial[(16,16),(32,32)](rswg,Fswg,tawg,xsg,ysg,zsg,ksg,tsg,phsg,xdetwg,ydetwg,rg[0])

		# field strength calculation in each slice
		Es1wg = cuda.device_array(shape=(mm,Nsubpxl_w, Nsubpxl_w,2),dtype='complex128')
		Es2wg = cuda.device_array(shape=(mm,Nsubpxl_w, Nsubpxl_w,2),dtype='complex128')
		Es3wg = cuda.device_array(shape=(mm,Nsubpxl_w, Nsubpxl_w,2),dtype='complex128')
		for i in range(mm):
			getslice[(32,32),(32,32)](Es1wg[i],Es2wg[i],Es3wg[i],
				Fsg[:,xlb:xub,ylb:yub],taxisg[i],T2g[i],
				tag[:,xlb:xub,ylb:yub],tausg,polxg,polyg,dtslistg)

		# summing up field for speckle pattern
		get_img[(32,32),(32,32)](Imgg[xlb:xub,ylb:yub],Es1wg,Es2wg,Es3wg)

img = Imgg.copy_to_host()
# recombine sub-pixels to get real pixel intensity (numerical integration)
sum1 = img.reshape(Npxl,nsubpxl,Nsubpxl).sum(axis=1)
sum2 = (sum1.T.reshape(Npxl,nsubpxl,Npxl).sum(axis=1)).T
contrast = (sub2.std()/sum2.mean())**2

toc = time.time()
print('	job',numjob,',{} ms,'.format(round((toc-tic)*1e3,2)),mm,'slices')

# I/O
with h5py.File(fname,'a') as f:
	grp = f.create_group('{}'.format(numjob))
	grp.create_dataset('source', data=np.stack([xs,ys,zs,tlit,ts,phs,omgs,ks,taus,pols]))
	grp.create_dataset('pattern', data=img)
	grp.create_dataset('pattern_sub', data=sum2)
	grp.create_dataset('contrast', data=[contrast])
	grp.create_dataset('others', data=[spotsig, nl_t, nl_s])
