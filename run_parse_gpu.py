from Functions_gpu import *
import time, argparse

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

if Tmodel == 'Delta':       tlit, I_ratio, nl_t = delta(tpulse, Natom); tpulse = 0.
if Tmodel == 'Square':		tlit, I_ratio, nl_t = square(tpulse, Natom)
if Tmodel == 'Square_mono': tlit, I_ratio, nl_t = squaremono(tpulse,Natom)
if Tmodel == 'Gaus':		tlit, I_ratio, nl_t = Gaus(tpulse,Natom)
if Tmodel == 'Gaus_mono':	tlit, I_ratio, nl_t = Gausmono(tpulse,Natom)
if Tmodel == 'SASE':		tlit, I_ratio, nl_t = SASE(tpulse,Natom)
if Tmodel == 'SASE_mono':	tlit, I_ratio, nl_t = SASEmono(tpulse,Natom)

# I/O file name
fname = dirname+'{}atom_{}_{}nm.h5'.format(Natom,Tmodel,round(dsamp*1e9,2))
# if args.r is not None:
# 	fname = dirname+'{}atom_{}fs_{}_{}nm_{}nm_sig.h5'.format(Natom,tpulse,Tmodel,round(dsamp*1e9,2),round(spotsig*1e9,2))

tic = time.time()

# generate source information and feed to the gpu
(xs,ys,zs,ts,phs,omgs,ks,taus,pols) = sframe(tpulse,Natom,dsamp,tlit,spotsig)

rsg = cuda.device_array(shape=(Natom,Npxl,Npxl),dtype='float64')        # holder for source positions
Fsg = cuda.device_array(shape=(Natom,Npxl,Npxl),dtype='complex128')     # holder for field spatial component
tag = cuda.device_array(shape=(Natom,Npxl,Npxl),dtype='float64')        # holder for time of arrival

xsg = cuda.to_device(xs)            # source position
ysg = cuda.to_device(ys)            # """
zsg = cuda.to_device(zs)            # """
tsg = cuda.to_device(ts)            # emission time
phsg = cuda.to_device(phs)          # phase
omgsg = cuda.to_device(omgs)        # angular frequency
ksg = cuda.to_device(ks)            # wave vector
tausg = cuda.to_device(taus)        # lifetime
polxg = cuda.to_device(np.cos(pols))        # polarization x
polyg = cuda.to_device(np.sin(pols))        # polarization y
dtslistg = cuda.to_device(dtslist)  # lifetime for emission lines
#strans(xs,ys,zs,ts,phs,omgs,ks,taus,dtslist,xdet,ydet,r)

# calculate spatial component
xdetg = cuda.to_device(xdet)    # detector pixel position
ydetg = cuda.to_device(ydet)    # """
rg = cuda.to_device([r])        # source-detector distance
spatial[(32,32),(32,32)](rsg,Fsg,tag, xsg,ysg,zsg,ksg,tsg,phsg, xdetg,ydetg,rg[0])

# calculate temporal component
ta = tag.copy_to_host()							# grab calculated time of arrival from gpu
trange = ta.max()-ta.min() + 16*dtslist.max()	# trange = 16 lifetimes after arrival of the last wavefront

# trange = np.max([30e-15, trange])                 # manually compensate for the very very short pulses
mm = int(np.round(trange*1e16)+1)				# number of 100 atto-second slices
tmm = np.arange(mm)/10							# time axis (fs, not shifted)
taxis = tmm*1e-15 + np.min(rsg)/c 				# shifted time axis used for calculation
taxisg = cuda.to_device(taxis)

T2g = cuda.device_array(shape=(mm,Natom),dtype='complex128')
temporal[(64,32),(32,32)](T2g,omgsg,taxisg)

# calculate field in each time slice
Es1g = cuda.device_array(shape=(mm,Npxl,Npxl,2),dtype='complex128')
Es2g = cuda.device_array(shape=(mm,Npxl,Npxl,2),dtype='complex128')
Es3g = cuda.device_array(shape=(mm,Npxl,Npxl,2),dtype='complex128')
for i in range(mm):
    getslice[(32,32),(32,32)](Es1g[i],Es2g[i],Es3g[i],Fsg,taxisg[i],T2g[i],tag,tausg,polxg,polyg,dtslistg)

# summing up field for speckle pattern
Imgg = cuda.device_array(shape=(Npxl,Npxl),dtype='float64')

get_img[(32,32),(32,32)](Imgg,Es1g,Es2g,Es3g)
img = Imgg.copy_to_host()
contrast = (img.std()/img.mean())**2

toc = time.time()
print('    job',numjob,',{} ms,'.format(round((toc-tic)*1e3,2)),mm,'slices')

# I/O
with h5py.File(fname,'a') as f:
	grp = f.create_group('{}'.format(numjob))
	grp.create_dataset('source', data=np.stack([xs,ys,zs,tlit,ts,phs,omgs,ks,taus,pols]))
	grp.create_dataset('pattern', data=img)
	grp.create_dataset('contrast', data=[contrast])
	grp.create_dataset('others', data=[spotsig, nl_t, nl_s, tpulse])
