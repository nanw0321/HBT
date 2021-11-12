from Functions_gpu import *
import matplotlib.pyplot as plt
plt.ion()

tic = time.time()

# simulation variables
tpulse = 100
Natom = 1000
dsamp = 10 * 1e-9       # sample thickness

# generate source information and feed to the gpu
# tlit, _, _ = SASE(tpulse,Natom)
tlit, _, _ = Gaus(tpulse,Natom)
(xs,ys,zs,ts,phs,omgs,ks,taus,pols) = sframe(tpulse,Natom,dsamp,tlit)

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
print('trange:', round(trange*1e15,2),'fs')
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
print((toc-tic)*1e3,'ms',mm,'slices')
plt.figure()
plt.imshow(img/img.mean(),cmap='jet',interpolation='nearest')
plt.title(contrast)
plt.clim([0,3])
plt.savefig('speckle_{}.svg'.format(round(contrast,3)), format='svg',dpi=300,transparent=True)
