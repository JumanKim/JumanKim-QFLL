import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib

matplotlib.rc('xtick', labelsize=15)
matplotlib.rc('ytick', labelsize=15)

def load(fname):
    ''' load the file using std open'''
    f = open(fname,'r')

    data = []
    for line in f.readlines():
        data.append(line.replace('\n','').split())

    f.close()

    return data

def exp_decay(t, sigma,period):
    return np.exp(-t*sigma)*np.cos(2.0*np.pi*t/period)

def normlrtz(x, amp1, cen1, wid1):#wid1: FWHM
    return amp1/np.pi*(wid1/2.0)/((x-cen1)**2.0+(wid1/2.0)**2.0)

path = os.getcwd() + "/g1sav1.0//"
files = [ file for file in os.listdir(path) if file.startswith( ("fftsav") ) ]


xlist=[]

flist = []

for file in files:
    if file.split('rp_')[-1].split('_')[0]=='on':
        flist.append(file)
files=flist

for file in files:
    #print(file)
    xlist.append(float(file.split('.txt')[0].split('_')[-1]))
    #print(file)
    
files = np.array(files)[np.argsort(xlist)]
print(path)
print(len(files))
#for file in files:
#    print(file)
Zmesh = []
for file in files:
    xaxis  = file.split('_')[-2]
    data   = load(path +'/'+ file)
    data   = np.asarray(data).astype(complex)
    x = data[:,0] ##us
    y = data[:,1]
    xaxis = str(float(file.split('.txt')[-2].split('_')[-1]))
    Zmesh.append(y)

Xmesh, Ymesh = np.meshgrid(np.linspace(min(xlist), max(xlist), len(files)),np.real(x))
Zmesh = np.array(Zmesh).T
Zmesh = np.log(Zmesh)
fig = plt.figure()
# cp = plt.contourf(Xmesh, Ymesh, Zmesh, levels = np.linspace(Zmesh.reshape(-1, 1).min(), Zmesh.reshape(-1, 1).max(), 50),cmap='viridis')
cp = plt.contourf(Xmesh, Ymesh, Zmesh, levels = np.linspace(Zmesh.reshape(-1, 1).min(), Zmesh.reshape(-1, 1).max(), 50),cmap='RdBu_r')
plt.xlim([-5,0])
plt.ylim([-0.20,0.15])
plt.colorbar(cp)
plt.xlabel('$\Delta_{\mathrm{ca}}/2\pi$',fontsize=15,math_fontfamily='cm')
plt.ylabel('Frequency w.r.t atomic resonance (MHz)',fontsize=15)
fig.tight_layout()
fig.savefig('contour.png',dpi=300)
    