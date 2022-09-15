# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 13:53:17 2022

@author: Jinuk
"""
import numpy as np
from numba import njit, prange, jit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import random
import os
from scipy import fftpack
from scipy.optimize import curve_fit #https://smlee729.github.io/python/simulation/2015/03/25/2-curve_fitting.html
from multiprocessing import Pool
import multiprocessing
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.rc('xtick', labelsize=15)
matplotlib.rc('ytick', labelsize=15)
## plot colors(defalut cycle) #https://matplotlib.org/3.1.0/gallery/color/color_cycle_default.html
prop_cycle = plt.rcParams['axes.prop_cycle']
colors     = prop_cycle.by_key()['color']
colors     = ['black','red','blue']
fmt_list   = ['o','v','^','<','>','o','v','^','<','>']
linestyle_list = ['-','--','-.',':','-','--','-.',':','-','--']

def normlrtz(x, amp1, cen1, wid1):#wid1: FWHM
    return amp1/np.pi*(wid1/2.0)/((x-cen1)**2.0+(wid1/2.0)**2.0)

# disable numba's just-in-time compilation
# os.environ["NUMBA_DISABLE_JIT"]= str(1)

ttt = time.time()
now = time.localtime()
print("%04d/%02d/%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))

### what to calculate
# whattodo = 'evolve'                      # time evolution calc  t vs n
# whattodo = 'outputpower'                 # output power
whattodo = 'g1calc'                      # g1 ftn calculation

### what parameters to change?  /  If do not want to change any parameters, input an array of length 1.
# whattovary = 'clusnum'                   # number of cluster in the cavity
whattovary = 'dca'                     # cavity-atom detuning
# whattovary = 'dpa'                     # pump  -atom detuning

numofproc = multiprocessing.cpu_count()-2  # num of cpu core
ntraj     = 1000                           # num of trajectories to simulate

cs      = 10000               # number of atoms in a cluster
clusnum = 500                 # number of cluster in the cavity

# 138Ba
# kappa   = 2*np.pi*230       # MHz cavity dissipation rate
# g       = 2*np.pi*0.22      # MHz coupling strength
# tau     = 0.14              # microsecond interaction time
# ctlth   = 1000              # number of g1 time element / g1 time range 0 ~ ctlth*stpsize*dt
# stpsize = 500               # unit time for g1 ftn : stpsize*dt

# 87Sr
# kappa   = 2*np.pi*240       # MHz cavity dissipation rate
# g       = 2*np.pi*0.078     # MHz coupling strength
# tau     = 0.13              # microsecond interaction time
# ctlth   = 1000              # number of g1 time element / g1 time range 0 ~ ctlth*stpsize*dt
# stpsize = 50000             # unit time for g1 ftn : stpsize*dt
# cstpsiz = 1                 # unit time for g1 ftn : cstpsiz*stpsize*dt

# 40Ca
# kappa   = 2*np.pi*250       # MHz cavity dissipation rate
# g       = 2*np.pi*0.0174    # MHz coupling strength
# tau     = 0.13              # microsecond interaction time
# ctlth   = 1000              # number of g1 time element / g1 time range 0 ~ ctlth*stpsize*dt
# stpsize = 50000             # unit time for g1 ftn : stpsize*dt

# 171Yb
# kappa   = 2*np.pi*0.202    # MHz cavity dissipation rate
# g       = 2*np.pi*48E-6    # MHz coupling strength
# tau     = 140              # microsecond interaction time
# ctlth   = 1000             # number of g1 time element / g1 time range 0 ~ ctlth*stpsize*dt
# stpsize = 100000           # unit time for g1 ftn : stpsize*dt
# cstpsiz = 1                # unit time for g1 ftn : cstpsiz*stpsize*dt


# test
cs      = 2                  # number of atoms in a cluster
clusnum = 1000               # number of cluster in the cavity
ctlth   = 10000              # number of g1 time element / g1 time range 0 ~ ctlth*cstpsiz*stpsize*dt
stpsize = 8                  # unit time for recording
cstpsiz = 1                  # unit time for g1 ftn : cstpsiz*stpsize*dt

kappa   = 2*np.pi*50        # MHz cavity dissipation rate
g       = 2*np.pi*0.5       # MHz coupling strength
tau     = 1.0               # microsecond interaction time
Gammac  = g**2.0/kappa      # Gamma_c

dt      = tau/200           # microsecond unit time
delDtau = 0.2*np.pi         # doppler width

delD    = delDtau/tau
delT    = 2*np.pi*00.0      # MHz doppler shift by tilting atomic beam
delca   = 2*np.pi*02.5      # MHz cavity-atom detuning
delpa   = 2*np.pi*01.2      # MHz pump-atom detuning

rhoee = 0.9                                 # cos theta/2 |g> + sin theta/2 |e> rho_ee = sin^2 theta/2
theta = 2.0*np.arcsin(np.sqrt(rhoee))       # cos theta/2 |g> + sin theta/2 |e> rho_ee = sin^2 theta/2

randomphase       = 'off' # 'on' : kz in [0,2pi) 'off':kz=0  z: position along the cavity axis
fftcalc           = 'on' # do fft
fftplot           = 'on' # plot fft results
fftfit            = 'on' # fit  fft results
pump_linewidth    = 2*np.pi*0.01           # fwhm of pumplaser MHz
sqrtFWHM          = np.sqrt(pump_linewidth)
accumulated_phase = 0.0                     # pump laser phase

if whattodo == 'evolve':
    t_final = 5*tau           # microsec, simulation time
    stpsize = 1
elif whattodo == 'outputpower':
    t_final = 500*tau         # microsec, simulation time
    stpsize = 1
else:
    t_final = 10000            # microsec, simulation time

t_length = int(t_final/dt)
t_list   = np.linspace(0,t_final,t_length)

clusnum_list = np.linspace(5,clusnum,20)
clusnum_list = clusnum_list.astype(int)
# clusnum_list = np.array([clusnum])
dca_list   = np.linspace(-2*delca,0.0,50)

dpa_list   = np.linspace(-delpa,delpa,50)

if whattovary == 'clusnum':
    vlist = cs*clusnum_list
elif whattovary == 'dca':
    vlist = dca_list
elif whattovary == 'dpa':
    vlist = dpa_list


## single trajectory, time evolution, numerically solve stochastic differential equation
@njit(nogil=True)#,parallel=True)
def sngltraj(kappa,g,delca,delpa,tau,clusnum,accumulated_phase,sqrtFWHM):
    Gammac = g**2.0*kappa/4.0/(kappa**2.0/4.0+delca**2.0)
    Gamma0 = g**2.0/kappa
    GammaD = g**2.0*delca/2.0/(kappa**2.0/4.0+delca**2.0)
    sx  = np.full(clusnum,0.0)
    sy  = np.full(clusnum,0.0)
    sz  = np.full(clusnum,0.0)
    eta = np.full(clusnum,0.0)
    z   = np.full(clusnum,0.0) # initial injection location
    vz  = np.full(clusnum,0.0) # velociy along the cavity axis

    jx = 0.0
    jy = 0.0

    output  = np.full(t_length//stpsize,0.0)
    jxlist  = np.full(t_length//stpsize,0.0)
    jylist  = np.full(t_length//stpsize,0.0)

    nn         = 0
    inoutindex = 0

    for i in range(t_length):
        tt = i*dt
        accumulated_phase += sqrtFWHM*np.random.normal(loc=0.0,scale=np.sqrt(dt)) # simulate phase noise
        jx = np.sum(eta*sx)
        jy = np.sum(eta*sy)

        if i%stpsize == 0:
            output[i//stpsize] = (np.abs(jx)**2.0+np.abs(jy)**2.0)
            jxlist[i//stpsize] = jx
            jylist[i//stpsize] = jy

        ### stochstic variables
        xip = np.random.normal(0,np.sqrt(dt))
        xiq = np.random.normal(0,np.sqrt(dt))
        # xip = np.random.normal(0,np.sqrt(dt),clusnum)
        # xiq = np.random.normal(0,np.sqrt(dt),clusnum)
        # xip = np.full(clusnum,np.random.normal(0,np.sqrt(dt)))
        # xiq = np.full(clusnum,np.random.normal(0,np.sqrt(dt)))

        ########## simple first order evolution #####################
        # dsx =  dt*Gammac/2*eta*(jx*sz-eta*sx*(sz+1)) - dt*GammaD/2*eta*(jy*sz-eta*sy*(sz+1)) - Gammac/np.sqrt(Gamma0)*eta*sz*xip - GammaD/np.sqrt(Gamma0)*eta*sz*xiq
        # dsy =  dt*Gammac/2*eta*(jy*sz-eta*sy*(sz+1)) + dt*GammaD/2*eta*(jx*sz-eta*sx*(sz+1)) + Gammac/np.sqrt(Gamma0)*eta*sz*xiq - GammaD/np.sqrt(Gamma0)*eta*sz*xip
        # dsz = -dt*Gammac/2*eta*(jx*sx+jy*sy-eta*(sx**2.0+sy**2.0)) + dt*GammaD/2*eta*(jy*sx-jx*sy) - dt*Gammac*eta**2.0*(sz+1) + Gammac/np.sqrt(Gamma0)*eta*(sx*xip-sy*xiq) + GammaD/np.sqrt(Gamma0)*eta*(sx*xiq+sy*xip)
        # sx += dsx
        # sy += dsy
        # sz += dsz
        # z   += dt*vz
        # eta = np.cos(z)
        #############################################################
        ########## Runge-Kutta 4th order #####################
        # Thomas Gard, Introduction to Stochastic Differential Equations p.201
        F0x  = + Gammac/2*eta*(jx*sz-eta*sx*(sz+1)) - GammaD/2*eta*(jy*sz-eta*sy*(sz+1))
        G0xp = - Gammac/np.sqrt(Gamma0)*eta*sz
        G0xq = - GammaD/np.sqrt(Gamma0)*eta*sz
        F0y  = + Gammac/2*eta*(jy*sz-eta*sy*(sz+1)) + GammaD/2*eta*(jx*sz-eta*sx*(sz+1))
        G0yp = - GammaD/np.sqrt(Gamma0)*eta*sz
        G0yq = + Gammac/np.sqrt(Gamma0)*eta*sz
        F0z  = - Gammac/2*eta*(jx*sx+jy*sy-eta*(sx**2.0+sy**2.0)) + GammaD/2*eta*(jy*sx-jx*sy) - Gammac*eta**2.0*(sz+1)
        G0zp = + Gammac/np.sqrt(Gamma0)*eta*sx + GammaD/np.sqrt(Gamma0)*eta*sy
        G0zq = - Gammac/np.sqrt(Gamma0)*eta*sy + GammaD/np.sqrt(Gamma0)*eta*sx

        sx1  = sx + 0.5*dt*F0x + 0.5*(G0xp*xip + G0xq*xiq)
        sy1  = sy + 0.5*dt*F0y + 0.5*(G0yp*xip + G0yp*xiq)
        sz1  = sz + 0.5*dt*F0z + 0.5*(G0zp*xip + G0zq*xiq)

        eta1 = np.cos(z + 0.5*dt*vz)
        jx1  = np.sum(eta1*sx1)
        jy1  = np.sum(eta1*sy1)

        F1x  = + Gammac/2*eta1*(jx1*sz1-eta1*sx1*(sz1+1)) - GammaD/2*eta1*(jy1*sz1-eta1*sy1*(sz1+1))
        G1xp = - Gammac/np.sqrt(Gamma0)*eta1*sz1
        G1xq = - GammaD/np.sqrt(Gamma0)*eta1*sz1
        F1y  = + Gammac/2*eta1*(jy1*sz1-eta1*sy1*(sz1+1)) + GammaD/2*eta1*(jx1*sz1-eta1*sx1*(sz1+1))
        G1yp = - GammaD/np.sqrt(Gamma0)*eta1*sz1
        G1yq = + Gammac/np.sqrt(Gamma0)*eta1*sz1
        F1z  = - Gammac/2*eta1*(jx1*sx1+jy1*sy1-eta1*(sx1**2.0+sy1**2.0)) + GammaD/2*eta1*(jy1*sx1-jx1*sy1) - Gammac*eta1**2.0*(sz1+1)
        G1zp = + Gammac/np.sqrt(Gamma0)*eta1*sx1 + GammaD/np.sqrt(Gamma0)*eta1*sy1
        G1zq = - Gammac/np.sqrt(Gamma0)*eta1*sy1 + GammaD/np.sqrt(Gamma0)*eta1*sx1

        sx2  = sx + 0.5*dt*F1x + 0.5*(G1xp*xip + G1xq*xiq)
        sy2  = sy + 0.5*dt*F1y + 0.5*(G1yp*xip + G1yp*xiq)
        sz2  = sz + 0.5*dt*F1z + 0.5*(G1zp*xip + G1zq*xiq)

        eta2 = eta1
        jx2  = np.sum(eta2*sx2)
        jy2  = np.sum(eta2*sy2)

        F2x  = + Gammac/2*eta2*(jx2*sz2-eta2*sx2*(sz2+1)) - GammaD/2*eta2*(jy2*sz2-eta2*sy2*(sz2+1))
        G2xp = - Gammac/np.sqrt(Gamma0)*eta2*sz2
        G2xq = - GammaD/np.sqrt(Gamma0)*eta2*sz2
        F2y  = + Gammac/2*eta2*(jy2*sz2-eta2*sy2*(sz2+1)) + GammaD/2*eta2*(jx2*sz2-eta2*sx2*(sz2+1))
        G2yp = - GammaD/np.sqrt(Gamma0)*eta2*sz2
        G2yq = + Gammac/np.sqrt(Gamma0)*eta2*sz2
        F2z  = - Gammac/2*eta2*(jx2*sx2+jy2*sy2-eta2*(sx2**2.0+sy2**2.0)) + GammaD/2*eta2*(jy2*sx2-jx2*sy2) - Gammac*eta2**2.0*(sz2+1)
        G2zp = + Gammac/np.sqrt(Gamma0)*eta2*sx2 + GammaD/np.sqrt(Gamma0)*eta2*sy2
        G2zq = - Gammac/np.sqrt(Gamma0)*eta2*sy2 + GammaD/np.sqrt(Gamma0)*eta2*sx2

        sx3  = sx + dt*F2x + G2xp*xip + G2xq*xiq
        sy3  = sy + dt*F2y + G2yp*xip + G2yp*xiq
        sz3  = sz + dt*F2z + G2zp*xip + G2zq*xiq

        eta3 = np.cos(z + dt*vz)
        jx3  = np.sum(eta3*sx3)
        jy3  = np.sum(eta3*sy3)

        F3x  = + Gammac/2*eta3*(jx3*sz3-eta3*sx3*(sz3+1)) - GammaD/2*eta3*(jy3*sz3-eta3*sy3*(sz3+1))
        G3xp = - Gammac/np.sqrt(Gamma0)*eta3*sz3
        G3xq = - GammaD/np.sqrt(Gamma0)*eta3*sz3
        F3y  = + Gammac/2*eta3*(jy3*sz3-eta3*sy3*(sz3+1)) + GammaD/2*eta3*(jx3*sz3-eta3*sx3*(sz3+1))
        G3yp = - GammaD/np.sqrt(Gamma0)*eta3*sz3
        G3yq = + Gammac/np.sqrt(Gamma0)*eta3*sz3
        F3z  = - Gammac/2*eta3*(jx3*sx3+jy3*sy3-eta3*(sx3**2.0+sy3**2.0)) + GammaD/2*eta3*(jy3*sx3-jx3*sy3) - Gammac*eta3**2.0*(sz3+1)
        G3zp = + Gammac/np.sqrt(Gamma0)*eta3*sx3 + GammaD/np.sqrt(Gamma0)*eta3*sy3
        G3zq = - Gammac/np.sqrt(Gamma0)*eta3*sy3 + GammaD/np.sqrt(Gamma0)*eta3*sx3


        sx += (F0x+2*F1x+2*F2x+F3x)*dt/6 + (G0xp+2*G1xp+2*G2xp+G3xp)*xip/6 + (G0xq+2*G1xq+2*G2xq+G3xq)*xiq/6
        sy += (F0y+2*F1y+2*F2y+F3y)*dt/6 + (G0yp+2*G1yp+2*G2yp+G3yp)*xip/6 + (G0yq+2*G1yq+2*G2yq+G3yq)*xiq/6
        sz += (F0z+2*F1z+2*F2z+F3z)*dt/6 + (G0zp+2*G1zp+2*G2zp+G3zp)*xip/6 + (G0zq+2*G1zq+2*G2zq+G3zq)*xiq/6
        z  += dt*vz
        eta = np.cos(z)
        #############################################################

        while tt - nn*tau/clusnum > 0: # atom in/out
            ## single atom injection
            # sx[inoutindex] = 1.0 if random.random() < (1 + np.sin(theta)*np.cos(-delpa*tt + accumulated_phase))/2.0 else -1.0 # projection noise
            # sy[inoutindex] = 1.0 if random.random() < (1 - np.sin(theta)*np.sin(-delpa*tt + accumulated_phase))/2.0 else -1.0
            # sz[inoutindex] = 1.0 if random.random() < rhoee else -1.0
            ## cluster injection
            sx[inoutindex] = 2.0*np.sum(np.random.random(cs)< (1 + np.sin(theta)*np.cos(-delpa*tt + accumulated_phase))/2.0)-cs
            sy[inoutindex] = 2.0*np.sum(np.random.random(cs)< (1 - np.sin(theta)*np.sin(-delpa*tt + accumulated_phase))/2.0)-cs
            sz[inoutindex] = 2.0*np.sum(np.random.random(cs)< rhoee)-cs
            ## when atom num in a cluster is large
            # p1 = (1 + np.sin(theta)*np.cos(-delpa*tt + accumulated_phase))/2.0  ## prob that sx=1 
            # mu = (2*p1-1)
            # sigma = np.sqrt(cs**2 + 2*mu*cs**2 + mu**2*cs**2 + 4*cs*p1 - 4*cs**2*p1 - 4*mu*cs**2*p1 - 4*cs*p1**2 +  4*cs**2*p1**2)
            # sx[inoutindex] = np.random.normal(cs*mu,sigma)
            # p1 = (1 - np.sin(theta)*np.sin(-delpa*tt + accumulated_phase))/2.0  ## prob that sy=1
            # mu = (2*p1-1)
            # sigma = np.sqrt(cs**2 + 2*mu*cs**2 + mu**2*cs**2 + 4*cs*p1 - 4*cs**2*p1 - 4*mu*cs**2*p1 - 4*cs*p1**2 +  4*cs**2*p1**2)
            # sy[inoutindex] = np.random.normal(cs*mu,sigma)
            # p1 = rhoee                                                                 ## prob that sz=1
            # mu = (2*p1-1)
            # sigma = np.sqrt(cs**2 + 2*mu*cs**2 + mu**2*cs**2 + 4*cs*p1 - 4*cs**2*p1 - 4*mu*cs**2*p1 - 4*cs*p1**2 +  4*cs**2*p1**2)
            # sz[inoutindex] = np.random.normal(cs*mu,sigma)

            if randomphase == 'on':
                z[inoutindex]   = random.random()*2.0*np.pi          # injection location (random)
            else:
                z[inoutindex]   = 0.0                                # injection location (anti-node only)
            vz[inoutindex]  = np.random.normal(delT,delD)            # velocity along the cavity axis rad/us
            nn += 1
            inoutindex = nn%clusnum

    output = Gammac*tau*output/4/clusnum/cs # photon per atom
    return output,jxlist,jylist

if whattodo =='evolve':
    print("----------------check list-------------")
    print("kappa tau >> 1")
    print(kappa*tau)
    print("kappa /g / sqrt(N) >> 1")
    print(kappa/g/np.sqrt(cs*clusnum))
    print("kappa /deltaD >> 1")
    print(kappa/delD)
    print("---------------------------------------")

    print("NtauGammac")
    print(cs*clusnum*tau*Gammac)
    print("unittime")
    print(dt)
    ################# parallell by pool  #################
    pool = multiprocessing.Pool(processes=numofproc)
    result = pool.starmap(sngltraj,[(kappa,g,delca,delpa,tau,clusnum,accumulated_phase,sqrtFWHM) for i in range(ntraj)]) #starmap  : for multiple argumetns
    pool.close()
    pool.join()
    output = []
    for i in range(ntraj):
        output.append(result[i][0])
    # output,jxlist,jylist = sngltraj(kappa,g,delca,delpa,tau,clusnum,accumulated_phase,sqrtFWHM)
    output = sum(np.array(output)) / ntraj
    print(np.average(output[len(output)//2:]))
    ######################################################

    ### plot and sav the results
    fig = plt.figure()
    plt.plot(t_list,output,'.')
    fig.savefig('evolve_'+str(rhoee)+'_atomn_'+str(clusnum)+'_delT_'+str(delT/2.0/np.pi)+'_deltDtau_'+str(delDtau/np.pi)+'_dca_'+str(delca/np.pi/2)+'_dpa_'+str(delpa/np.pi/2) + '_rp_'+ randomphase +'_plwth_'+str(pump_linewidth/np.pi/2)+'_.png')
    f=open('evolve'+str(rhoee)+'_atomn_'+str(clusnum)+'_delT_'+str(delT/2.0/np.pi)+'_deltDtau_'+str(delDtau/np.pi)+'_dca_'+str(delca/np.pi/2)+'_dpa_'+str(delpa/np.pi/2) + '_rp_'+ randomphase +'_plwth_'+str(pump_linewidth/np.pi/2)+'.txt','w')
    np.savetxt(f, np.vstack((t_list,output)).T)#,newline=' ')
    f.close()

if whattodo =='outputpower':
    print("----------------check list-------------")
    print("kappa tau >> 1")
    print(kappa*tau)
    print("kappa /g / sqrt(N) >> 1")
    print(kappa/g/np.sqrt(cs*clusnum))
    print("kappa /deltaD >> 1")
    print(kappa/delD)
    print("---------------------------------------")

    print("NtauGammac")
    print(cs*clusnum*tau*Gammac)
    print("unittime")
    print(dt)
    ################# parallell by pool  #################
    pool = multiprocessing.Pool(processes=numofproc)
    if whattovary=='clusnum':
        result = pool.starmap(sngltraj,[(kappa,g,delca,delpa,tau,clusnum,accumulated_phase,sqrtFWHM) for clusnum in clusnum_list]) #starmap  : for multiple argumetns
    elif whattovary=='dca':
        result = pool.starmap(sngltraj,[(kappa,g,delca,delpa,tau,clusnum,accumulated_phase,sqrtFWHM) for delca in dca_list]) 
    elif whattovary=='dpa':
        result = pool.starmap(sngltraj,[(kappa,g,delca,delpa,tau,clusnum,accumulated_phase,sqrtFWHM) for delpa in dpa_list])
    pool.close()
    pool.join()
    output = []
    for i in range(len(vlist)):
        output.append(result[i][0])
    output = np.array(output)
    p_avg = []
    for i in range(len(vlist)):
        p_avg.append(np.average(output[i][len(output):]))    
    ######################################################

    fig = plt.figure()
    if whattovary=='clusnum':
        vlist = vlist*Gammac*tau
    else:
        vlist = vlist/2/np.pi
    plt.plot(vlist,p_avg,'.')
    plt.ylim([0,1])
    fig.savefig('output_rhoee_'+str(rhoee)+'_atomn_'+str(clusnum)+'_delT_'+str(delT/2.0/np.pi)+'_deltDtau_'+str(delDtau/np.pi)+'_dca_'+str(delca/np.pi/2)+'_dpa_'+str(delpa/np.pi/2) + '_rp_'+ randomphase +'.png')
    f=open('output_rhoee_'+str(rhoee)+'_atomn_'+str(clusnum)+'_delT_'+str(delT/2.0/np.pi)+'_deltDtau_'+str(delDtau/np.pi)+'_dca_'+str(delca/np.pi/2)+'_dpa_'+str(delpa/np.pi/2) + '_rp_'+ randomphase +'.txt','w')
    np.savetxt(f, np.vstack((vlist,p_avg)).T)#,newline=' ') #*Gammac*tau
    f.close()

if whattodo =='g1calc':
    if ctlth*stpsize*dt>0.95*t_final:
        print("simulation time should be longer!------------------------------------------------------------------------------------")
    print("----------------check list-------------")
    print("kappa tau >> 1")
    print(kappa*tau)
    print("kappa /g / sqrt(N) >> 1")
    print(kappa/g/np.sqrt(cs*clusnum))
    print("kappa /deltaD >> 1")
    print(kappa/delD)
    print("---------------------------------------")

    print("NtauGammac")
    print(cs*clusnum*tau*Gammac)
    print("unittime")
    print(dt)

    ################# parallell by pool  #################
    pool = multiprocessing.Pool(processes=numofproc)
    if whattovary=='clusnum':
        result = pool.starmap(sngltraj,[(kappa,g,delca,delpa,tau,clusnum,accumulated_phase,sqrtFWHM) for clusnum in clusnum_list]) #starmap  : for multiple argumetns
    elif whattovary=='dca':
        result = pool.starmap(sngltraj,[(kappa,g,delca,delpa,tau,clusnum,accumulated_phase,sqrtFWHM) for delca in dca_list]) 
    elif whattovary=='dpa':
        result = pool.starmap(sngltraj,[(kappa,g,delca,delpa,tau,clusnum,accumulated_phase,sqrtFWHM) for delpa in dpa_list])
    pool.close()
    pool.join()
    jxx = []
    jyy = []
    for i in range(len(vlist)):
        jxx.append(result[i][1])
        jyy.append(result[i][2])
    jxx = np.array(jxx)
    jyy = np.array(jyy)

    now = time.localtime()
    print('g1calc start')
    print("%04d/%02d/%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))
    fig_g1c = plt.figure()
    fig_fft = plt.figure()
    fig_lnw = plt.figure()
    ax1     = fig_g1c.subplots()
    ax2     = fig_fft.subplots()
    ax3     = fig_lnw.subplots()
    vp_list    = [] # xaxis
    amp_list   = []
    cen_list   = [] # lasing frequency
    sigma_list = [] # FWHM
    
    # Create a new directory if it does not exist 
    path = os.getcwd()+'//g1sav'
    isExist = os.path.exists(path)
    if not isExist:
      os.makedirs(path)
    for i in prange(len(vlist)):
        if whattovary == 'clusnum':
            vary   = vlist[i]*Gammac*tau
        else:
            vary   = vlist[i]/2/np.pi
        # jxlist = jxx[i][len(jxx[0])//10:] # steadystate only
        # jylist = jyy[i][len(jyy[0])//10:] # steadystate only
        jxlist = jxx[i][1:] # steadystate only
        jylist = jyy[i][1:] # steadystate only
        ########### g1 calculation #############
        cftn = []                         # auto correlation fuction
        jxplist = jxlist[:-cstpsiz*ctlth]
        jyplist = jylist[:-cstpsiz*ctlth]
        for index in range(ctlth):
            jxpplist = jxlist[cstpsiz*index:cstpsiz*index-cstpsiz*ctlth]
            jypplist = jylist[cstpsiz*index:cstpsiz*index-cstpsiz*ctlth]
            # cftn.append((np.dot(jxplist,jxpplist))/len(jxplist)+(np.dot(jyplist,jypplist))/len(jyplist))
            cftn.append((np.dot(jxpplist+1.0j*jypplist,jxplist-1.0j*jyplist))/len(jxplist)/4.0)

        cftn  = np.array(cftn)
        # cftn  = cftn/max(cftn)                          # normalization g^(1)(0)=1
        ctime = cstpsiz*stpsize*dt*np.array(range(ctlth)) # x axis for g1 ftn
        cftn  *= np.exp(1.0j*delT*ctime)                  # rotating frame

        ax1.plot(ctime,cftn,'.')
        ax1.plot(ctime,cftn)

        ###### save the result
        fig_g1c.savefig(   'g1sav_'+str(rhoee)+'_atomn_'+str(clusnum)+'_delT_'+str(delT/2.0/np.pi)+'_deltDtau_'+str(delDtau/np.pi)+'_dca_'+str(delca/np.pi/2)+'_dpa_'+str(delpa/np.pi/2) + '_rp_'+ randomphase +'_plwth_'+str(pump_linewidth/np.pi/2)+'.png')
        f=open('g1sav/g1sav_rhoee_'+str(rhoee)+'_atomn_'+str(clusnum)+'_delT_'+str(delT/2.0/np.pi)+'_deltDtau_'+str(delDtau/np.pi)+'_dca_'+str(delca/np.pi/2)+'_dpa_'+str(delpa/np.pi/2) + '_rp_'+ randomphase +'_plwth_'+str(pump_linewidth/np.pi/2)+'_v_'+str(vary)+'.txt','w')
        np.savetxt(f, np.vstack((np.real(ctime),np.real(cftn),np.imag(cftn))).T)
        f.close()

        if fftcalc == 'on':
            x = ctime
            y = cftn
            x = np.concatenate((-x[::-1][:-1],x))
            y = np.concatenate((np.conjugate(y[::-1][:-1]),y))
            g_x = x
            g_y = y
            yf = fftpack.fft(y, x.size)
            amp = np.abs(yf) # get amplitude spectrum 
            freq = fftpack.fftfreq(x.size, (x[1]-x[0])) # MHz
            ind =freq.argsort()
            amp = amp[ind]
            freq = freq[ind]
            f=open('g1sav/fftsav_rhoee_'+str(rhoee)+'_atomn_'+str(clusnum)+'_delT_'+str(delT/2.0/np.pi)+'_deltDtau_'+str(delDtau/np.pi)+'_dca_'+str(delca/np.pi/2)+'_dpa_'+str(delpa/np.pi/2) + '_rp_'+ randomphase +'_plwth_'+str(pump_linewidth/np.pi/2)+'_v_'+str(vary)+'.txt','w')
            np.savetxt(f, np.vstack((np.real(freq), (1/amp.size)*np.real(np.abs(amp)/(freq[1]-freq[0])))).T)
            f.close()
            y = (1/amp.size)*amp/(freq[1]-freq[0])
            if fftplot == 'on':
                ax2.plot(freq, (1/amp.size)*amp/(freq[1]-freq[0]),'k.',label=vlist[i])
                ax2.plot(freq, (1/amp.size)*amp/(freq[1]-freq[0]),'k-',alpha=0.8)
                # ax2.set_xlim(-3.0,3.0)
               # plt.yscale('log')
               # plt.ylim(bottom=0.0)
                # legend = plt.legend(loc='upper right')
            if fftfit == 'on':
                try:
                    index = np.argmax(np.abs(y))
                    fitrange = 30
                    yfit =     np.abs(y[index-fitrange:index+fitrange])
                    xfit = np.real(freq[index-fitrange:index+fitrange])
                    initial_guess = [max(yfit),np.real(freq[index]),(max(xfit)-min(xfit))/2]    # p0=[amp1, cen1, sigma1]
                    popt, pcov = curve_fit(normlrtz, xfit,yfit, p0=initial_guess)
                    # print("FWHM_lasing")
                    # print(popt[-1])
                    if whattovary == 'clusnum':
                        vp_list.append(vlist[i]*Gammac*tau)
                    else:
                        vp_list.append(vlist[i]/2/np.pi)

                    amp_list.append(popt[0])
                    cen_list.append(popt[1])
                    sigma_list.append(popt[2])
                    if fftplot == 'on':
                        ax2.plot(xfit, normlrtz(xfit, *popt), color='red', linewidth=2, label="fitting")
                        ax2.set_xlim(xfit[0],xfit[-1])
                        # legend = plt.legend(loc='upper right')
                except:
                    print("fftfitting error occured!")
    if fftplot == 'on':
        fig_fft.savefig('fftsav_'   +str(rhoee)+'_atomn_'+str(clusnum)+'_delT_'+str(delT/2.0/np.pi)+'_deltDtau_'+str(delDtau/np.pi)+'_dca_'+str(delca/np.pi/2)+'_dpa_'+str(delpa/np.pi/2) + '_rp_'+ randomphase +'_plwth_'+str(pump_linewidth/np.pi/2)+'.png')
    if fftfit  == 'on':
        ax3.plot(vp_list,sigma_list)
        ax3.set_yscale('log')
        fig_lnw.savefig('fftresult_'+str(rhoee)+'_atomn_'+str(clusnum)+'_delT_'+str(delT/2.0/np.pi)+'_deltDtau_'+str(delDtau/np.pi)+'_dca_'+str(delca/np.pi/2)+'_dpa_'+str(delpa/np.pi/2) + '_rp_'+ randomphase +'_plwth_'+str(pump_linewidth/np.pi/2)+'.png')
        f=open('fftresult_rhoee_'   +str(rhoee)+'_atomn_'+str(clusnum)+'_delT_'+str(delT/2.0/np.pi)+'_deltDtau_'+str(delDtau/np.pi)+'_dca_'+str(delca/np.pi/2)+'_dpa_'+str(delpa/np.pi/2) + '_rp_'+ randomphase +'_plwth_'+str(pump_linewidth/np.pi/2)+'.txt','w')
        np.savetxt(f, np.vstack((np.real(vp_list),amp_list,cen_list,sigma_list)).T)
        f.close()

print("Finished in %.06f min" % ((time.time()-ttt)/60.0))
