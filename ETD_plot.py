# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 17:06:03 2019

@author: user
"""
import numpy as np
from matplotlib import pyplot as pl
from scipy.interpolate import griddata
from tqdm import tqdm
import os
focdist=0.3
cut_angle=np.pi/2.
data=[]
for m in tqdm(np.arange(40)):
    datatemp=open(os.getcwd()+"\\ETD_Temp\\focdist=%.2f,cut_angle=%.1f" % (focdist,cut_angle*180/np.pi)+",ETD_%d.txt" % m).read().split()
    #print(len(datatemp))
    data=data+datatemp
data=list(map(float,data))
x=np.array([data[i] for i in range(0,len(data),3)])
y=np.array([data[i] for i in range(1,len(data),3)])
t=np.array([data[i] for i in range(2,len(data),3)])
#for i in range(len(t)):
#    if t[i]==1499:
#        t[i]=500

#pl.scatter(x,y,color='k')
pl.plot(t)
xy=list(zip(x,y,t/max(t)))
xy=np.array(sorted(xy,key=lambda x:x[0]))
xi=np.linspace(0,1,1000)
yi=np.linspace(-1,1,1000)
zi=griddata((xy.T[0],xy.T[1]),xy.T[2],(xi[None,:],yi[:,None]),method='linear')
vi=np.linspace(-0.01,1,10,endpoint=True)
pl.contourf(xi,yi,zi,vi,cmap='jet')
v=np.linspace(0,1,11,endpoint=True)
pl.colorbar(ticks=v,label='escape time',extend='both')
pl.title('ETD (cut width=%.2f, cut angle=%.1f'u'\N{DEGREE SIGN})' % (focdist,cut_angle*180/np.pi))
pl.xlim(0.3,0.7)
pl.ylim(-1,1)
pl.xlabel('$\eta$')
pl.ylabel('sin($\chi$)')
pl.savefig(os.getcwd()+'\\ETD_Temp\\focdist=%.2f,cut_angle=%.1f' % (focdist,cut_angle*180/np.pi)+',ETD.png',dpi=600)