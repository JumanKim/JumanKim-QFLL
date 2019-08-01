# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 19:22:20 2017

@author: user
"""
from matplotlib import pyplot as plt
import numpy as np
import lmfit
from lmfit import  Model
import os
import pandas as pd
#Nonlinear_fitting: https://lmfit.github.io/lmfit-py/model.html
#Bubble sort: http://interactivepython.org/runestone/static/pythonds/SortSearch/TheBubbleSort.html
fig=plt.figure()
fig.set_dpi(600)

def bubbleSort(alist):
    for passnum in range(len(alist)-1,0,-1):
        for i in range(passnum):
            if alist[i]>alist[i+1]:
                temp = alist[i]
                alist[i] = alist[i+1]
                alist[i+1] = temp
def exponential(x,amp,decay_time,yoffset,xoffset):
    return amp*np.exp(-(x)/decay_time)

def log(x):
     if x>0:
          return np.log(x)
     else:
          return -1e10
def bent_linear(x,a,b,c,d,p1,p2):
     return np.piecewise(x,[x<p1,(x>=p1) & (x<p2),p2<=x],[lambda x:a*x+b,lambda x:c*x+(a-c)*p1+b,lambda x:d*x+(c-d)*p2+(a-c)*p1+b])
def linear(x,a,b):
     return a*x+b
def plot_decay(path):
     fp=open(path+"decay_rate.txt",'w')
     for n in range(1,8):
          focdist=n*0.1
          data=np.loadtxt('focdist='+str(focdist)+','+'point_decay.txt')
          x=data[:,0]
          y=[(899-i)/900. for i in range(len(x))] #한 step 당 한개씩 빠져나감
          #print y
          bubbleSort(x)
          #newx=[x[0]]
          #newy=[y[0]]
          newx=[]
          newy=[]
          #동일 시간에 빠져나가는 거 모두 합함
          for i in range(1,len(x)):
               if log(x[i])>-11:
                    if x[i]!=x[i-1]:
                      newx.append(log(x[i]))
                      newy.append(log(y[i]))
          print(len(newx),len(newy))



          gmodel=Model(linear)
          #result=gmodel.fit(newy, x=newx, amp=0.5,decay_rate=-1000,xoffset=0.0,yoffset=0.0)
          result=gmodel.fit(newy,x=newx,a=-3,b=0)
          print(result.fit_report())

          exec('p'+str(n)+",=plt.plot(newx,newy,'C'+str(n-1))")
          plt.plot(newx, result.best_fit,'r-')
          fp.write("%.1f %f\n" % (focdist,result.params.valuesdict()['a']))
     plt.title("log-log plot of decay")
     plt.xlabel('log(time) (log(s))')
     plt.ylabel('log(fraction)')
     #plt.xlim(0,1)
     #plt.ylim(-1.0,1.0)
     plt.autoscale(enable='True',axis='both')
     plt.legend([p1,p2,p3,p4,p5,p6,p7],['0.1 mm','0.2 mm','0.3 mm','0.4 mm','0.5 mm','0.6 mm','0.7 mm'],loc='best')
     plt.savefig(path+'focdist'+','+'Decay graph_log_sub.png')
     fp.close()
#plot_decay('\\\\147.46.50.37\\sharing\\KJMAN\\Penrose cavity progress\\penrose cavity\\cleaved\\')
def decay_rate(path):
    pls=[]
    pfs=[]
    plabels=[]
    cut_angle_list=[]
    cut_width_list=[]
    iteration_list=[]
#    writer=pd.ExcelWriter('Penrose_decay_data.xlsx')
    for n in range(1,6):
        cut_angle=n*np.pi/2./5.
        data=open(os.getcwd()+"\\cut_angle=%.1f" % (cut_angle*180/np.pi)+",decay_rate.txt").read().split()
        data=list(map(float,data))
        x=np.array([data[i] for i in range(0,len(data),3)])
        y=np.array([data[i] for i in range(1,len(data),3)])
        xy=list(zip(x,y))
        xy=np.array(sorted(xy,key=lambda x:x[0]))
        gmodel=Model(exponential)
        result=gmodel.fit(xy.T[1],x=xy.T[0],amp=500.,decay_time=0.1,yoffset=0.,xoffset=0.)
        print(result.fit_report())
        p,=plt.plot(xy.T[0],xy.T[1],'C'+str(n-1)+'o-')
        pls.append(p)
        pf,=plt.plot(xy.T[0],result.best_fit,'C'+str(n-1))
        pfs.append(pf)
        plabels.append('%.0f$^\circ$: f(x)=%.1f*Exp((x-(%.1e))/%.1e)+(%.1e)' % (cut_angle*180/np.pi,result.params.valuesdict()['amp'],result.params.valuesdict()['xoffset'],result.params.valuesdict()['decay_time'],result.params.valuesdict()['xoffset']))
        cut_angle_list.append(cut_angle*180/np.pi)
        cut_width_list.append(xy.T[0])
        iteration_list.append(xy.T[1])
#        cut_angle_list=np.array(cut_angle_list)
#        cut_width_list=np.array(cut_width_list)
#        iteration_list=np.array(iteration_list)    
        Excelexport={'cut width': xy.T[0], 'number of iterations': xy.T[1]}
        df = pd.DataFrame(Excelexport)
        export_excel=df.to_excel(os.getcwd()+r'\\Barrier_Temp\\cut_angle=%.1f,Penrose_decay_data.xlsx'% (cut_angle*180/np.pi), index = None, header=True)
    plt.title("cut angle=18,36,54,72,90"u'\N{DEGREE SIGN}')
    plt.yscale('log')
    #plt.ylim(10,1000)
    plt.xlim(0,0.35)
    plt.xlabel("cut width (mm)")
    plt.ylabel("number of interactions")
    plt.legend(pls,['cut angle:18','36','54','72','90'],loc='best')
    print(path)
    plt.savefig(path+"\\Decay_rate.png",dpi=600)
decay_rate(os.getcwd())
#decay_rate('\\\\147.46.50.37\\sharing\\KJMAN\\Penrose cavity progress\\penrose cavity\\cleaved\\')